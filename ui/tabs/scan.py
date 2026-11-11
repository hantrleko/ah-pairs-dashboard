"""一组对照和离线全池扫描。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ah_pairs.analysis import analyze_frame
from ah_pairs.cross_section import load_local_panels, monthly_cheap_book
from ah_pairs.i18n import tx
from ah_pairs.portfolio import portfolio_summary
from ah_pairs.scan import SCAN_SPEC_DEFAULT, read_scan, scan_is_older
from ah_pairs.universe import liquid_subset
from ui.cache import cached_pair
from ui.format import follow_pair_selection, pct, rank_lines, render_quick_compare, show_table


def _portfolio_caption(summary: dict) -> None:
    if not summary.get("ok"):
        st.caption(summary.get("message") or tx("等风险组合没有算出。", "No equal-risk book."))
        return
    st.markdown(
        tx(
            f"等风险组合：**{summary['n_pairs']}** 对，{summary['n_days']} 天，净收益 {pct(summary['total_return'])}，夏普 {summary['sharpe']:.2f}，最大回撤 {pct(summary['max_drawdown'])}。权重是过去 60 日策略波动的倒数，用到的是前一天的波动。这不是无风险组合，也不是持仓建议。",
            f"Equal-risk book: **{summary['n_pairs']}** pairs, {summary['n_days']} days, net {pct(summary['total_return'])}, Sharpe {summary['sharpe']:.2f}, max drawdown {pct(summary['max_drawdown'])}. Weights are inverse 60-day strategy volatility, lagged one day. This is not risk-free and not a position recommendation.",
        )
    )


def render_scan(page) -> None:
    st.markdown(
        tx(
            "上面四只和下面大约十二只是现算。全部股票池不要在页面里跑：用 `python -m ah_pairs.scan --all` 写到缓存目录的 `scan/`，或放到仓库的 `data/scan_snapshot/`。这页读较新的那一份。点表格里的一行，就打开那一对。",
            "The four names and the dozen below are computed here. Do not scan the full universe in the browser. `python -m ah_pairs.scan --all` writes scan/ in the cache, or data/scan_snapshot/ in the repo. This page reads the newer file. Select a row to open that pair.",
        )
    )
    render_quick_compare(page, "compare_from_scan")
    saved_summary, saved_returns = read_scan()
    if isinstance(saved_summary, pd.DataFrame) and not saved_summary.empty:
        st.markdown(tx("**已保存的离线扫描**", "**Saved offline scan**"))
        asof = str(saved_summary["asof"].iloc[0]) if "asof" in saved_summary.columns else ""
        sample_start = str(saved_summary["start"].iloc[0]) if "start" in saved_summary.columns else ""
        sample_end = str(saved_summary["end"].iloc[0]) if "end" in saved_summary.columns else ""
        spec = str(saved_summary["spec"].iloc[0]) if "spec" in saved_summary.columns else ""
        if spec == SCAN_SPEC_DEFAULT:
            spec_text = tx("不做空，法定成本，次日收盘。", "Short A-shares off, statutory costs, next-close fill.")
        else:
            spec_text = spec
        st.caption(
            tx(
                f"生成日 {asof}。样本 {sample_start} 至 {sample_end}。{spec_text}这是一次离线描述，不是当日强弱排行。",
                f"As of {asof}. Sample {sample_start} to {sample_end}. {spec_text} This is one offline description, not a live ranking.",
            )
        )
        if scan_is_older(asof, getattr(page, "cache_stamp", "")):
            st.caption(tx("扫描早于当前行情。", "The scan is older than the current prices."))
        st.caption(tx("点一行就切换到那一对。p 值更小只表示这次离线样本更接近协整，不是最协整的发现。", "Select a row to open that pair. A smaller p-value only means this offline sample is closer to cointegration."))
        hidden = [name for name in ("asof", "spec", "start", "end") if name in saved_summary.columns]
        follow_pair_selection(show_table(saved_summary.drop(columns=hidden), key="saved_scan_table", selectable=True), saved_summary)
        rank_lines(saved_summary)
        if isinstance(saved_returns, pd.DataFrame) and not saved_returns.empty:
            _portfolio_caption(portfolio_summary(saved_returns))
    else:
        st.caption(tx("还没有离线扫描文件。", "No offline scan file yet."))
    names = liquid_subset(page.universe).head(12)
    st.caption("、".join(names["name"].tolist()))
    if st.button(tx("扫描这组公司并做等风险组合", "Scan this group and build the equal-risk book")):
        rows = []
        returns = {}
        progress = st.progress(0.0, text=tx("准备扫描", "Preparing"))
        targets = list(names.itertuples(index=False))
        for i, item in enumerate(targets):
            progress.progress((i + 1) / len(targets), text=str(item.name))
            try:
                prices, _notes = cached_pair(item.a_code, item.h_code, page.warm, str(page.end_day))
                result = analyze_frame(prices, page.params, eval_start=pd.Timestamp(page.start_day), full_diagnostics=False)
                eg = result["eg"]
                stats = result["stats"]
                rows.append(
                    {
                        "公司": item.name,
                        "A": item.a_code,
                        "H": item.h_code,
                        "eg_pvalue": eg["pvalue"],
                        "sharpe": stats["sharpe"],
                        "total_return": stats["total_return"],
                        "max_drawdown": stats["max_drawdown"],
                        "n_trades": stats["n_trades"],
                        "total_cost": stats["total_cost"],
                    }
                )
                returns[item.name] = result["frame"]["ret_strat"]
            except Exception as exc:
                rows.append({"公司": item.name, "A": item.a_code, "H": item.h_code, "说明": str(exc)})
        progress.empty()
        st.session_state["scan_rows"] = rows
        st.session_state["scan_returns"] = pd.DataFrame(returns)
    if "scan_rows" in st.session_state:
        table = pd.DataFrame(st.session_state["scan_rows"])
        st.caption(tx("点一行就切换到那一对。", "Select a row to switch to that pair."))
        follow_pair_selection(
            show_table(
                table,
                {
                    "eg_pvalue": st.column_config.NumberColumn("EG p", format="%.3f"),
                    "sharpe": st.column_config.NumberColumn(tx("夏普", "Sharpe"), format="%.2f"),
                    "total_return": st.column_config.NumberColumn(tx("净收益", "Net"), format="%.2%"),
                    "max_drawdown": st.column_config.NumberColumn(tx("最大回撤", "Max drawdown"), format="%.2%"),
                    "n_trades": st.column_config.NumberColumn(tx("交易次数", "Trades"), format="%d"),
                    "total_cost": st.column_config.NumberColumn(tx("成本合计", "Costs"), format="%.2%"),
                },
                key="live_scan_table",
                selectable=True,
            ),
            table,
        )
        rank_lines(table)
        stored = st.session_state.get("scan_returns")
        if isinstance(stored, pd.DataFrame):
            _portfolio_caption(portfolio_summary(stored))
    _cross_section(page)


def _cross_section(page) -> None:
    st.markdown(tx("**月度截面**", "**Monthly cross-section**"))
    st.caption(
        tx(
            "每月用当时已经知道的溢价 z，做多 A 最便宜的四分之一，并做空这些名字的 H 股。不做空 A。只读本地缓存，没有文件的股票跳过，这里不下载。",
            "Each month, using only the premium z known at the time, go long the cheapest quartile of A-shares and short those names’ H-shares. A-shares are not shorted. This reads the local cache, skips names without a file, and does not download.",
        )
    )
    xs_key = f"{page.start_day}|{page.end_day}|{page.params.z_window}|{int(page.include_costs)}"
    if st.button(tx("计算月度截面", "Compute the monthly book"), key="run_cross_section"):
        names = liquid_subset(page.universe)
        with st.spinner(tx("只读本地行情…", "Reading local prices…")):
            panels, skipped = load_local_panels(names, page.start_day, page.end_day)
            result = monthly_cheap_book(panels, z_window=int(page.params.z_window), costs=page.params.costs)
        result["skipped"] = skipped
        st.session_state["xs_key"] = xs_key
        st.session_state["xs_result"] = result
    result = st.session_state.get("xs_result") if st.session_state.get("xs_key") == xs_key else None
    if not isinstance(result, dict):
        return
    if not result.get("ok"):
        st.info(result.get("message") or tx("月度截面没有结果。", "No monthly book."))
        if result.get("skipped"):
            st.caption(tx(f"跳过 {len(result['skipped'])} 只。", f"Skipped {len(result['skipped'])} names."))
        return
    st.markdown(
        tx(
            f"用了 **{result['n_names']}** 只本地行情，{result['n_months']} 个月，平均每月 {result['mean_book']:.1f} 只。净收益 {pct(result['total_return'])}，夏普 {result['sharpe']:.2f}，最大回撤 {pct(result['max_drawdown'])}，成本 {pct(result['total_cost'])}。",
            f"**{result['n_names']}** names from the local cache, {result['n_months']} months, {result['mean_book']:.1f} names a month on average. Net {pct(result['total_return'])}, Sharpe {result['sharpe']:.2f}, max drawdown {pct(result['max_drawdown'])}, costs {pct(result['total_cost'])}.",
        )
    )
    skipped = result.get("skipped") or []
    if skipped:
        shown = "、".join(skipped[:12])
        extra_zh = f" …共 {len(skipped)} 只" if len(skipped) > 12 else ""
        extra_en = f" …{len(skipped)} in total" if len(skipped) > 12 else ""
        st.caption(tx(f"这些代码没有本地行情，没有放进组合：{shown}{extra_zh}。", f"These codes had no local prices and were left out: {shown}{extra_en}."))
    latest = "、".join(result.get("latest_book") or [])
    signal_day = pd.Timestamp(result["latest_signal"]).date()
    st.caption(
        tx(
            f"最近一次信号在 {signal_day}，选中 {latest}。z 窗口 {result['z_window']} 日。换仓在下一个收盘，当天涨跌仍算旧组合。这不是上面的单对回测，也不是持仓建议。",
            f"The latest signal is {signal_day}: {latest}. The z window is {result['z_window']} days. The book changes at the next close, so that day’s move still belongs to the old book. This is not the single-pair backtest above, and it is not a position recommendation.",
        )
    )
