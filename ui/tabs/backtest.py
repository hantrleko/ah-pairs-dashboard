"""回测。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ah_pairs.backtest import later_window_performance, params_to_json
from ah_pairs.bundle import research_zip
from ah_pairs.export import equity_png, metrics_latex, metrics_table, spread_png
from ah_pairs.i18n import tx
from ah_pairs.labels import cost_case_label, walk_row_label
from ah_pairs.plots import equity_figure
from ui.cache import cached_costs, cached_sharpe, cached_walk
from ui.format import kpis, meta, num, pct, show_chart, show_table, tone


def _rename_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    show = frame.copy()
    if "role" in show.columns:
        folds = show["fold"] if "fold" in show.columns else ""
        show["区间"] = [
            walk_row_label(str(role), str(fold), str(fallback))
            for role, fold, fallback in zip(show["role"], folds, show["区间"])
        ]
    columns = {
        "区间": tx("区间", "Slice"),
        "开仓 |z|": tx("开仓 |z|", "Entry |z|"),
        "平仓 |z|": tx("平仓 |z|", "Exit |z|"),
        "净收益": tx("净收益", "Net"),
        "夏普": tx("夏普", "Sharpe"),
        "最大回撤": tx("最大回撤", "Max drawdown"),
        "交易次数": tx("交易次数", "Trades"),
        "成本合计": tx("成本合计", "Costs"),
    }
    keep = [name for name in columns if name in show.columns]
    return show[keep].rename(columns=columns)


def _cost_display(table: pd.DataFrame) -> pd.DataFrame:
    show = table.copy()
    if "code" in show.columns:
        show["情形"] = show["code"].map(cost_case_label)
    columns = {
        "情形": tx("情形", "Case"),
        "净收益": tx("净收益", "Net"),
        "夏普": tx("夏普", "Sharpe"),
        "交易次数": tx("交易次数", "Trades"),
        "成本合计": tx("成本合计", "Costs"),
        "融券费合计": tx("融券费合计", "Borrow"),
    }
    keep = [name for name in columns if name in show.columns]
    return show[keep].rename(columns=columns)


def _number_config(frame: pd.DataFrame) -> dict:
    config = {}
    percent = {tx("净收益", "Net"), tx("最大回撤", "Max drawdown"), tx("成本合计", "Costs"), tx("融券费合计", "Borrow")}
    for column in frame.columns:
        if column in percent:
            config[column] = st.column_config.NumberColumn(format="%.2%")
        elif column in {tx("夏普", "Sharpe")}:
            config[column] = st.column_config.NumberColumn(format="%.2f")
        elif column in {tx("开仓 |z|", "Entry |z|"), tx("平仓 |z|", "Exit |z|")}:
            config[column] = st.column_config.NumberColumn(format="%.1f")
        elif column == tx("交易次数", "Trades"):
            config[column] = st.column_config.NumberColumn(format="%d")
    return config


@st.fragment
def _bundle_downloads(page, walk_table) -> None:
    params = page.params
    ready_key = f"{page.picked.a_code}|{params_to_json(params)}|{page.start_day}|{page.end_day}"
    if st.button(tx("生成研究包和图片", "Build the bundle and figures"), key="build_bundle"):
        extra = {"walk_forward": walk_table} if isinstance(walk_table, pd.DataFrame) else None
        st.session_state["export_ready"] = ready_key
        st.session_state["export_zip"] = research_zip(
            page.result,
            page.picked["name"],
            page.picked.a_code,
            page.picked.h_code,
            params.z_window,
            extra=extra,
        )
        st.session_state["export_spread"] = spread_png(page.frame, params.z_window)
        st.session_state["export_equity"] = equity_png(page.frame)
    if st.session_state.get("export_ready") != ready_key:
        st.caption(tx("研究包和 PNG 在按下按钮后生成。", "The bundle and PNG files are built when you press the button."))
        return
    blob = st.session_state.get("export_zip")
    if blob:
        st.download_button(tx("研究包 ZIP", "Research bundle ZIP"), blob, file_name=f"{page.picked.a_code}_research.zip", mime="application/zip")
    spread_bytes = st.session_state.get("export_spread")
    equity_bytes = st.session_state.get("export_equity")
    if spread_bytes and equity_bytes:
        p1, p2 = st.columns(2)
        p1.download_button(tx("价差图 PNG", "Spread PNG"), spread_bytes, file_name=f"{page.picked.a_code}_spread.png", mime="image/png", key="spread_png_backtest")
        p2.download_button(tx("净值图 PNG", "Equity PNG"), equity_bytes, file_name=f"{page.picked.a_code}_equity.png", mime="image/png", key="equity_png_backtest")


def render_backtest(page) -> None:
    stats = page.stats
    params = page.params
    frame = page.frame
    kpis(
        [
            (tx("年化收益", "CAGR"), pct(stats["cagr"]), tone(stats["cagr"])),
            (tx("年化波动", "Volatility"), pct(stats["vol"]), ""),
            (tx("成本合计", "Costs"), pct(stats["total_cost"]), ""),
            (tx("融券费合计", "Borrow"), pct(stats["total_borrow"]), ""),
        ],
        columns=4,
    )
    values = tuple(float(item) for item in frame["ret_strat"].to_numpy(dtype=float) if np.isfinite(item))
    n_trials = int(len(page.result.get("sensitivity"))) if isinstance(page.result.get("sensitivity"), pd.DataFrame) else 1
    ann = float(stats.get("bars_per_year") or 252.0)
    boot, prob = cached_sharpe(values, max(n_trials, 1), ann)
    if boot.get("ok"):
        st.markdown(
            tx(
                f"夏普 **{num(stats['sharpe'], 2)}**，区块自助法 95% 区间 **[{num(boot['low'], 2)}, {num(boot['high'], 2)}]**（块长 {boot['block']} 日，{boot['n_boot']} 次，按每年 {ann:.0f} 根 K 线年化）。",
                f"Sharpe **{num(stats['sharpe'], 2)}**, block-bootstrap 95% interval **[{num(boot['low'], 2)}, {num(boot['high'], 2)}]** (block {boot['block']} days, {boot['n_boot']} draws, annualized with {ann:.0f} bars per year).",
            )
        )
        if np.isfinite(boot["low"]) and boot["low"] < 0 < boot["high"]:
            st.caption(tx("区间跨过 0。这组样本不能把夏普和 0 分开。", "The interval crosses zero. This sample does not separate the Sharpe ratio from zero."))
    if np.isfinite(stats.get("sharpe_common", np.nan)):
        st.caption(
            tx(
                f"只在同时开市日上计算的夏普是 {num(stats['sharpe_common'], 2)}。单边开市日的零收益不再混进这一条。",
                f"The Sharpe ratio on common sessions only is {num(stats['sharpe_common'], 2)}. Zero returns on one-sided days are left out of that figure.",
            )
        )
    if prob.get("ok"):
        st.caption(
            tx(
                f"按日收益估计，夏普大于 0 的概率（PSR）是 {prob['psr']:.2f}。若把侧栏夏普看成 {prob['n_trials']} 个开仓阈值里挑出来的，压缩夏普概率是 {prob['dsr']:.2f}。两者都不是未来赚钱的概率。主结果并没有自动改用网格里最好的阈值。",
                f"The probabilistic Sharpe ratio against zero is {prob['psr']:.2f}. If this Sharpe were the best of {prob['n_trials']} entry thresholds, the deflated Sharpe probability is {prob['dsr']:.2f}. Neither is the probability of future profit. The headline result was not replaced by the best cell of the grid.",
            )
        )
    token = params_to_json(params)
    wf_key = f"{page.picked.a_code}|{page.start_day}|{page.end_day}|{token}|{len(frame)}"
    st.markdown("**Walk-forward**")
    if st.button(tx("计算 walk-forward", "Compute the walk-forward"), key="run_walk"):
        st.session_state["wf_key"] = wf_key
    wf = {"ok": False}
    if st.session_state.get("wf_key") == wf_key:
        with st.spinner(tx("计算 walk-forward…", "Computing the walk-forward…")):
            wf = cached_walk(token, frame)
    else:
        st.info(tx("这一步要试很多组阈值。点按钮后才计算，换参数后需要再点一次。", "This step searches many thresholds. It runs when you press the button, and again after the parameters change."))
    oos = wf.get("oos_equity") if wf.get("ok") else None
    show_chart(equity_figure(page.view, oos=oos if isinstance(oos, pd.Series) else None), "chart_equity")
    oos_note = tx("绿线是各折验证段接起来的样本外净值。", "The green line stitches the validation slices. It is the out-of-sample path.") if isinstance(oos, pd.Series) and len(oos) else ""
    st.caption(meta(tx("策略净值和对照", "Strategy and benchmarks"), page.view.index, page.cache_stamp, tx("只持有 A、只持有 H、等权 A+H 用的是同一段日期。", "A-only, H-only and equal-weight A+H use the same dates.") + oos_note))
    short_text = tx("本回测允许做空 A 股，多数标的不一定能融到券，结果偏乐观。", "This backtest may short A-shares. Many names cannot be borrowed, so the result is optimistic.") if params.allow_short_a else tx("本回测不允许做空 A 股。A 股偏贵时不开仓。", "This backtest does not short A-shares. It stays flat when A is rich.")
    st.caption(short_text)
    one_sided = page.result.get("n_one_sided", stats.get("n_one_sided", 0))
    common = page.result.get("n_common", stats.get("n_common", stats["n_days"]))
    show_table(
        pd.DataFrame(
            {
                tx("项目", "Item"): [
                    tx("同时开市", "Common sessions"),
                    tx("单边开市", "One-sided sessions"),
                    tx("日历年数", "Calendar years"),
                    tx("总收益", "Total return"),
                    tx("夏普", "Sharpe"),
                    tx("交易", "Trades"),
                ],
                tx("怎么算", "Definition"): [
                    f"{common}",
                    tx(f"{one_sided}。这些日子不新开仓、不调 β，已有仓位计入开市一边。", f"{one_sided}. No new trades or beta changes. An open position earns the side that is open."),
                    tx(
                        f"{num(stats.get('years'), 2)} 年，跨度 {stats.get('span_days', '—')} 天，每年约 {ann:.0f} 根 K 线。年化不用固定的 252。",
                        f"{num(stats.get('years'), 2)} years, span {stats.get('span_days', '—')} days, about {ann:.0f} bars a year. Annualization is not a fixed 252.",
                    ),
                    tx("期末净值 − 1。初始净值是 1，不是一笔货币资金。", "Ending equity minus 1. Starting equity is 1, not a cash amount."),
                    tx("日收益均值 / 标准差 × 每年 K 线数的平方根，无风险利率按 0。", "Mean daily return / standard deviation × square root of bars per year. The risk-free rate is zero."),
                    tx(f"已平仓 {stats['n_trades']} 次。{'已扣成本。' if page.include_costs else '未扣成本。'}{'已按 T+1。' if params.fill == 'next_close' else '未按 T+1，比实盘乐观。'}", f"{stats['n_trades']} closed trades. {'Costs on.' if page.include_costs else 'Costs off.'} {'T+1 on.' if params.fill == 'next_close' else 'Same-close fill, which is optimistic.'}"),
                ],
            }
        )
    )
    cost_rows = pd.DataFrame(
        {
            tx("项目", "Item"): [
                tx("A 股佣金", "A commission"),
                tx("A 股印花税", "A stamp"),
                tx("H 股佣金", "H commission"),
                tx("H 股印花税", "H stamp"),
                tx("滑点", "Slippage"),
                tx("开平仓成本", "Open and close"),
                tx("β 再平衡成本", "Beta rebalance"),
                tx("融券费", "Borrow"),
            ],
            tx("占初始净值", "Share of starting equity"): [
                stats.get("cost_a_commission", 0.0),
                stats.get("cost_a_stamp", 0.0),
                stats.get("cost_h_commission", 0.0),
                stats.get("cost_h_stamp", 0.0),
                stats.get("cost_slippage", 0.0),
                stats.get("cost_turnover", 0.0),
                stats.get("cost_rebalance", 0.0),
                stats.get("total_borrow", 0.0),
            ],
        }
    )
    st.markdown(tx("**成本拆开**", "**Cost breakdown**"))
    show_table(cost_rows, {tx("占初始净值", "Share of starting equity"): st.column_config.NumberColumn(format="%.2%")})
    st.caption(tx("开平仓是方向变化那天的佣金、印花税和滑点。β 再平衡是方向没变、只因对冲比率移动产生的成本。融券费按持仓另行计提，一年仍按 252 天。", "Open and close is commission, stamp and slippage on days the direction changes. Beta rebalance is the cost when the direction stays and only the hedge ratio moves. Borrow accrues separately, still using 252 days a year."))
    later = later_window_performance(frame, page.result["trades"])
    if later.get("ok"):
        st.caption(
            tx(
                f"后半段 {pd.Timestamp(later['start']).date()} 至 {pd.Timestamp(later['end']).date()} 沿用同一套参数：净收益 {pct(later['total_return'])}。这不是 walk-forward。",
                f"The second half, {pd.Timestamp(later['start']).date()} to {pd.Timestamp(later['end']).date()}, keeps the same parameters: net {pct(later['total_return'])}. That is not a walk-forward.",
            )
        )
    if wf.get("ok"):
        if wf.get("scheme") == "anchored-year":
            st.markdown(
                tx(
                    f"按年扩展训练、下一年验证，共 **{wf['n_folds']}** 折。每一折在训练段同时选开仓和平仓阈值，验证段锁定。验证夏普中位数 **{num(wf['valid_sharpe_median'], 2)}**，范围 {num(wf['valid_sharpe_min'], 2)} 到 {num(wf['valid_sharpe_max'], 2)}。训练段一共试了 {wf['n_trials']} 个阈值组合。",
                    f"Anchored yearly folds: **{wf['n_folds']}**. Each fold picks both entry and exit on the training years and locks them on the next year. Median validation Sharpe **{num(wf['valid_sharpe_median'], 2)}**, from {num(wf['valid_sharpe_min'], 2)} to {num(wf['valid_sharpe_max'], 2)}. The search tried {wf['n_trials']} threshold pairs.",
                )
            )
        else:
            st.markdown(
                tx(
                    f"样本不够按年分折，所以只切一刀。训练到 **{pd.Timestamp(wf['train_end']).date()}**，选中的开仓阈值是 **{wf['chosen_entry']:.1f}**、平仓阈值是 **{wf['chosen_exit']:.1f}**。验证段 {pd.Timestamp(wf['valid_start']).date()} 至 {pd.Timestamp(wf['valid_end']).date()} 锁定这两个数。",
                    f"The sample is too short for yearly folds, so there is one split. Training ends **{pd.Timestamp(wf['train_end']).date()}** and selects entry **{wf['chosen_entry']:.1f}** and exit **{wf['chosen_exit']:.1f}**. Validation from {pd.Timestamp(wf['valid_start']).date()} to {pd.Timestamp(wf['valid_end']).date()} locks both.",
                )
            )
        walked = _rename_metrics(wf["table"])
        show_table(walked, _number_config(walked))
        st.caption(tx("验证段没有参与挑选。上面的绿线把这些验证段按时间接上，全样本行仍只是把参数再放到历史上。", "The validation slice did not choose the thresholds. The green line joins those slices in time. A full-sample row only replays parameters on history."))
    elif st.session_state.get("wf_key") == wf_key:
        st.info(wf.get("message") or tx("这段样本做不了 walk-forward。", "This sample cannot support a walk-forward."))
    st.markdown(tx("**成本敏感度**", "**Cost sensitivity**"))
    costs = _cost_display(cached_costs(token, frame))
    show_table(costs, _number_config(costs))
    sensitivity = page.result.get("sensitivity")
    if isinstance(sensitivity, pd.DataFrame) and not sensitivity.empty:
        st.markdown(tx("**开仓阈值敏感性**", "**Entry-threshold sensitivity**"))
        show = sensitivity.copy()
        show[tx("当前", "Current")] = show["current"].map({True: tx("是", "yes"), False: ""})
        renamed = show.rename(
            columns={
                "entry_z": tx("开仓 |z|", "Entry |z|"),
                "sharpe": tx("夏普", "Sharpe"),
                "max_drawdown": tx("最大回撤", "Max drawdown"),
                "total_return": tx("净收益", "Net"),
                "n_trades": tx("交易次数", "Trades"),
            }
        )
        keep = [tx("开仓 |z|", "Entry |z|"), tx("夏普", "Sharpe"), tx("最大回撤", "Max drawdown"), tx("净收益", "Net"), tx("交易次数", "Trades"), tx("当前", "Current")]
        show_table(renamed[keep], _number_config(renamed))
        st.caption(tx("同一段样本试了多次。最小的 p 值或最好的夏普会被挑得偏乐观。", "The same sample is tried many times. The best Sharpe in the grid is biased upward."))
    trades = page.result["trades"]
    if trades.empty:
        st.info(tx("这个参数组合在评估区间里没有开仓。", "This parameter set did not open a trade."))
    else:
        show = trades.copy()
        show[tx("方向", "Side")] = show["direction"].map({1: tx("做多价差", "Long spread"), -1: tx("做空价差", "Short spread")})
        show[tx("状态", "Status")] = show["closed"].map({True: tx("已平仓", "Closed"), False: tx("持仓中", "Open")})
        show[tx("开仓", "Entry")] = pd.to_datetime(show["entry_date"]).dt.date
        show[tx("平仓", "Exit")] = pd.to_datetime(show["exit_date"]).dt.date
        show_table(show.rename(columns={"entry_z": "z0", "exit_z": "z1", "holding_days": tx("持有", "Days"), "trade_return": tx("收益", "Return")})[[tx("开仓", "Entry"), tx("平仓", "Exit"), tx("方向", "Side"), "z0", "z1", tx("持有", "Days"), tx("收益", "Return"), tx("状态", "Status")]])
    summary = metrics_table(page.result, page.picked["name"], page.picked.a_code, page.picked.h_code)
    d1, d2, d3 = st.columns(3)
    d1.download_button(tx("指标 CSV", "Metrics CSV"), summary.to_csv(index=False).encode("utf-8-sig"), file_name=f"{page.picked.a_code}_metrics.csv", mime="text/csv")
    d2.download_button(tx("指标 LaTeX", "Metrics LaTeX"), metrics_latex(summary).encode("utf-8"), file_name=f"{page.picked.a_code}_metrics.tex", mime="text/plain")
    d3.download_button(
        tx("交易明细 CSV（完整精度）", "Trades CSV (full precision)"),
        (trades.to_csv(index=False).encode("utf-8-sig") if not trades.empty else b"entry_date,exit_date\n"),
        file_name=f"{page.picked.a_code}_trades.csv",
        mime="text/csv",
    )
    _bundle_downloads(page, wf.get("table") if wf.get("ok") else None)
