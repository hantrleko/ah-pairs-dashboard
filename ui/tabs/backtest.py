"""回测。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ah_pairs.backtest import later_window_performance, params_to_json
from ah_pairs.borrow import borrow_status, load_borrow
from ah_pairs.holding import holding_comparison
from ah_pairs.bundle import research_zip
from ah_pairs.data import load_h_dividends, read_h_dividends
from ah_pairs.desk import load_walk
from ah_pairs.dividends import h_dividend_tax_row
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


def _holding_block(page) -> None:
    table = holding_comparison(page.frame, page.params.costs)
    if table.empty:
        return
    labels = {
        "always_a": tx("永远持有 A", "Always hold A"),
        "always_h": tx("永远持有 H", "Always hold H"),
        "switch": tx("按溢价切换", "Switch on the premium"),
    }
    st.markdown(tx("**已经要持有这家公司**", "**If you were going to hold the company**"))
    show_table(
        pd.DataFrame(
            {
                tx("做法", "Choice"): [labels[code] for code in table["code"]],
                tx("净收益", "Net"): [pct(value) for value in table["total_return"]],
                tx("夏普", "Sharpe"): [num(value, 2) for value in table["sharpe"]],
                tx("最大回撤", "Max drawdown"): [pct(value) for value in table["max_drawdown"]],
                tx("成本", "Cost"): [pct(value) for value in table["total_cost"]],
                tx("换仓次数", "Switches"): [int(value) for value in table["switches"]],
            }
        )
    )
    st.caption(
        tx(
            "这三行不做空。z < 0 持有 A，z ≥ 0 持有 H。和主回测一样，收盘后才知道信号，下一次两边都开市的收盘才换，当天涨跌仍算旧仓位。永远持有的两行也扣了建仓成本。它不替换上面的配对回测。",
            "These three rows never short. Hold A when z < 0 and H when z ≥ 0. As in the main backtest, the signal is known after the close and the trade waits for the next close when both markets are open; that day’s move still belongs to the old holding. The two buy-and-hold rows also pay the entry cost. They do not replace the pairs backtest above.",
        )
    )


def _borrow_block(page) -> None:
    st.markdown(tx("**融券名单**", "**Borrow list**"))
    if st.button(tx("更新融券名单", "Update the borrow list"), key="refresh_borrow"):
        try:
            with st.spinner(tx("下载融券标的…", "Downloading the borrow list…")):
                load_borrow(refresh=True)
        except Exception as exc:
            st.info(str(exc))
        else:
            st.rerun()
    status = borrow_status(page.picked.a_code)
    if not status.get("ok"):
        st.caption(tx("还没有融券名单。点上面的按钮才会下载。打开页面不下载。默认不做空，所以名单不会改上面的数字。", "There is no borrow list yet. The button downloads it. Opening the page does not. Shorting stays off by default, so the list does not change the figures above."))
        return
    ratio = status.get("margin_ratio")
    ratio_text = "—" if ratio is None or not np.isfinite(ratio) else f"{ratio:.0%}"
    if status["eligible"]:
        line = tx(
            f"名单日期 {status['asof']}，共 {status['n_names']} 只。{page.picked['name']} 在名单上，保证金比例 {ratio_text}。",
            f"List date {status['asof']}, {status['n_names']} names. {page.picked['name']} is on it. Margin ratio {ratio_text}.",
        )
    else:
        line = tx(
            f"名单日期 {status['asof']}，共 {status['n_names']} 只。{page.picked['name']} 不在名单上。",
            f"List date {status['asof']}, {status['n_names']} names. {page.picked['name']} is not on it.",
        )
    st.caption(line)
    st.caption(
        tx(
            "保证金比例不是融券年化费率，回测费率仍用侧栏。名单只覆盖下载到的日期。打开做空之后，这些日子里不在名单上就不能做空 A；更早的日子仍按开关，不会把今天的资格套到整段历史。默认关闭时，名单不改上面的数字。",
            "The margin ratio is not the annual borrow fee. The backtest still uses the sidebar fee. The list covers only the dates that were downloaded. After shorting is turned on, a name off the list cannot be shorted on those dates. Earlier dates still follow the switch. Today’s eligibility is not pasted onto the whole history. While the switch stays off, the list does not change the figures above.",
        )
    )


def render_backtest(page) -> None:
    if getattr(page, "engine", "v14") == "v15":
        from ah_pairs.v15.report import result_key
        from ah_pairs.v15.walkforward import write_folds, walk_forward

        key = result_key(page.picked.a_code, page.start_day, page.end_day, page.params.allow_short_a)
        st.markdown(tx("**V1.5 样本外**", "**V1.5 out of sample**"))
        st.caption(tx("这一步只在按下按钮后运行。它不改 v14 的回测，结果写到 output/wf_v15/。折数不足 4 时，下面的表不能当成稳定结论。", "This runs only after the button. It does not change the v14 backtest. Results go to output/wf_v15/. Fewer than 4 folds is not a stable claim."))
        if st.button(tx("计算 V1.5 样本外折", "Run the V1.5 out-of-sample folds"), key="run_v15_wf"):
            with st.spinner(tx("按折选择参数并做 Reality Check…", "Choosing parameters by fold and running the reality check…")):
                try:
                    found = walk_forward(page.frame, allow_short_a=bool(page.params.allow_short_a), seed=7)
                    target = Path("output") / "wf_v15" / f"{page.picked.a_code}_{page.picked.h_code}.parquet"
                    if isinstance(found.get("folds"), pd.DataFrame) and len(found["folds"]):
                        write_folds(found["folds"], target)
                    st.session_state[key] = found
                    st.session_state.pop("v15_error", None)
                    st.rerun()
                except Exception as exc:
                    st.session_state["v15_error"] = str(exc)
        if st.session_state.get("v15_error") and key not in st.session_state:
            st.caption(tx(f"V1.5 没有算完：{st.session_state['v15_error']}", f"V1.5 did not finish: {st.session_state['v15_error']}"))
        found = st.session_state.get(key)
        if isinstance(found, dict) and isinstance(found.get("folds"), pd.DataFrame) and len(found["folds"]):
            show = found["folds"].copy()
            for column in ("test_start", "test_end"):
                show[column] = pd.to_datetime(show[column]).dt.date.astype(str)
            show_table(
                show,
                column_config={
                    "oos_ret": st.column_config.NumberColumn("oos_ret", format="%.2%"),
                    "oos_dd": st.column_config.NumberColumn("oos_dd", format="%.2%"),
                    "blocked_share": st.column_config.NumberColumn("blocked_share", format="%.0%"),
                    "oos_sharpe": st.column_config.NumberColumn("oos_sharpe", format="%.2f"),
                    "rc_p": st.column_config.NumberColumn("rc_p", format="%.3f"),
                    "spa_p": st.column_config.NumberColumn("spa_p", format="%.3f"),
                },
            )
            if bool(show["implausible"].any()) if "implausible" in show.columns else False:
                st.caption(tx("有一折样本外夏普高于 2。这格标成 implausible，先核对成本是不是漏了，不要直接采用。", "One fold has an out-of-sample Sharpe above 2. It is marked implausible. Check for a missing cost before using it."))
        st.caption(tx("下面的净值、成本和 walk-forward 仍是 v14。不要把它们读成 V1.5 的样本外结果。", "The equity, costs and walk-forward below are still v14. They are not the V1.5 out-of-sample result."))
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
        loaded = load_walk(page)
        if loaded is not None:
            wf = loaded
            st.caption(
                tx(
                    f"这是默认规格在 {loaded['asof']} 预计算的 walk-forward，样本到 {loaded['sample_end']}。结束日比这天晚不超过 10 天仍用这一份；更早的结束日不会用。对冲、窗口、开平仓、成本或起始日不同时，点按钮才重算，不会把这份结果当成新规格。",
                    f"This is the default-spec walk-forward precomputed on {loaded['asof']}, through {loaded['sample_end']}. An end date up to 10 days later still uses it. An earlier end date does not. A different hedge, window, entry, exit, cost or start date is recomputed from the button. This result is not treated as that new specification.",
                )
            )
        else:
            st.info(tx("这一步要试很多组阈值。点按钮后才计算，换参数后需要再点一次。", "This step searches many thresholds. It runs when you press the button, and again after the parameters change."))
    oos = wf.get("oos_equity") if wf.get("ok") else None
    show_chart(equity_figure(page.view, oos=oos if isinstance(oos, pd.Series) else None), "chart_equity")
    oos_note = tx("绿线是各折验证段接起来的样本外净值。", "The green line stitches the validation slices. It is the out-of-sample path.") if isinstance(oos, pd.Series) and len(oos) else ""
    st.caption(meta(tx("策略净值和对照", "Strategy and benchmarks"), page.view.index, page.cache_stamp, tx("只持有 A、只持有 H、等权 A+H 用的是同一段日期。", "A-only, H-only and equal-weight A+H use the same dates.") + oos_note))
    short_text = tx("本回测允许做空 A 股，多数标的不一定能融到券，结果偏乐观。", "This backtest may short A-shares. Many names cannot be borrowed, so the result is optimistic.") if params.allow_short_a else tx("本回测不允许做空 A 股。A 股偏贵时不开仓。", "This backtest does not short A-shares. It stays flat when A is rich.")
    st.caption(short_text)
    _holding_block(page)
    _borrow_block(page)
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
    div_key = f"{page.picked.h_code}|dividends"
    if st.button(tx("下载 H 股现金分红", "Download H-share cash dividends"), key="load_h_div"):
        try:
            with st.spinner(tx("下载 H 股现金分红…", "Downloading H-share cash dividends…")):
                st.session_state[div_key] = load_h_dividends(page.picked.h_code, refresh=True)
                st.session_state.pop("h_div_error", None)
        except Exception as exc:
            st.session_state["h_div_error"] = str(exc)
    if st.session_state.get("h_div_error") and div_key not in st.session_state:
        st.caption(tx(f"分红没有下载下来：{st.session_state['h_div_error']}", f"Dividend download failed: {st.session_state['h_div_error']}"))
    dividends = st.session_state.get(div_key)
    if dividends is None:
        dividends = read_h_dividends(page.picked.h_code)
    tax_row = h_dividend_tax_row(frame, dividends, page.result.get("trades"))
    costs = _cost_display(pd.concat([cached_costs(token, frame), tax_row.drop(columns=["status"])], ignore_index=True))
    show_table(costs, _number_config(costs))
    if dividends is None:
        st.caption(tx("还没有 H 股现金分红缓存。已发布副本里有的话会直接读。点上面的按钮才会下载。这一行先空着，主回测仍用复权价。", "No cached H-share cash dividends yet. A published copy is read when it exists. The button above is what downloads them. This row stays blank, and the main backtest still uses adjusted closes."))
    elif tax_row["status"].iloc[0] != "ok":
        st.caption(tx("没有分红序列。这一行不算。主回测仍用复权价。", "No dividend series. This row is not computed. The main backtest still uses adjusted closes."))
    else:
        st.caption(tx("这一行只在前一日持有 H 股多头时，按现金红利的 20% 预扣，并用当日汇率换成人民币。空仓和做空 H 股不扣。这是预扣近似，不是券商交割单。主回测没有改用这一行。", "This row withholds 20% of the cash dividend, converted at that day's exchange rate, only when the book was long H the day before. Flat and short-H days pay nothing. It is a withholding approximation, not a broker statement. The main backtest does not switch to this row."))
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
