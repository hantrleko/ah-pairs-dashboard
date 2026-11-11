"""协整诊断。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ah_pairs.breaks import gregory_hansen
from ah_pairs.i18n import tx
from ah_pairs.market import read_ah_premium_index
from ah_pairs.plots import rolling_p_figure, static_resid_figure
from ah_pairs.relative import relative_premium
from ui.format import half_life_text, meta, num, pct, pval, show_chart, show_table, span_frame


def render_diagnose(page) -> None:
    eg = page.eg
    johansen = page.johansen
    static_hl = page.static_hl
    traded_hl = page.traded_hl
    st.markdown(
        tx(
            "左边是整段评估区间里同时开市日的 Engle–Granger，回归用到了这些日子的全部价格，用来判断「这段时期是否协整」。交易仓位用的是另一套 β：滚动 OLS 或卡尔曼滤波，都只用当天以前的价格。",
            "The left panel is Engle–Granger on days when both markets were open. That regression uses every price in the window and only describes whether this sample looks cointegrated. The traded beta is a different object: rolling OLS or a Kalman filter, both using only prices before that day.",
        )
    )
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Engle–Granger**")
        adf = page.result.get("residual_adf") or {}
        show_table(
            pd.DataFrame(
                {
                    tx("项目", "Item"): [
                        tx("同时开市天数", "Common days"),
                        "α",
                        "β",
                        "R²",
                        tx("EG t 统计量", "EG t"),
                        tx("5% 临界值", "5% critical value"),
                        tx("EG p 值", "EG p-value"),
                        tx("残差 ADF 统计量", "Residual ADF stat"),
                        tx("残差 ADF p 值", "Residual ADF p"),
                        "φ",
                        tx("半衰期（交易日）", "Half-life (days)"),
                    ],
                    tx("数值", "Value"): [
                        f"{eg['nobs']}",
                        num(eg["alpha"], 3),
                        num(eg["beta"], 3),
                        num(eg["r2"], 3),
                        num(eg["t_stat"], 2),
                        num(eg["crit_5"], 2),
                        pval(eg["pvalue"]),
                        num(adf.get("adf_stat"), 2),
                        pval(adf.get("adf_pvalue")),
                        num(static_hl["phi"], 3),
                        half_life_text(static_hl),
                    ],
                }
            )
        )
        if adf.get("note"):
            st.caption(adf["note"] if page.lang == "zh" else tx(adf["note"], "Ordinary ADF on residuals from the same sample. The p-value is optimistic and must not replace the Engle–Granger p-value."))
    with d2:
        st.markdown(tx("**Johansen 迹检验**", "**Johansen trace test**"))
        if not johansen.get("ok"):
            st.info(johansen.get("message") or tx("Johansen 检验没有返回结果。", "Johansen did not return a result."))
        else:
            show_table(
                pd.DataFrame(
                    {
                        tx("项目", "Item"): [tx("差分滞后", "Diff lags"), tx("迹统计量（r=0）", "Trace (r=0)"), tx("5% 临界值", "5% critical value"), tx("5% 结论", "5% call")],
                        tx("数值", "Value"): [
                            f"{johansen['k_ar_diff']}",
                            num(johansen["trace_r0"], 2),
                            num(johansen["trace_crit_95"], 2),
                            tx("拒绝无协整", "reject no cointegration") if johansen["trace_reject_r0_95"] else tx("不拒绝", "do not reject"),
                        ],
                    }
                )
            )
        st.caption(
            tx(
                f"交易价差的半衰期是 {half_life_text(traded_hl)}。它和左边的全样本残差不是同一条序列。",
                f"Half-life of the traded spread: {half_life_text(traded_hl)}. That is not the full-sample residual on the left.",
            )
        )
    fit = page.result.get("kalman_fit") or {}
    if page.params.hedge == "kalman":
        reason = {
            "sidebar": tx("δ 用侧栏里的数。", "Delta is the sidebar value."),
            "warmup_short": tx("预热样本不够长，δ 仍用侧栏里的数。", "The warm-up is too short, so delta stays at the sidebar value."),
            "warmup": tx("δ 用评估区间之前的预热样本，按预测误差似然选取。", "Delta was chosen by prediction-error likelihood on the warm-up before the evaluation window."),
            "in_sample": tx("没有单独的预热区间，δ 用了评估样本本身，这一项不是样本外。", "There is no separate warm-up, so delta uses the evaluation sample and is not out of sample."),
        }.get(fit.get("reason"), "")
        delta = float(fit.get("delta", page.params.kalman_delta))
        st.caption(tx(f"卡尔曼 δ = {delta:.1e}。{reason}", f"Kalman delta = {delta:.1e}. {reason}"))
    rolling = page.result.get("rolling_p")
    if rolling is not None and len(rolling):
        shown = span_frame(rolling, page.chart_span)
        show_chart(rolling_p_figure(shown), "chart_rolling_p_diag")
        st.caption(meta(tx("滚动 Engle–Granger", "Rolling Engle–Granger"), shown.index, page.cache_stamp, tx("虚线是 0.05。每个点只用当时已经过去的 252 个同时开市日。", "The dotted line is 0.05. Each point uses the previous 252 common sessions.")))
    show_chart(static_resid_figure(page.view), "chart_resid_diag")
    st.markdown(tx("**相对市场溢价**", "**Premium versus the market**"))
    st.caption(
        tx(
            "用个股溢价减去恒生 AH 溢价指数。指数 100 当作平价，所以减的是指数/100 − 1。系数固定为 1，问的是这只股票相对市场的偏离会不会回到常数附近。",
            "Subtract the Hang Seng AH premium index from this stock’s premium. A reading of 100 is parity, so the index enters as level/100 − 1. The coefficient is fixed at 1. The question is whether this stock’s gap versus the market returns toward a constant.",
        )
    )
    relative = relative_premium(page.common["premium"], read_ah_premium_index())
    if not relative.get("ok"):
        st.info(relative.get("message") or tx("相对溢价没有结果。", "No relative-premium result."))
    else:
        show_table(
            pd.DataFrame(
                {
                    tx("项目", "Item"): [
                        tx("重叠天数", "Overlap"),
                        tx("ADF 统计量", "ADF stat"),
                        tx("5% 临界值", "5% critical value"),
                        tx("ADF p 值", "ADF p-value"),
                        tx("半衰期（交易日）", "Half-life (days)"),
                        tx("最近相对溢价", "Latest relative premium"),
                    ],
                    tx("数值", "Value"): [
                        f"{relative['nobs']}",
                        num(relative["adf_stat"], 2),
                        num(relative["crit_5"], 2),
                        pval(relative["adf_pvalue"]),
                        half_life_text({"half_life": relative["half_life"], "mean_reverting": relative["mean_reverting"]}),
                        pct(relative["latest"]),
                    ],
                }
            )
        )
        st.caption(
            tx(
                "这是相对溢价的普通 ADF，原假设是它有单位根。它不是 Engle–Granger，也不替换上面的 EG p 值。p 值不是获利概率。",
                "This is an ordinary ADF on the relative premium. The null is a unit root. It is not Engle–Granger and does not replace the EG p-value above. The p-value is not the probability of profit.",
            )
        )
    st.markdown(tx("**结构断点**", "**A single break**"))
    st.caption(
        tx(
            "Gregory–Hansen 允许协整关系变一次：模型 C 只改截距，模型 C/S 连斜率一起改。临界值来自 1996 年论文表 1 的 ADF*，一个回归元。模型 C 的 5% 是 −4.61。点按钮才搜索断点。",
            "Gregory–Hansen allows the cointegrating relation to change once. Model C shifts the intercept. Model C/S also shifts the slope. Critical values are the ADF* table in Gregory and Hansen (1996) for one regressor. Model C’s 5% value is −4.61. The search runs from the button.",
        )
    )
    gh_key = f"gh|{page.picked.a_code}|{page.start_day}|{page.end_day}"
    if st.button(tx("计算结构断点", "Compute the break test"), key="run_gh"):
        with st.spinner(tx("搜索断点…", "Searching for a break…")):
            st.session_state[gh_key] = gregory_hansen(page.common["log_a"], page.common["log_h"])
    gh = st.session_state.get(gh_key)
    if isinstance(gh, dict) and gh.get("ok"):
        show_table(
            pd.DataFrame(
                [
                    {
                        tx("模型", "Model"): tx(row["name"], row["model"]),
                        tx("ADF*", "ADF*"): num(row["stat"], 2),
                        tx("5% 临界值", "5% critical value"): num(row["crit_5"], 2),
                        tx("5% 结论", "5% call"): tx("拒绝无协整", "reject no cointegration") if row["reject_5"] else tx("不拒绝", "do not reject"),
                        tx("断点", "Break"): pd.Timestamp(row["break_date"]).date(),
                        tx("截距变化", "Intercept change"): num(row["level_shift"], 3),
                    }
                    for row in gh["models"]
                ]
            )
        )
        st.caption(
            tx(
                f"样本 {gh['nobs']} 天，两端各去掉 {gh['trim']:.0%}，滞后 {gh['lag']}。各断点用同一个滞后。拒绝只说明允许这一次断点后残差更像平稳，不替换 EG p 值。",
                f"Sample {gh['nobs']} days, {gh['trim']:.0%} trimmed at each end, lag {gh['lag']}. Every break uses that same lag. A rejection says the residuals look more stationary once this one break is allowed. It does not replace the EG p-value.",
            )
        )
    elif isinstance(gh, dict):
        st.info(gh.get("message") or tx("结构断点没有结果。", "No break-test result."))
