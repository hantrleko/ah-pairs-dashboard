"""稳定性。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ah_pairs.data import unadjusted_log_columns
from ah_pairs.i18n import tx
from ah_pairs.labels import slice_label
from ah_pairs.plots import hedge_figure, reject_rate_figure, rolling_p_figure, stability_heatmap, static_resid_figure
from ah_pairs.stability import eg_pvalue, window_count_facts
from ui.cache import cached_stability
from ui.format import half_life_text, kpis, meta, num, pct, pval, show_chart, show_table, span_frame


def render_stability(page) -> None:
    static_hl = page.static_hl
    st.markdown(tx("这一页看关系换窗口之后还在不在。地图上的检验不是交易信号。", "This page asks whether the relation survives a change of window. The map is not a trading signal."))
    st.caption(
        tx(
            f"全样本残差 φ = {num(static_hl.get('phi'), 3)}，半衰期 = {half_life_text(static_hl)}。交易用的是另一条因果价差。",
            f"Full-sample residual φ = {num(static_hl.get('phi'), 3)}, half-life = {half_life_text(static_hl)}. Trading uses a different causal spread.",
        )
    )
    adf = page.result.get("residual_adf") or {}
    st.caption(
        tx(
            f"残差 ADF 统计量 {num(adf.get('adf_stat'), 2)}，p 值 {pval(adf.get('adf_pvalue'))}。残差 ADF 不能代替 Engle–Granger p 值。",
            f"Residual ADF statistic {num(adf.get('adf_stat'), 2)}, p-value {pval(adf.get('adf_pvalue'))}. Residual ADF must not replace the Engle–Granger p-value.",
        )
    )
    rolling = page.result.get("rolling_p")
    if rolling is not None and len(rolling):
        shown = span_frame(rolling, page.chart_span)
        show_chart(rolling_p_figure(shown), "chart_rolling_p_stable")
        st.caption(meta(tx("滚动 Engle–Granger", "Rolling Engle–Granger"), shown.index, page.cache_stamp))
    show_chart(hedge_figure(page.view, active=page.params.hedge), "chart_hedge_stable")
    show_chart(static_resid_figure(page.view), "chart_resid_stable")
    common = page.common
    spec_key = f"{page.picked.a_code}|{page.start_day}|{page.end_day}|lag1"
    if st.session_state.get("spec_key") != spec_key:
        st.session_state["spec_lag1"] = eg_pvalue(common["log_a"], common["log_h"])
        st.session_state["spec_key"] = spec_key
    lag1 = st.session_state.get("spec_lag1", np.nan)
    st.markdown(
        tx(
            f"摘要里的 EG p = **{pval(page.eg['pvalue'])}**，滞后由 AIC 选择，样本是 {len(common)} 个同时开市日。稳定性地图里同一次全区间、滞后固定 1 阶的 p = **{pval(lag1)}**。这是两种设定。",
            f"The summary EG p = **{pval(page.eg['pvalue'])}** uses AIC lags on {len(common)} common sessions. The same full window with lag fixed at 1, as in the map, has p = **{pval(lag1)}**. Those are two specifications.",
        )
    )
    facts = window_count_facts(len(common))
    detail = tx("、", ", ").join(
        tx(f"{length} 日 {count} 个", f"{count} windows of {length} days") for length, count in facts["parts"]
    ) or tx("没有足够长的窗口", "no window is long enough")
    st.caption(
        tx(
            f"这段有 {facts['n_days']} 个价格点。步长 {facts['step']} 个交易日，每种长度都包含最后一个终点。按这个设计大约 {facts['total']} 个窗口（{detail}）。相邻窗口共用大部分交易日，不能把检验次数当成独立样本。下面的 BH 比例只作参考：窗口重叠，校正既不是严格的独立试验，也不代替 AIC 那一次全样本检验。",
            f"This sample has {facts['n_days']} prices. The step is {facts['step']} sessions, and every length includes the last endpoint. That design is about {facts['total']} windows ({detail}). Neighbours share most days, so the count is not an independent sample. The BH rate below is only a reference: the windows overlap, and the adjustment is neither a strict independent-trial correction nor a substitute for the one full-sample AIC test.",
        )
    )
    unadj_key = f"{page.picked.a_code}|{page.start_day}|{page.end_day}|unadj"
    if st.button(tx("比较未复权收盘", "Compare unadjusted closes"), key="load_unadjusted"):
        try:
            with st.spinner(tx("下载未复权收盘…", "Downloading unadjusted closes…")):
                columns = unadjusted_log_columns(page.picked.a_code, page.picked.h_code, common["fx"], common.index)
            st.session_state["unadj"] = {"key": unadj_key, "frame": columns}
        except Exception as exc:
            st.error(tx(f"未复权价格没有下载到：{exc}", f"Unadjusted prices were not downloaded: {exc}"))
    saved_unadj = st.session_state.get("unadj")
    if isinstance(saved_unadj, dict) and saved_unadj.get("key") == unadj_key:
        extra = saved_unadj["frame"].reindex(common.index)
        common = common.copy()
        common["log_a_raw"] = extra["log_a_raw"]
        common["log_h_raw"] = extra["log_h_raw"]
        st.caption(tx("未复权收盘已对齐到这张表。下面的地图要重新计算，切片里才会出现这一行。Yahoo 的复权价含分红再投资，和港股通税后红利不是同一条价格。", "Unadjusted closes are aligned to this table. Recompute the map to add that row. Yahoo’s adjusted close reinvests dividends and is not the after-tax Stock Connect price."))
    if len(common) < 200:
        st.info(tx("稳定性地图至少需要大约 200 个同时开市日。", "The stability map needs about 200 common sessions."))
        return
    stability_key = f"{page.picked.a_code}|{page.start_day}|{page.end_day}"
    if st.button(tx("计算稳定性地图", "Compute the stability map"), key="run_stability"):
        with st.spinner(tx("正在换窗口重算协整…", "Recomputing cointegration across windows…")):
            st.session_state["stability"] = {"key": stability_key, **cached_stability(common)}
    saved = st.session_state.get("stability")
    if isinstance(saved, dict) and saved.get("key") != stability_key:
        st.caption(tx("股票或日期变了，需要重新计算。", "The pair or the dates changed. Compute again."))
        saved = None
    if not isinstance(saved, dict):
        return
    report = saved["report"]
    rate = report["reject_rate"]
    fragility = report["fragility"]
    words = tx("算不出来", "n/a")
    if np.isfinite(fragility):
        words = tx("比较稳", "fairly stable") if fragility < 0.25 else tx("不太稳", "uneven") if fragility < 0.6 else tx("很脆弱", "fragile")
    kpis(
        [
            (tx("成立比例", "Rejection rate"), pct(rate), "pos" if np.isfinite(rate) and rate >= 0.5 else "neg"),
            (tx("BH 后比例", "BH rate"), pct(report.get("bh_rate", np.nan)), ""),
            (tx("脆弱程度", "Fragility"), pct(fragility), "neg" if np.isfinite(fragility) and fragility >= 0.5 else ""),
            (tx("检验次数", "Tests"), f"{report['n_tests']}", ""),
            (tx("判断", "Reading"), words, ""),
        ],
        columns=5,
    )
    st.caption(
        tx(
            f"BH 在这 {report['n_tests']} 个重叠窗口里拒绝了 {report.get('bh_n_reject', 0)} 个。它比直接数 p<0.05 更严，也仍然不是独立样本的错误发现率。",
            f"BH rejects {report.get('bh_n_reject', 0)} of these {report['n_tests']} overlapping windows. That is stricter than counting p<0.05, and it is still not an independent-trial false-discovery rate.",
        )
    )
    worst = report["worst"]
    show_chart(stability_heatmap(saved["grid"]), "chart_stability")
    by_length = saved.get("by_length")
    if isinstance(by_length, pd.DataFrame) and not by_length.empty:
        show_chart(reject_rate_figure(by_length), "chart_reject_rate")
    if isinstance(worst, pd.DataFrame) and not worst.empty:
        top = worst.iloc[0]
        st.markdown(
            tx(
                f"p 值最高的窗口是 **{pd.Timestamp(top['start']).date()} 至 {pd.Timestamp(top['end']).date()}**，长度 {int(top['length'])}，p = {pval(top['pvalue'])}。",
                f"The highest p-value is **{pd.Timestamp(top['start']).date()} to {pd.Timestamp(top['end']).date()}**, length {int(top['length'])}, p = {pval(top['pvalue'])}.",
            )
        )
        labels = [f"{pd.Timestamp(row.start).date()} → {pd.Timestamp(row.end).date()} · p {pval(row.pvalue)}" for row in worst.itertuples(index=False)]
        picked_window = st.selectbox(tx("失稳窗口", "Fragile window"), labels, key="unstable_window")
        if st.button(tx("把信号图切到这个窗口", "Show this window on the signal chart"), key="focus_unstable"):
            chosen = worst.iloc[labels.index(picked_window)]
            st.session_state["chart_focus"] = {"start": str(pd.Timestamp(chosen["start"]).date()), "end": str(pd.Timestamp(chosen["end"]).date())}
            st.rerun()
    checks = saved["checks"].copy()
    if "code" in checks.columns:
        checks["情形"] = checks["code"].map(slice_label)
    checks[tx("滞后 1 阶 p", "Lag-1 p")] = checks["pvalue"].map(pval)
    checks[tx("AIC p", "AIC p")] = checks["pvalue_aic"].map(pval) if "pvalue_aic" in checks.columns else "—"
    checks = checks.rename(columns={"情形": tx("情形", "Slice"), "天数": tx("天数", "Days")})
    show_table(checks[[tx("情形", "Slice"), tx("天数", "Days"), tx("滞后 1 阶 p", "Lag-1 p"), tx("AIC p", "AIC p")]])
