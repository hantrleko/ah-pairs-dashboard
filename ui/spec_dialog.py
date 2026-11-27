"""规格对话框。侧栏表单的移植：信号、成本、高级三组，应用后重算。"""

from __future__ import annotations

import streamlit as st

from ah_pairs.backtest import StrategyParams
from ah_pairs.costs import CostParams
from ah_pairs.i18n import tx
from ah_pairs.links import SpecLink


def build_params(spec: SpecLink) -> StrategyParams:
    """从链接种子构造参数，高级项用默认规格。"""
    include_costs = bool(spec.include_costs)
    return StrategyParams(
        hedge=spec.hedge,
        hedge_window=int(spec.hedge_window),
        z_window=int(spec.z_window),
        entry_z=float(spec.entry_z),
        exit_z=float(spec.exit_z),
        stop_z=4.0,
        kalman_delta=1e-4,
        kalman_delta_auto=True,
        allow_short_a=bool(spec.allow_short_a),
        require_positive_beta=True,
        fill=spec.fill,
        beta_band=0.10,
        costs=CostParams(
            a_commission=0.00025 if include_costs else 0.0,
            h_commission=0.0004 if include_costs else 0.0,
            a_borrow_annual=0.05 if include_costs else 0.0,
            h_borrow_annual=0.02 if include_costs else 0.0,
            use_statutory_stamp=include_costs,
            a_stamp_sell=0.0005 if include_costs else 0.0,
            h_stamp=0.001 if include_costs else 0.0,
            slippage=0.0,
        ),
    )


def _bps(label: str, default: float, key: str) -> float:
    return st.number_input(label, min_value=0.0, max_value=50.0, value=default, step=0.5, key=key) / 10_000


@st.dialog(tx("调整规格", "Adjust the specification"), width="large")
def spec_dialog() -> None:
    seed = st.session_state["applied_params"]
    seed_dates = st.session_state["applied_dates"]
    seed_costs = bool(st.session_state["applied_costs_on"])

    date_range = st.date_input(tx("评估区间", "Evaluation window"), value=seed_dates, key="dlg_dates")
    st.caption(tx("区间之前的行情只用来预热 beta、z，以及卡尔曼 δ。", "Prices before this window warm up beta, z, and the Kalman delta."))

    st.markdown(tx("**信号**", "**Signal**"))
    hedge = st.radio(
        tx("对冲比率", "Hedge ratio"),
        ["rolling", "kalman"],
        index=0 if seed.hedge == "rolling" else 1,
        format_func=lambda code: tx("滚动 OLS", "Rolling OLS") if code == "rolling" else tx("卡尔曼滤波", "Kalman"),
        horizontal=True,
        key="dlg_hedge",
    )
    hcol1, hcol2 = st.columns(2)
    hedge_window = hcol1.slider(tx("估计窗口（交易日）", "Estimation window (days)"), 40, 250, int(seed.hedge_window), key="dlg_hw")
    z_window = hcol2.slider(tx("z 窗口（交易日）", "z window (days)"), 20, 120, int(seed.z_window), key="dlg_zw")
    zcol1, zcol2 = st.columns(2)
    entry_z = zcol1.slider(tx("开仓 |z|", "Entry |z|"), 1.0, 3.5, float(seed.entry_z), 0.1, key="dlg_zin")
    exit_z = zcol2.slider(tx("平仓 |z|", "Exit |z|"), 0.0, 1.5, float(seed.exit_z), 0.1, key="dlg_zout")
    kalman_auto = st.checkbox(tx("用预热样本估计卡尔曼 δ", "Estimate Kalman delta on the warm-up sample"), value=bool(seed.kalman_delta_auto), key="dlg_kauto")
    kalman_delta = st.select_slider(
        tx("卡尔曼 δ（手动）", "Kalman delta (manual)"),
        options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        value=float(seed.kalman_delta),
        disabled=hedge != "kalman" or kalman_auto,
        key="dlg_kdelta",
    )
    fill = st.radio(
        tx("T+1 约束", "T+1 fill"),
        ["next_close", "same_close"],
        index=0 if seed.fill == "next_close" else 1,
        format_func=lambda code: tx("已考虑：次日收盘", "On: next close") if code == "next_close" else tx("未考虑：当日收盘", "Off: same close"),
        horizontal=True,
        key="dlg_fill",
    )

    st.markdown(tx("**高级**", "**Advanced**"))
    acol1, acol2 = st.columns(2)
    stop_z = acol1.slider(tx("止损 |z|（0 表示不用）", "Stop |z| (0 = off)"), 0.0, 6.0, float(seed.stop_z), 0.1, key="dlg_stop")
    beta_band = acol2.slider(tx("β 再平衡带", "Beta rebalance band"), 0.0, 0.50, float(seed.beta_band), 0.01, key="dlg_band")
    allow_short_a = st.checkbox(
        tx("允许做空 A 股", "Allow short A-shares"),
        value=bool(seed.allow_short_a),
        key="dlg_short",
        help=tx("默认关闭。多数 A 股不一定能融到券。打开后，页顶会标明这个假设偏乐观。", "Off by default. Many A-shares cannot be borrowed. Turning it on is flagged on the page."),
    )
    require_positive_beta = st.checkbox(tx("β 不为正时空仓", "Flat when beta is not positive"), value=bool(seed.require_positive_beta), key="dlg_posbeta")

    st.markdown(tx("**成本**", "**Costs**"))
    include_costs = st.checkbox(tx("计入交易成本", "Include trading costs"), value=seed_costs, key="dlg_cost")
    statutory = st.checkbox(tx("使用法定印花税档期", "Statutory stamp schedule"), value=True, key="dlg_statutory")
    ccol1, ccol2 = st.columns(2)
    a_commission = ccol1.number_input(tx("A 股佣金（bps，单边）", "A commission (bps, one way)"), 0.0, 50.0, float(seed.costs.a_commission) * 10_000, 0.5, key="dlg_acomm") / 10_000
    h_commission = ccol2.number_input(tx("H 股佣金（bps，单边）", "H commission (bps, one way)"), 0.0, 50.0, float(seed.costs.h_commission) * 10_000, 0.5, key="dlg_hcomm") / 10_000
    ccol3, ccol4 = st.columns(2)
    a_stamp = ccol3.number_input(tx("A 股卖出印花税（bps）", "A sell stamp (bps)"), 0.0, 50.0, float(seed.costs.a_stamp_sell) * 10_000, 0.5, key="dlg_astamp") / 10_000
    h_stamp = ccol4.number_input(tx("H 股印花税（bps，单边）", "H stamp (bps, one way)"), 0.0, 50.0, float(seed.costs.h_stamp) * 10_000, 0.5, key="dlg_hstamp") / 10_000
    bcol1, bcol2, bcol3 = st.columns(3)
    a_borrow = bcol1.number_input(tx("A 股融券年化（%）", "A borrow (% per year)"), 0.0, 30.0, float(seed.costs.a_borrow_annual) * 100, 0.5, key="dlg_aborrow") / 100
    h_borrow = bcol2.number_input(tx("H 股融券年化（%）", "H borrow (% per year)"), 0.0, 30.0, float(seed.costs.h_borrow_annual) * 100, 0.5, key="dlg_hborrow") / 100
    slippage = bcol3.number_input(tx("滑点（bps，单边）", "Slippage (bps, one way)"), 0.0, 50.0, float(seed.costs.slippage) * 10_000, 0.5, key="dlg_slip") / 10_000

    if st.button(tx("应用参数", "Apply parameters"), type="primary", width="stretch"):
        if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
            st.error(tx("请选择开始和结束两个日期。", "Choose both a start and an end date."))
            return
        try:
            draft = StrategyParams(
                hedge=hedge,
                hedge_window=int(hedge_window),
                z_window=int(z_window),
                entry_z=float(entry_z),
                exit_z=float(exit_z),
                stop_z=float(stop_z),
                kalman_delta=float(kalman_delta),
                kalman_delta_auto=bool(kalman_auto),
                allow_short_a=bool(allow_short_a),
                require_positive_beta=bool(require_positive_beta),
                fill=fill,
                beta_band=float(beta_band),
                costs=CostParams(
                    a_commission=float(a_commission) if include_costs else 0.0,
                    h_commission=float(h_commission) if include_costs else 0.0,
                    a_borrow_annual=float(a_borrow) if include_costs else 0.0,
                    h_borrow_annual=float(h_borrow) if include_costs else 0.0,
                    use_statutory_stamp=bool(statutory) and include_costs,
                    a_stamp_sell=float(a_stamp) if include_costs else 0.0,
                    h_stamp=float(h_stamp) if include_costs else 0.0,
                    slippage=float(slippage) if include_costs else 0.0,
                ),
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state["applied_params"] = draft
        st.session_state["applied_dates"] = (date_range[0], date_range[1])
        st.session_state["applied_costs_on"] = bool(include_costs)
        st.rerun()
