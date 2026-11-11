"""侧栏。行情选择马上生效；估计和对冲参数放进表单，提交后才重算。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ah_pairs.backtest import StrategyParams
from ah_pairs.costs import CostParams
from ah_pairs.i18n import set_lang, tx
from ah_pairs.links import SpecLink, encode_spec, parse_spec_query
from ah_pairs.universe import LIQUID_A_CODES
from ui.cache import cached_universe
from ui.format import QUICK


def _link_spec() -> SpecLink:
    if "link_spec" not in st.session_state:
        st.session_state["link_spec"] = parse_spec_query(dict(st.query_params))
    return st.session_state["link_spec"]


def _sync_query(pair: str, lang: str, start, end, params: StrategyParams, include_costs: bool) -> None:
    spec = SpecLink(
        start=pd.Timestamp(start).date(),
        end=pd.Timestamp(end).date(),
        hedge=params.hedge,
        hedge_window=int(params.hedge_window),
        z_window=int(params.z_window),
        entry_z=float(params.entry_z),
        exit_z=float(params.exit_z),
        fill=params.fill,
        allow_short_a=bool(params.allow_short_a),
        include_costs=bool(include_costs),
    )
    desired = encode_spec(spec, pair=str(pair), lang=str(lang))
    current = {key: st.query_params.get(key) for key in desired}
    if current != desired:
        st.query_params.update(desired)


def _bps(label: str, default: float, key: str) -> float:
    return st.number_input(label, min_value=0.0, max_value=50.0, value=default, step=0.5, key=key) / 10_000


def render_sidebar() -> dict:
    lang = st.selectbox(
        tx("语言", "Language"),
        ["zh", "zh-Hant", "en"],
        format_func=lambda code: {"zh": "简体中文", "zh-Hant": "繁體中文", "en": "English"}[code],
        key="lang",
    )
    set_lang(lang)
    st.caption("EF Lab · eugenefinance.tech")
    st.header(tx("样本", "Sample"))
    try:
        universe = cached_universe()
    except Exception as exc:
        st.error(tx(f"股票池没有加载：{exc}", f"Universe failed to load: {exc}"))
        st.stop()
    liquid_order = {code: i for i, code in enumerate(LIQUID_A_CODES)}
    universe = universe.copy()
    universe["_ord"] = universe["a_code"].map(lambda code: liquid_order.get(code, 10_000))
    universe = universe.sort_values(["_ord", "name"]).reset_index(drop=True)
    labels = {row.a_code: f"{row.name}   {row.a_code} / {row.h_code}" for row in universe.itertuples(index=False)}
    codes = universe["a_code"].tolist()
    if "pair_select" not in st.session_state or st.session_state["pair_select"] not in codes:
        requested = str(st.query_params.get("pair", "")).zfill(6)
        st.session_state["pair_select"] = requested if requested in codes else ("601318" if "601318" in codes else codes[0])
    left, right = st.columns(2)
    for column, pairs in ((left, QUICK[:2]), (right, QUICK[2:])):
        with column:
            for code, name_zh, name_en in pairs:
                if st.button(tx(name_zh, name_en), key=f"quick_{code}", use_container_width=True):
                    st.session_state["pair_select"] = code
                    st.rerun()
    st.caption(
        tx(
            "这四个按钮和下面的公司名单是同一个选择。腾讯没有对应 A 股。谁更接近协整，用「摘要」里的比较。",
            "These buttons and the list below are the same choice. Tencent has no A-share. Use the summary comparison for which pair looks closer to cointegration.",
        )
    )
    a_code = st.selectbox(tx("公司", "Company"), codes, format_func=lambda code: labels.get(code, code), key="pair_select")
    picked = universe.loc[universe["a_code"] == a_code].iloc[0]
    spec = _link_spec()
    if "applied_params" in st.session_state:
        seed = st.session_state["applied_params"]
        seed_dates = st.session_state["applied_dates"]
        seed_costs = bool(st.session_state["applied_costs_on"])
        hedge_seed = seed.hedge
        window_seed = int(seed.hedge_window)
        z_seed = int(seed.z_window)
        entry_seed = float(seed.entry_z)
        exit_seed = float(seed.exit_z)
        fill_seed = seed.fill
        short_seed = bool(seed.allow_short_a)
    else:
        seed_dates = (spec.start, spec.end)
        seed_costs = spec.include_costs
        hedge_seed = spec.hedge
        window_seed = spec.hedge_window
        z_seed = spec.z_window
        entry_seed = spec.entry_z
        exit_seed = spec.exit_z
        fill_seed = spec.fill
        short_seed = spec.allow_short_a

    with st.form("params_form"):
        date_range = st.date_input(
            tx("评估区间", "Evaluation window"),
            value=seed_dates,
            key="spec_dates",
        )
        st.caption(tx("区间之前的行情只用来预热 beta、z，以及卡尔曼 δ。", "Prices before this window warm up beta, z, and the Kalman delta."))
        st.header(tx("信号", "Signal"))
        hedge = st.radio(
            tx("对冲比率", "Hedge ratio"),
            ["rolling", "kalman"],
            index=0 if hedge_seed == "rolling" else 1,
            format_func=lambda code: tx("滚动 OLS", "Rolling OLS") if code == "rolling" else tx("卡尔曼滤波", "Kalman"),
            horizontal=True,
            key="spec_hedge",
            help=tx("两条 β 都会画出来。仓位只用选中的这一条，而且只用当天以前的价格。", "Both betas are plotted. Positions use only the selected one, and only past prices."),
        )
        hedge_window = st.slider(tx("估计窗口（交易日）", "Estimation window (days)"), 40, 250, window_seed, key="spec_hw")
        kalman_auto = st.checkbox(
            tx("用预热样本估计卡尔曼 δ", "Estimate Kalman delta on the warm-up sample"),
            value=True,
            help=tx("在评估区间之前，用一步预测误差的高斯似然挑选 δ。预热不够时仍用下面的数。", "Maximum prediction-error likelihood before the evaluation window. Falls back to the slider if the warm-up is short."),
        )
        kalman_delta = st.select_slider(
            tx("卡尔曼 δ（手动）", "Kalman delta (manual)"),
            options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            value=1e-4,
            disabled=hedge != "kalman" or kalman_auto,
        )
        z_window = st.slider(tx("z 窗口（交易日）", "z window (days)"), 20, 120, z_seed, key="spec_zw")
        entry_z = st.slider(tx("开仓 |z|", "Entry |z|"), 1.0, 3.5, entry_seed, 0.1, key="spec_zin")
        exit_z = st.slider(tx("平仓 |z|", "Exit |z|"), 0.0, 1.5, exit_seed, 0.1, key="spec_zout")
        with st.expander(tx("进阶参数", "Advanced")):
            stop_z = st.slider(tx("止损 |z|（0 表示不用）", "Stop |z| (0 = off)"), 0.0, 6.0, 4.0, 0.1)
            beta_band = st.slider(tx("β 再平衡带", "Beta rebalance band"), 0.0, 0.50, 0.10, 0.01)
            allow_short_a = st.checkbox(
                tx("允许做空 A 股", "Allow short A-shares"),
                value=short_seed,
                key="spec_short",
                help=tx("默认关闭。多数 A 股不一定能融到券。打开后，页顶会标明这个假设偏乐观。", "Off by default. Many A-shares cannot be borrowed. Turning it on is flagged on the page."),
            )
            require_positive_beta = st.checkbox(tx("β 不为正时空仓", "Flat when beta is not positive"), value=True)
        fill = st.radio(
            tx("T+1 约束", "T+1 fill"),
            ["next_close", "same_close"],
            index=0 if fill_seed == "next_close" else 1,
            format_func=lambda code: tx("已考虑：次日收盘", "On: next close") if code == "next_close" else tx("未考虑：当日收盘", "Off: same close"),
            horizontal=True,
            key="spec_fill",
        )
        include_costs = st.checkbox(tx("计入交易成本", "Include trading costs"), value=seed_costs, key="spec_cost")
        with st.expander(tx("成本", "Costs")):
            statutory = st.checkbox(tx("使用法定印花税档期", "Statutory stamp schedule"), value=True)
            a_commission = _bps(tx("A 股佣金（bps，单边）", "A commission (bps, one way)"), 2.5, "a_comm")
            a_stamp = _bps(tx("A 股卖出印花税（bps）", "A sell stamp (bps)"), 5.0, "a_stamp")
            h_commission = _bps(tx("H 股佣金（bps，单边）", "H commission (bps, one way)"), 4.0, "h_comm")
            h_stamp = _bps(tx("H 股印花税（bps，单边）", "H stamp (bps, one way)"), 10.0, "h_stamp")
            a_borrow = st.number_input(tx("A 股融券年化（%）", "A borrow (% per year)"), 0.0, 30.0, 5.0, 0.5) / 100
            h_borrow = st.number_input(tx("H 股融券年化（%）", "H borrow (% per year)"), 0.0, 30.0, 2.0, 0.5) / 100
            slippage = _bps(tx("滑点（bps，单边）", "Slippage (bps, one way)"), 0.0, "slip")
        submitted = st.form_submit_button(tx("应用参数", "Apply parameters"), use_container_width=True)

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info(tx("请选择开始和结束两个日期。", "Choose both a start and an end date."))
        st.stop()
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
        st.stop()
    if submitted or "applied_params" not in st.session_state:
        st.session_state["applied_params"] = draft
        st.session_state["applied_dates"] = (date_range[0], date_range[1])
        st.session_state["applied_costs_on"] = bool(include_costs)
    if spec.rejected:
        st.caption(
            tx(
                f"链接里这些参数无效，已用页面默认：{'、'.join(spec.rejected)}",
                f"Ignored invalid link fields: {', '.join(spec.rejected)}",
            )
        )
    st.caption(tx("改完参数后要点「应用参数」，协整和回测才会重算。", "Cointegration and the backtest rerun after you apply."))
    engine = st.radio(
        tx("回测引擎", "Backtest engine"),
        ["v14", "v15"],
        index=0,
        format_func=lambda code: "V1.4" if code == "v14" else "V1.5",
        horizontal=True,
        key="engine_version",
        help=tx(
            "默认仍是 V1.4。V1.5 的样本外折在回测页点按钮后才计算，不会覆盖 V1.4。",
            "V1.4 stays the default. V1.5 out-of-sample folds run from a button on the backtest page and do not replace V1.4.",
        ),
    )
    if st.button(tx("更新股票池", "Refresh universe")):
        try:
            from ah_pairs.universe import refresh_universe

            refresh_universe()
            cached_universe.clear()
            st.rerun()
        except Exception as exc:
            st.error(tx(f"更新股票池失败：{exc}", f"Universe refresh failed: {exc}"))
    reload_prices = st.button(tx("重新下载这对行情", "Redownload this pair"))
    st.divider()
    st.caption(tx("其他工具", "Links"))
    st.markdown("[EF Lab](https://eugenefinance.tech)")
    st.markdown("[GitHub](https://github.com/hantrleko/ah-pairs-dashboard)")
    start_day, end_day = st.session_state["applied_dates"]
    _sync_query(a_code, lang, start_day, end_day, st.session_state["applied_params"], bool(st.session_state["applied_costs_on"]))
    return {
        "universe": universe,
        "picked": picked,
        "start_day": start_day,
        "end_day": end_day,
        "params": st.session_state["applied_params"],
        "include_costs": bool(st.session_state["applied_costs_on"]),
        "reload": reload_prices,
        "engine": engine if engine in {"v14", "v15"} else "v14",
    }
