"""A+H 协整与动态配对交易看板。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ah_pairs.analysis import analyze_frame, screen_one, warmup_start
from ah_pairs.backtest import StrategyParams
from ah_pairs.costs import CostParams
from ah_pairs.data import load_pair
from ah_pairs.export import equity_png, metrics_latex, metrics_table, spread_png
from ah_pairs.northbound import latest_valid, load_northbound
from ah_pairs.plots import (
    equity_figure,
    hedge_figure,
    northbound_figure,
    premium_figure,
    prices_figure,
    rolling_p_figure,
    signal_figure,
    spread_figure,
    static_resid_figure,
)
from ah_pairs.universe import LIQUID_A_CODES, liquid_subset, load_universe, refresh_universe

st.set_page_config(page_title="A+H 协整与配对交易", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
.stApp { background: #f3efe6; color: #1c1915; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 1.15rem; padding-bottom: 2.5rem; max-width: 1180px; }
section[data-testid="stSidebar"] {
  background: #ebe4d8;
  border-right: 1px solid #ddd4c6;
}
section[data-testid="stSidebar"] > div { padding-top: 1.1rem; }
section[data-testid="stSidebar"] h2 {
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  color: #6f675e;
  font-weight: 600;
  margin-top: 0.4rem;
}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {
  font-size: 0.86rem;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stDateInput div[data-baseweb="input"] {
  background: #fffcf8;
  border-color: #ddd4c6;
}
.stTabs [data-baseweb="tab-list"] {
  gap: 0.15rem;
  border-bottom: 1px solid #e4dacc;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  color: #6f675e;
  border-radius: 0;
  padding: 0.45rem 0.7rem;
  font-size: 0.92rem;
}
.stTabs [aria-selected="true"] {
  color: #1c1915;
  border-bottom: 2px solid #1f4b3a;
}
div[data-testid="stAlert"] {
  background: #f7f1e4;
  border: 0;
  border-left: 3px solid #a6843d;
  border-radius: 2px;
  color: #1c1915;
}
div[data-testid="stDataFrame"] { border: 1px solid #e7dfd2; border-radius: 2px; }
div[data-testid="stPlotlyChart"] {
  background: #fffcf8;
  border: 1px solid #e7dfd2;
  padding: 0.2rem 0.35rem 0;
  margin-bottom: 0.35rem;
}
div[data-testid="stCaptionContainer"] p { color: #8a8175; font-size: 0.82rem; }
.note {
  margin: 0 0 0.95rem;
  padding: 0.15rem 0 0.15rem 0.7rem;
  border-left: 3px solid #a6843d;
  color: #4e483f;
  font-size: 0.9rem;
}
.note.ok { border-left-color: #1f4b3a; }
.mast {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1.5rem;
  padding: 0.15rem 0 0.85rem;
  border-bottom: 1px solid #e4dacc;
  margin-bottom: 0.85rem;
}
.mast .kicker {
  margin: 0;
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  color: #6f675e;
}
.mast h1 {
  margin: 0.15rem 0 0;
  font-family: Palatino Linotype, Songti SC, SimSun, serif;
  font-weight: 500;
  font-size: 2.15rem;
  letter-spacing: -0.02em;
  line-height: 1.1;
  color: #1c1915;
}
.mast .meta { margin: 0.35rem 0 0; color: #6f675e; font-size: 0.9rem; }
.mast .pill {
  border: 1px solid #ddd4c6;
  background: #fffcf8;
  padding: 0.55rem 0.8rem 0.45rem;
  min-width: 7.5rem;
  text-align: right;
}
.mast .pill span { display: block; font-size: 0.72rem; letter-spacing: 0.12em; color: #6f675e; }
.mast .pill strong {
  display: block;
  margin-top: 0.15rem;
  font-size: 1.35rem;
  font-weight: 560;
  font-variant-numeric: tabular-nums;
  color: #1c1915;
}
.kpi-row {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0.55rem;
  margin: 0.2rem 0 0.85rem;
}
.kpi-row.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.kpi {
  background: #fffcf8;
  border: 1px solid #e7dfd2;
  padding: 0.62rem 0.75rem 0.55rem;
}
.kpi span {
  display: block;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  color: #6f675e;
}
.kpi strong {
  display: block;
  margin-top: 0.18rem;
  font-size: 1.28rem;
  font-weight: 560;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
  color: #1c1915;
}
.kpi.neg strong { color: #9b3a3a; }
.kpi.pos strong { color: #1d6b45; }
@media (max-width: 900px) {
  .kpi-row, .kpi-row.four { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .mast { flex-direction: column; align-items: flex-start; }
}
</style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_universe() -> pd.DataFrame:
    return load_universe()


@st.cache_data(show_spinner=False)
def cached_pair(a_code: str, h_code: str, start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
    return load_pair(a_code, h_code, start, end, refresh=False)


@st.cache_data(show_spinner=False)
def cached_northbound() -> pd.Series:
    return load_northbound(refresh=False)


def _pct(value) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.2%}"


def _num(value, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _pval(value) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _bps(label: str, default: float, key: str) -> float:
    return st.number_input(label, min_value=0.0, max_value=50.0, value=default, step=0.5, key=key) / 10_000


def _half_life_text(result: dict | None) -> str:
    if not result:
        return "—"
    if result.get("mean_reverting") and np.isfinite(result.get("half_life", np.nan)):
        return _num(result["half_life"], 1)
    return "不收敛"


def _tone(value) -> str:
    if value is None or not np.isfinite(value) or value == 0:
        return ""
    return "neg" if value < 0 else "pos"


def _kpis(items: list[tuple[str, str, str]], columns: int = 6) -> None:
    kind = "kpi-row four" if columns == 4 else "kpi-row"
    cells = "".join(
        f'<div class="kpi {tone}"><span>{label}</span><strong>{value}</strong></div>'
        for label, value, tone in items
    )
    st.markdown(f'<div class="{kind}">{cells}</div>', unsafe_allow_html=True)


with st.sidebar:
    st.header("样本")
    try:
        universe = cached_universe()
    except Exception as exc:
        st.error(f"股票池没有加载：{exc}")
        st.stop()
    liquid_order = {code: i for i, code in enumerate(LIQUID_A_CODES)}
    universe = universe.copy()
    universe["_ord"] = universe["a_code"].map(lambda code: liquid_order.get(code, 10_000))
    universe = universe.sort_values(["_ord", "name"]).reset_index(drop=True)
    labels = {
        row.a_code: f"{row.name}   {row.a_code} / {row.h_code}"
        for row in universe.itertuples(index=False)
    }
    codes = universe["a_code"].tolist()
    if "pair_select" not in st.session_state or st.session_state["pair_select"] not in codes:
        st.session_state["pair_select"] = "601318" if "601318" in codes else codes[0]
    quick = [("601318", "平安"), ("601398", "工行"), ("600036", "招行"), ("601857", "石油")]
    left, right = st.columns(2)
    for column, pairs in ((left, quick[:2]), (right, quick[2:])):
        with column:
            for code, name in pairs:
                if st.button(name, key=f"quick_{code}", use_container_width=True):
                    st.session_state["pair_select"] = code
                    st.rerun()
    st.caption("腾讯没有对应 A 股，不在这张名单里。")
    a_code = st.selectbox(
        "公司",
        codes,
        format_func=lambda code: labels.get(code, code),
        key="pair_select",
    )
    picked = universe.loc[universe["a_code"] == a_code].iloc[0]
    date_range = st.date_input(
        "评估区间",
        value=(pd.Timestamp("2018-01-01").date(), pd.Timestamp.today().date()),
    )
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("请选择开始和结束两个日期。")
        st.stop()
    start_day, end_day = date_range
    st.caption("区间之前的行情只用来预热 beta 和 z。")
    if st.button("更新股票池"):
        try:
            refresh_universe()
            cached_universe.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"更新股票池失败：{exc}")

    st.header("信号")
    hedge_label = st.radio("对冲比率", ["滚动 OLS", "卡尔曼滤波"], horizontal=True)
    hedge = "rolling" if hedge_label == "滚动 OLS" else "kalman"
    hedge_window = st.slider("估计窗口（交易日）", 40, 250, 120)
    kalman_delta = st.select_slider(
        "卡尔曼 δ",
        options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        value=1e-4,
        disabled=hedge != "kalman",
    )
    z_window = st.slider("z 窗口（交易日）", 20, 120, 60)
    entry_z = st.slider("开仓 |z|", 1.0, 3.5, 2.0, 0.1)
    exit_z = st.slider("平仓 |z|", 0.0, 1.5, 0.5, 0.1)
    stop_z = st.slider("止损 |z|（0 表示不用）", 0.0, 6.0, 4.0, 0.1)
    beta_band = st.slider("β 再平衡带", 0.0, 0.50, 0.10, 0.01)
    fill_label = st.radio("成交", ["次日收盘", "当日收盘"], horizontal=True)
    fill = "next_close" if fill_label == "次日收盘" else "same_close"
    allow_short_a = st.checkbox("允许做空 A 股", value=True)
    require_positive_beta = st.checkbox("β 不为正时空仓", value=True)

    st.header("成本")
    statutory = st.checkbox("使用法定印花税档期", value=True)
    a_commission = _bps("A 股佣金（bps，单边）", 2.5, "a_comm")
    a_stamp = _bps("A 股卖出印花税（bps，关闭法定档期时使用）", 5.0, "a_stamp")
    h_commission = _bps("H 股佣金（bps，单边）", 4.0, "h_comm")
    h_stamp = _bps("H 股印花税（bps，单边，关闭法定档期时使用）", 10.0, "h_stamp")
    a_borrow = st.number_input("A 股融券年化（%）", 0.0, 30.0, 5.0, 0.5) / 100
    h_borrow = st.number_input("H 股融券年化（%）", 0.0, 30.0, 2.0, 0.5) / 100
    slippage = _bps("汇率与成交滑点（bps，单边）", 0.0, "slip")
    reload_prices = st.button("重新下载这对行情")

try:
    params = StrategyParams(
        hedge=hedge,
        hedge_window=int(hedge_window),
        z_window=int(z_window),
        entry_z=float(entry_z),
        exit_z=float(exit_z),
        stop_z=float(stop_z),
        kalman_delta=float(kalman_delta),
        allow_short_a=bool(allow_short_a),
        require_positive_beta=bool(require_positive_beta),
        fill=fill,
        beta_band=float(beta_band),
        costs=CostParams(
            a_commission=float(a_commission),
            h_commission=float(h_commission),
            a_borrow_annual=float(a_borrow),
            h_borrow_annual=float(h_borrow),
            use_statutory_stamp=bool(statutory),
            a_stamp_sell=float(a_stamp),
            h_stamp=float(h_stamp),
            slippage=float(slippage),
        ),
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if reload_prices:
    try:
        load_pair(picked.a_code, picked.h_code, "2010-01-01", str(end_day), refresh=True)
        cached_pair.clear()
    except Exception as exc:
        st.error(f"重新下载失败：{exc}")

warm = warmup_start(start_day, params).strftime("%Y-%m-%d")
try:
    with st.spinner("读取行情并计算协整与回测…"):
        prices, notes = cached_pair(picked.a_code, picked.h_code, warm, str(end_day))
        north = None
        north_error = ""
        try:
            north = cached_northbound()
        except Exception as exc:
            north_error = str(exc)
        result = analyze_frame(prices, params, eval_start=pd.Timestamp(start_day), northbound=north)
except Exception as exc:
    st.error(str(exc))
    st.stop()

frame = result["frame"]
eg = result["eg"]
stats = result["stats"]
static_hl = result["half_life_static"]
traded_hl = result["half_life_traded"]
johansen = result["johansen"]

for note in notes:
    st.caption(note)

premium_now = frame["premium"].dropna()
premium_last = float(premium_now.iloc[-1]) if len(premium_now) else np.nan
st.markdown(
    f"""
<div class="mast">
  <div>
    <p class="kicker">V0.2 · A + H 价差</p>
    <h1>{picked["name"]}</h1>
    <p class="meta">{picked.a_code} · {picked.h_code} · 最近交易日 {frame.index.max().date()} · {frame.index.min().date()} — {frame.index.max().date()} · {len(frame)} 个同时开市日。A、H 节假日不同，只保留两边都开市的日期。</p>
  </div>
  <div class="pill"><span>最近溢价</span><strong>{_pct(premium_last)}</strong></div>
</div>
    """,
    unsafe_allow_html=True,
)
_kpis(
    [
        ("EG p 值", _pval(eg["pvalue"]), ""),
        ("全样本 β", _num(eg["beta"], 3), ""),
        ("半衰期（日）", _half_life_text(static_hl), ""),
        ("净收益", _pct(stats["total_return"]), _tone(stats["total_return"])),
        ("夏普", _num(stats["sharpe"], 2), _tone(stats["sharpe"])),
        ("最大回撤", _pct(stats["max_drawdown"]), "neg" if np.isfinite(stats["max_drawdown"]) else ""),
    ]
)
if eg["reject_5"]:
    verdict = (
        f'<p class="note ok">Engle–Granger p = {_pval(eg["pvalue"])}，5% 水平上拒绝无协整。'
        "这是样本内描述，交易用的 β 只用当天以前的价格。</p>"
    )
else:
    verdict = '<p class="note">这段区间没有在 5% 水平上拒绝无协整。回测仍会按规则下单，价差不一定回到均值。</p>'
st.markdown(verdict, unsafe_allow_html=True)

tab_px, tab_coint, tab_bt, tab_flow, tab_scan, tab_note = st.tabs(
    ["价差与信号", "协整诊断", "回测", "北向资金", "一组对照", "方法说明"]
)

with tab_px:
    st.plotly_chart(prices_figure(frame), use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(premium_figure(frame), use_container_width=True)
    right.plotly_chart(hedge_figure(frame), use_container_width=True)
    st.plotly_chart(spread_figure(frame, params.z_window), use_container_width=True)
    st.plotly_chart(signal_figure(frame, params.entry_z, params.exit_z, params.stop_z), use_container_width=True)

with tab_coint:
    st.markdown(
        "左边是整段评估区间的 Engle–Granger，回归用到了区间内的全部价格，用来判断「这段时期是否协整」。"
        "交易仓位用的是另一套 β：滚动 OLS 或卡尔曼滤波，都只用当天以前的价格。"
    )
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Engle–Granger**")
        st.dataframe(
            pd.DataFrame(
                {
                    "项目": ["重叠天数", "α", "β", "R²", "EG t 统计量", "5% 临界值", "EG p 值", "残差 ADF 统计量", "残差 ADF p 值", "φ", "κ（每日）", "半衰期（交易日）", "长期均值"],
                    "数值": [
                        f"{eg['nobs']}",
                        _num(eg["alpha"], 3),
                        _num(eg["beta"], 3),
                        _num(eg["r2"], 3),
                        _num(eg["t_stat"], 2),
                        _num(eg["crit_5"], 2),
                        _pval(eg["pvalue"]),
                        _num(result["residual_adf"].get("adf_stat"), 2),
                        _pval(result["residual_adf"].get("adf_pvalue")),
                        _num(static_hl["phi"], 3),
                        _num(static_hl["kappa"], 4),
                        _half_life_text(static_hl),
                        _num(static_hl["long_run_mean"], 3),
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"半衰期来自 AR(1)：价差_t = c + φ × 价差_(t−1) + e。φ = {_num(static_hl.get('phi'), 3)}，"
            "半衰期 = −ln(2) / ln(φ)。φ 越接近 1，回复越慢。"
        )
        adf_note = (result.get("residual_adf") or {}).get("note")
        if adf_note:
            st.caption(adf_note)
    with d2:
        st.markdown("**Johansen 迹检验**")
        if not johansen.get("ok"):
            st.info(johansen.get("message") or "Johansen 检验没有返回结果。")
        else:
            st.dataframe(
                pd.DataFrame(
                    {
                        "项目": ["差分滞后", "迹统计量（r=0）", "5% 临界值", "5% 结论", "最大特征根", "最大特征根 5% 临界值"],
                        "数值": [
                            f"{johansen['k_ar_diff']}",
                            _num(johansen["trace_r0"], 2),
                            _num(johansen["trace_crit_95"], 2),
                            "拒绝无协整" if johansen["trace_reject_r0_95"] else "不拒绝",
                            _num(johansen["max_eig_r0"], 2),
                            _num(johansen["max_eig_crit_95"], 2),
                        ],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        st.markdown("**实际用于交易的价差**")
        st.dataframe(
            pd.DataFrame(
                {
                    "项目": ["φ", "半衰期（交易日）", "κ（每日）", "长期均值"],
                    "数值": [
                        _num(None if traded_hl is None else traded_hl["phi"], 3),
                        _half_life_text(traded_hl),
                        _num(None if traded_hl is None else traded_hl["kappa"], 4),
                        _num(None if traded_hl is None else traded_hl["long_run_mean"], 3),
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("这一行是因果价差在评估区间里的 AR(1)，和左边的全样本残差不是同一条序列。")
    rolling_p = result.get("rolling_p")
    if rolling_p is not None and len(rolling_p):
        st.plotly_chart(rolling_p_figure(rolling_p), use_container_width=True)
        st.caption("每个点用它之前 252 个同时开市日做 Engle–Granger，大约每 42 个交易日算一次。虚线是 0.05。")
    else:
        st.caption("评估区间不足 252 个交易日，没有画滚动协整 p 值。")
    st.plotly_chart(static_resid_figure(frame), use_container_width=True)

with tab_bt:
    _kpis(
        [
            ("年化收益", _pct(stats["cagr"]), _tone(stats["cagr"])),
            ("年化波动", _pct(stats["vol"]), ""),
            ("成本合计", _pct(stats["total_cost"]), ""),
            ("融券费合计", _pct(stats["total_borrow"]), ""),
        ],
        columns=4,
    )
    _kpis(
        [
            ("已平仓次数", _num(stats["n_trades"], 0), ""),
            ("胜率", _pct(stats["win_rate"]), ""),
            ("平均持有（开市日）", _num(stats["avg_holding_days"], 1), ""),
            ("持仓占比", _pct(stats["exposure"]), ""),
        ],
        columns=4,
    )
    st.plotly_chart(equity_figure(frame), use_container_width=True)
    fill_text = "信号在收盘产生，下一个同时开市日的收盘成交。" if params.fill == "next_close" else "信号和成交放在同一个收盘。这比次日成交更紧，仍是日频，没有日内来回。"
    short_text = "允许做空 A 股，并扣除侧栏里的融券年化。" if params.allow_short_a else "不允许做空 A 股：价差偏高时策略空仓，只在 A 股相对便宜时买 A、卖 H。"
    st.caption(
        f"{frame.index.min().date()} 至 {frame.index.max().date()}，{stats['n_days']} 个同时开市日。"
        f"夏普按 252 天年化，无风险利率按 0。成本含侧栏里的佣金、法定印花税、融券费和滑点。{fill_text}{short_text}"
        " 净值图里的只持有 A、只持有 H 和等权 A+H 是对照，不是这个策略的基准。"
    )
    sensitivity = result.get("sensitivity")
    if isinstance(sensitivity, pd.DataFrame) and not sensitivity.empty:
        st.markdown("**开仓阈值敏感性**")
        show_s = sensitivity.copy()
        show_s["当前"] = show_s["current"].map({True: "是", False: ""})
        st.dataframe(
            show_s.rename(
                columns={
                    "entry_z": "开仓 |z|",
                    "sharpe": "夏普",
                    "max_drawdown": "最大回撤",
                    "total_return": "净收益",
                    "n_trades": "交易次数",
                }
            )[["开仓 |z|", "夏普", "最大回撤", "净收益", "交易次数", "当前"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "开仓 |z|": st.column_config.NumberColumn(format="%.1f"),
                "夏普": st.column_config.NumberColumn(format="%.2f"),
                "最大回撤": st.column_config.NumberColumn(format="%.2%"),
                "净收益": st.column_config.NumberColumn(format="%.2%"),
            },
        )
        st.caption("其他参数保持侧栏不变，只改开仓 |z|。这张表会把同一段样本试很多次，不能当成独立样本外结果。")
    trades = result["trades"].copy()
    if trades.empty:
        st.info("这个参数组合在评估区间里没有开仓。")
    else:
        show = trades.copy()
        show["方向"] = show["direction"].map({1: "做多价差（买 A 卖 H）", -1: "做空价差（卖 A 买 H）"})
        show["状态"] = show["closed"].map({True: "已平仓", False: "持仓中"})
        show["开仓"] = pd.to_datetime(show["entry_date"]).dt.date
        show["平仓"] = pd.to_datetime(show["exit_date"]).dt.date
        st.dataframe(
            show.rename(
                columns={
                    "entry_z": "开仓 z",
                    "exit_z": "平仓 z",
                    "holding_days": "持有开市日",
                    "trade_return": "收益",
                }
            )[["开仓", "平仓", "方向", "开仓 z", "平仓 z", "持有开市日", "收益", "状态"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "开仓 z": st.column_config.NumberColumn(format="%.2f"),
                "平仓 z": st.column_config.NumberColumn(format="%.2f"),
                "收益": st.column_config.NumberColumn(format="%.2%"),
            },
        )
    export = frame.reset_index()
    summary = metrics_table(result, picked["name"], picked.a_code, picked.h_code)
    latex = metrics_latex(summary)
    spread_bytes = spread_png(frame, params.z_window)
    equity_bytes = equity_png(frame)
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "指标 CSV",
        summary.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{picked.a_code}_metrics.csv",
        mime="text/csv",
    )
    d2.download_button(
        "指标 LaTeX",
        latex.encode("utf-8"),
        file_name=f"{picked.a_code}_metrics.tex",
        mime="text/plain",
    )
    d3.download_button(
        "日频结果 CSV",
        export.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{picked.a_code}_{picked.h_code}_daily.csv",
        mime="text/csv",
    )
    if spread_bytes and equity_bytes:
        p1, p2 = st.columns(2)
        p1.download_button("价差图 PNG", spread_bytes, file_name=f"{picked.a_code}_spread.png", mime="image/png")
        p2.download_button("净值图 PNG", equity_bytes, file_name=f"{picked.a_code}_equity.png", mime="image/png")

with tab_flow:
    flow = result["northbound"]
    if north_error:
        st.info(f"北向资金没有载入：{north_error}")
    elif flow is None or not flow.get("ok"):
        st.info((flow or {}).get("message") or "北向净买入和这条价差的重叠不够，没有做回归。")
    else:
        last = latest_valid(north) if north is not None else None
        st.caption(
            "左边是交易价差的日变化，柱子是沪深港通北向合计净买入（亿元）。"
            "这是两市合计，不是这只股票自己的北向持仓。标准误是 HAC（5 阶）。"
            + (f" 净买入最后有数的日期是 {pd.Timestamp(last).date()}。" if last is not None else "")
        )
        st.plotly_chart(northbound_figure(frame), use_container_width=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "样本天数": flow["nobs"],
                        "起始": pd.Timestamp(flow["start"]).date(),
                        "结束": pd.Timestamp(flow["end"]).date(),
                        "系数": flow["beta"],
                        "标准误": flow["se"],
                        "t值": flow["t_stat"],
                        "p值": flow["pvalue"],
                        "R²": flow["r2"],
                        "相关系数": flow["corr"],
                    }
                ]
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "系数": st.column_config.NumberColumn(format="%.6f"),
                "标准误": st.column_config.NumberColumn(format="%.6f"),
                "t值": st.column_config.NumberColumn(format="%.2f"),
                "p值": st.column_config.NumberColumn(format="%.3f"),
                "R²": st.column_config.NumberColumn(format="%.4f"),
                "相关系数": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.markdown(flow["message"] + " 系数表示北向净买入增加 1 亿元时，对数价差当天变化多少。两者是同一天发生的。")

with tab_scan:
    st.markdown("用侧栏里的同一套参数，给流动性较好的几只 A+H 做对照。第一次会下载行情。")
    scan_names = liquid_subset(cached_universe()).head(12)
    st.caption("、".join(scan_names["name"].tolist()))
    if st.button("扫描这组公司"):
        rows = []
        progress = st.progress(0.0, text="准备扫描")
        targets = list(scan_names.itertuples(index=False))
        for i, item in enumerate(targets):
            progress.progress((i + 1) / len(targets), text=str(item.name))
            try:
                pair_prices, _pair_notes = cached_pair(item.a_code, item.h_code, warm, str(end_day))
                summary = screen_one(pair_prices, params, pd.Timestamp(start_day))
                rows.append({"公司": item.name, "A": item.a_code, "H": item.h_code, **summary})
            except Exception as exc:
                rows.append({"公司": item.name, "A": item.a_code, "H": item.h_code, "说明": str(exc)})
        progress.empty()
        st.session_state["scan_rows"] = rows
    if "scan_rows" in st.session_state:
        st.dataframe(
            pd.DataFrame(st.session_state["scan_rows"]),
            hide_index=True,
            use_container_width=True,
            column_config={
                "eg_pvalue": st.column_config.NumberColumn("EG p", format="%.3f"),
                "eg_beta": st.column_config.NumberColumn("β", format="%.2f"),
                "eg_t": st.column_config.NumberColumn("EG t", format="%.2f"),
                "half_life": st.column_config.NumberColumn("半衰期", format="%.1f"),
                "sharpe": st.column_config.NumberColumn("夏普", format="%.2f"),
                "cagr": st.column_config.NumberColumn("年化", format="%.2%"),
                "max_drawdown": st.column_config.NumberColumn("最大回撤", format="%.2%"),
                "latest_premium": st.column_config.NumberColumn("最新溢价", format="%.1%"),
            },
        )

with tab_note:
    st.markdown(
        """
**价格。** A 股、H 股用复权收盘价。H 股乘以港元兑人民币 `HKDCNY`，再和 A 股取对数。两边节假日不同，只保留同时有价格的日期，股价不向前填充。汇率最多向前填 5 个交易日。

**协整。** 全样本 Engle–Granger 是 `log A = α + β log H + e`。报告的 EG p 值来自协整分布。残差上的普通 ADF 只是补充：残差是同一段样本估出来的，这个 p 值会偏小，不能拿来代替 EG p 值。Johansen 迹检验含常数项。半衰期来自 AR(1)，`半衰期 = −ln 2 / ln φ`。滚动 p 值每个窗口只用当时已经过去的 252 个交易日。

**交易。** 回测用的 β 是滚动 OLS 或卡尔曼滤波，只用当天以前的价格。图上两条 β 都会画出来，仓位只用侧栏选中的那一条。z 分数用含当天在内的滚动均值和标准差，成交放在信号之后。默认下一个同时开市日收盘才成交，所以 A 股不会在买入的同一根日 K 里卖出。

**成本和日历。** 佣金、法定印花税、融券费、汇率与成交滑点都在侧栏里。印花税按日期分档。A、H 节假日不一致的日期整行删掉，不用前值填股价。

**前视和多重比较。** 全样本 β、EG 和 Johansen 用了评估区间里的全部价格，只能当描述。交易信号本身不用未来价格。开仓阈值敏感性、以及「一组对照」里按 p 值排序，都是同一段样本上的多次试验，最小的 p 值或最好的夏普会被挑得偏乐观。

**不要读成什么。** 样本内协整不保证以后还在。A/H 不能转换，溢价可以在高位停很多年。做空 A 股取决于能不能借到券。腾讯没有 A 股，不在这个股票池里。

**权重。** 做多价差是买 A、按 β 卖 H，`|w_A| + |w_H| = 1`。持仓期间把开仓时的 β 锁住，只有 β 的相对变化超过再平衡带才调整。

**北向。** 东财汇总的北向净买入，单位亿元。2024 年 8 月之后日频流向停止披露。回归只描述价差变化和全市场北向是否同一天同向。
        """
    )
