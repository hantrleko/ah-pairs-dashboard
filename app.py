"""A+H 协整与动态配对交易看板。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ah_pairs.analysis import analyze_frame, pick_lowest_p, screen_one, warmup_start
from ah_pairs.backtest import StrategyParams
from ah_pairs.costs import CostParams
from ah_pairs.data import load_pair
from ah_pairs.export import equity_png, metrics_latex, metrics_table, spread_png
from ah_pairs.northbound import latest_valid, load_northbound
from ah_pairs.plots import (
    core_signal_figure,
    equity_figure,
    hedge_figure,
    northbound_figure,
    premium_figure,
    prices_figure,
    reject_rate_figure,
    rolling_p_figure,
    stability_heatmap,
    static_resid_figure,
)
from ah_pairs.stability import fragility_report, length_summary, slice_checks, stability_grid
from ah_pairs.universe import LIQUID_A_CODES, liquid_subset, load_universe, refresh_universe

st.set_page_config(
    page_title="EF Lab | A+H 协整配对",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
LOGO = Path(__file__).resolve().parent / "assets" / "eugene-finance-logo.jpg"
st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
.stApp { background: #f4f7fb; color: #16324f; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 0.45rem; padding-bottom: 2.2rem; max-width: 1180px; }
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 0.35rem; }
section[data-testid="stSidebar"] {
  background: #eef3f8;
  border-right: 1px solid #d5dee8;
}
section[data-testid="stSidebar"] > div { padding-top: 0.85rem; }
section[data-testid="stSidebar"] h2 {
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  color: #5a6a7a;
  font-weight: 600;
  margin-top: 0.4rem;
}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {
  font-size: 0.86rem;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stDateInput div[data-baseweb="input"] {
  background: #ffffff;
  border-color: #d5dee8;
}
.stTabs [data-baseweb="tab-list"] {
  gap: 0.15rem;
  border-bottom: 1px solid #d5dee8;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  color: #5a6a7a;
  border-radius: 0;
  padding: 0.45rem 0.7rem;
  font-size: 0.92rem;
}
.stTabs [aria-selected="true"] {
  color: #1f4e79;
  border-bottom: 2px solid #1f4e79;
}
div[data-testid="stAlert"] {
  background: #f4f7fb;
  border: 0;
  border-left: 3px solid #1f4e79;
  border-radius: 2px;
  color: #16324f;
}
div[data-testid="stDataFrame"] { border: 1px solid #d5dee8; border-radius: 2px; }
div[data-testid="stPlotlyChart"] {
  background: #ffffff;
  border: 1px solid #d5dee8;
  padding: 0.2rem 0.35rem 0;
  margin-bottom: 0.35rem;
}
div[data-testid="stCaptionContainer"] p { color: #5a6a7a; font-size: 0.82rem; }
.brand h1 {
  margin: 0;
  color: #1f4e79;
  font-size: 1.65rem;
  font-weight: 650;
  letter-spacing: -0.02em;
  line-height: 1.25;
}
.brand .subtitle { margin: 0.35rem 0 0; color: #5a6a7a; font-size: 0.92rem; }
.note {
  margin: 0 0 0.95rem;
  padding: 0.15rem 0 0.15rem 0.7rem;
  border-left: 3px solid #1f4e79;
  color: #3d4c5c;
  font-size: 0.9rem;
}
.note.ok { border-left-color: #1d6b45; }
.mast {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1.5rem;
  padding: 0 0 0.55rem;
  border-bottom: 1px solid #d5dee8;
  margin: 0.15rem 0 0.55rem;
}
.mast .kicker, .brand .kicker {
  margin: 0 0 0.2rem;
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  color: #5a6a7a;
}
.pair-name {
  margin: 0;
  font-family: Palatino Linotype, Songti SC, SimSun, serif;
  font-weight: 500;
  font-size: 1.85rem;
  letter-spacing: -0.02em;
  line-height: 1.1;
  color: #16324f;
}
.mast .meta { margin: 0.35rem 0 0; color: #5a6a7a; font-size: 0.9rem; }
.mast .pill {
  border: 1px solid #d5dee8;
  background: #ffffff;
  padding: 0.55rem 0.8rem 0.45rem;
  min-width: 7.5rem;
  text-align: right;
}
.mast .pill span { display: block; font-size: 0.72rem; letter-spacing: 0.12em; color: #5a6a7a; }
.mast .pill strong {
  display: block;
  margin-top: 0.15rem;
  font-size: 1.35rem;
  font-weight: 560;
  font-variant-numeric: tabular-nums;
  color: #1f4e79;
}
.kpi-row {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0.55rem;
  margin: 0 0 0.65rem;
}
.kpi-row.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.kpi {
  background: #ffffff;
  border: 1px solid #d5dee8;
  border-radius: 8px;
  padding: 0.62rem 0.75rem 0.55rem;
  box-shadow: 0 1px 2px rgba(22, 50, 79, 0.05), 0 6px 16px rgba(22, 50, 79, 0.04);
}
.kpi span {
  display: block;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  color: #5a6a7a;
}
.kpi strong {
  display: block;
  margin-top: 0.18rem;
  font-size: 1.28rem;
  font-weight: 560;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
  color: #16324f;
}
.kpi.neg strong { color: #9b3a3a; }
.kpi.pos strong { color: #1d6b45; }
.site-footer {
  margin-top: 1.6rem;
  padding-top: 0.75rem;
  border-top: 1px solid #d5dee8;
  color: #5a6a7a;
  font-size: 0.82rem;
}
.site-footer a { color: #1f4e79; text-decoration: none; }
.lesson {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.7rem;
  margin: 0.35rem 0 0.9rem;
}
.lesson article {
  background: #ffffff;
  border: 1px solid #d5dee8;
  border-radius: 8px;
  padding: 0.85rem 0.95rem 0.8rem;
  box-shadow: 0 1px 2px rgba(22, 50, 79, 0.05);
}
.lesson strong { display: block; color: #1f4e79; margin-bottom: 0.35rem; font-size: 0.95rem; }
.lesson p { margin: 0; color: #3d4c5c; font-size: 0.92rem; line-height: 1.6; }
@media (max-width: 900px) {
  .kpi-row, .kpi-row.four { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .lesson { grid-template-columns: 1fr; }
  .mast { flex-direction: column; align-items: flex-start; }
  .brand h1 { font-size: 1.25rem; }
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


logo_col, title_col = st.columns([1, 5], vertical_alignment="center")
with logo_col:
    st.image(str(LOGO), width=156)
with title_col:
    st.markdown(
        """
<div class="brand">
  <p class="kicker">V0.8 · EUGENE FINANCE</p>
  <h1>EF Lab · A+H 股协整配对交易研究看板</h1>
  <p class="subtitle">Interactive Cointegration &amp; Dynamic Pairs Trading Lab · 仅供学习与研究，不构成投资建议</p>
</div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.image(str(LOGO), width=148)
    st.caption("EF Lab · eugenefinance.tech")
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
    st.caption("这四只谁更接近协整，用「先看懂」里的比较按钮按当前区间重算。腾讯没有对应 A 股。")
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
    st.markdown("**对冲**")
    hedge_label = st.radio(
        "对冲比率",
        ["滚动 OLS", "卡尔曼滤波"],
        horizontal=True,
        help="两条 β 都会画出来。仓位只用选中的这一条，而且只用当天以前的价格。",
    )
    hedge = "rolling" if hedge_label == "滚动 OLS" else "kalman"
    hedge_window = st.slider(
        "估计窗口（交易日）",
        40,
        250,
        120,
        help="滚动 OLS 的估计长度，也是卡尔曼滤波的初始样本。",
    )
    kalman_delta = st.select_slider(
        "卡尔曼 δ",
        options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        value=1e-4,
        disabled=hedge != "kalman",
        help="状态噪声相对观测噪声的大小。越大，β 越跟着近期价格走。",
    )
    z_window = st.slider(
        "z 窗口（交易日）",
        20,
        120,
        60,
        help="z 分数和价差图上 σ 带使用的滚动窗口。",
    )
    st.markdown("**开平仓**")
    entry_z = st.slider("开仓 |z|", 1.0, 3.5, 2.0, 0.1, help="|z| 超过这里才开仓。越大，交易越少。")
    exit_z = st.slider("平仓 |z|", 0.0, 1.5, 0.5, 0.1, help="|z| 回到这里以内就平仓，必须小于开仓阈值。")
    stop_z = st.slider("止损 |z|（0 表示不用）", 0.0, 6.0, 4.0, 0.1, help="|z| 继续扩大到这里就止损。0 表示不止损。")
    beta_band = st.slider(
        "β 再平衡带",
        0.0,
        0.50,
        0.10,
        0.01,
        help="持仓期间 β 相对开仓时变化超过这个比例，才调整对冲，避免每天付印花税。",
    )
    st.markdown("**成交与成本**")
    fill_label = st.radio(
        "T+1 约束",
        ["已考虑：次日收盘", "未考虑：当日收盘"],
        horizontal=True,
        help="A 股当天买入不能当天卖。选「已考虑」时，信号出现后的下一个同时开市日才成交。",
    )
    fill = "next_close" if fill_label.startswith("已考虑") else "same_close"
    allow_short_a = st.checkbox("允许做空 A 股", value=True, help="关掉后，A 股偏贵时不交易，只在 A 股相对便宜时买 A、卖 H。")
    require_positive_beta = st.checkbox("β 不为正时空仓", value=True, help="对冲比率不为正时不建仓。")
    include_costs = st.checkbox(
        "计入交易成本（A 股印花税和佣金、H 股费用）",
        value=True,
        help="默认开启。扣除 A 股卖方印花税和佣金、H 股印花税和费用、两边融券费，以及汇率与成交滑点。关掉后看到的是不扣这些费用的结果。",
    )

    with st.expander("成本", expanded=False):
        statutory = st.checkbox("使用法定印花税档期", value=True, help="A 股 2023-08-28 起卖方 0.05%，此前 0.10%。H 股 2021-08-01 至 2023-11-16 为 0.13%。")
        a_commission = _bps("A 股佣金（bps，单边）", 2.5, "a_comm")
        a_stamp = _bps("A 股卖出印花税（bps，关闭法定档期时使用）", 5.0, "a_stamp")
        h_commission = _bps("H 股佣金（bps，单边）", 4.0, "h_comm")
        h_stamp = _bps("H 股印花税（bps，单边，关闭法定档期时使用）", 10.0, "h_stamp")
        a_borrow = st.number_input("A 股融券年化（%）", 0.0, 30.0, 5.0, 0.5, help="做空 A 股时按这个年化费率逐日扣除。") / 100
        h_borrow = st.number_input("H 股融券年化（%）", 0.0, 30.0, 2.0, 0.5, help="做空 H 股时按这个年化费率逐日扣除。") / 100
        slippage = _bps("汇率与成交滑点（bps，单边）", 0.0, "slip")
    reload_prices = st.button("重新下载这对行情")
    st.divider()
    st.caption("其他工具")
    st.markdown("[EF Lab](https://eugenefinance.tech)")
    st.markdown("[本项目 GitHub](https://github.com/hantrleko/ah-pairs-dashboard)")

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

premium_now = frame["premium"].dropna()
premium_last = float(premium_now.iloc[-1]) if len(premium_now) else np.nan
st.markdown(
    f"""
<div class="mast">
  <div>
    <p class="kicker">V0.8 · 这一对</p>
    <p class="pair-name">{picked["name"]}</p>
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
prev = st.session_state.get("pair_snapshot")
if prev and prev.get("code") != picked.a_code:
    st.caption(
        f"刚从{prev['name']}换到{picked['name']}：EG p 从 {prev['p']} 变为 {_pval(eg['pvalue'])}。"
        "p 更小只说明这段样本更接近协整，不是策略一定更好。"
    )
st.session_state["pair_snapshot"] = {
    "code": picked.a_code,
    "name": picked["name"],
    "p": _pval(eg["pvalue"]),
}
for note in notes:
    st.caption(note)

QUICK_COMPARE = (
    ("601318", "平安"),
    ("601398", "工行"),
    ("600036", "招行"),
    ("601857", "石油"),
)


def _compare_key() -> str:
    return "|".join(
        [
            str(start_day),
            str(end_day),
            params.hedge,
            str(params.hedge_window),
            str(params.z_window),
            f"{params.entry_z:.1f}",
            f"{params.exit_z:.1f}",
            params.fill,
            "1" if params.allow_short_a else "0",
        ]
    )


def _render_quick_compare(button_key: str) -> None:
    st.markdown("**平安、工行、招行、石油，谁更接近协整**")
    st.caption("用左侧现在的日期和参数现算。p 值更小，只表示价差更像会回到附近。")
    if st.button("按当前区间比较这四只", key=button_key):
        rows = []
        progress = st.progress(0.0, text="准备比较")
        for i, (code, short_name) in enumerate(QUICK_COMPARE):
            progress.progress(i / len(QUICK_COMPARE), text=short_name)
            match = universe.loc[universe["a_code"] == code]
            if match.empty:
                rows.append({"简称": short_name, "公司": code, "eg_pvalue": np.nan, "说明": "名单里没有"})
                continue
            item = match.iloc[0]
            try:
                pair_prices, _pair_notes = cached_pair(item.a_code, item.h_code, warm, str(end_day))
                summary = screen_one(pair_prices, params, pd.Timestamp(start_day))
                rows.append(
                    {
                        "简称": short_name,
                        "公司": item["name"],
                        "A": item.a_code,
                        "H": item.h_code,
                        **summary,
                        "说明": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "简称": short_name,
                        "公司": item["name"],
                        "A": item.a_code,
                        "H": item.h_code,
                        "eg_pvalue": np.nan,
                        "说明": str(exc),
                    }
                )
        progress.empty()
        st.session_state["quick_compare"] = {"key": _compare_key(), "table": pd.DataFrame(rows)}
    saved = st.session_state.get("quick_compare")
    if not isinstance(saved, dict) or not isinstance(saved.get("table"), pd.DataFrame):
        return
    if saved.get("key") != _compare_key():
        st.caption("日期或参数已经变了，需要再比一次。")
        return
    table = saved["table"]
    best = pick_lowest_p(table)
    if best is not None:
        st.markdown(
            f"这四只里，**{best['公司']}** 的 EG p 值最小，是 {_pval(best['eg_pvalue'])}。"
            "这只说明它更接近协整，不说明扣完费用后一定赚钱。"
        )
    view = table.copy()
    if "eg_pvalue" in view.columns:
        view["EG p 值"] = view["eg_pvalue"].map(_pval)
    if "half_life" in view.columns:
        view["半衰期"] = view["half_life"].map(lambda v: _num(v, 1))
    if "sharpe" in view.columns:
        view["夏普"] = view["sharpe"].map(lambda v: _num(v, 2))
    if "total_return" not in view.columns and "说明" in view.columns:
        pass
    columns = [col for col in ["简称", "公司", "A", "H", "EG p 值", "半衰期", "夏普", "说明"] if col in view.columns]
    st.dataframe(view[columns], hide_index=True, use_container_width=True)


tab_easy, tab_px, tab_coint, tab_stable, tab_bt, tab_flow, tab_scan, tab_note = st.tabs(
    ["先看懂", "价差与信号", "协整诊断", "稳定性", "回测", "北向资金", "一组对照", "方法说明"]
)

with tab_easy:
    company = picked["name"]
    p_text = _pval(eg["pvalue"])
    life = _half_life_text(static_hl)
    same_or_not = "更像会回到平常水平" if eg["reject_5"] else "还不能说它会回到平常水平"
    st.markdown(
        f"""
这页看的是**同一家公司的两只股票**：内地的 A 股，和香港的 H 股。现在打开的是 **{company}**。

它们不能互相兑换。所以两边价格不一样是常态，不是捡钱的机会。这个看板只回答一件事：价差偏离之后，历史上有没有回到附近。
        """
    )
    st.markdown(
        """
<div class="lesson">
  <article>
    <strong>1. 两地各有一个价格</strong>
    <p>A 股用人民币。H 股用港元，这里先乘汇率，换成人民币再比较。只保留两地都开市的日子。</p>
  </article>
  <article>
    <strong>2. 看价差会不会回家</strong>
    <p>协整的大白话是：两条价格可以各自乱走，但它们的差距常常会回到一条旧轨道附近。</p>
  </article>
  <article>
    <strong>3. 偏得太多就试着对做</strong>
    <p>A 股相对贵，就假设以后变便宜：卖 A、买 H。A 股相对便宜，就反过来。回到附近就平仓。</p>
  </article>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"**用 {company} 现在这组数字对照着看**")
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"**最近溢价 {_pct(premium_last)}**\n\nA 股比换成人民币后的 H 股贵多少。正数是 A 更贵，负数是 H 更贵。"
    )
    c2.markdown(
        f"**EG p 值 {p_text}**\n\n越小，越能说这段时间价差像会回家。0.05 是常用分界。现在这页{same_or_not}。"
    )
    life_line = (
        f"**半衰期 {life} 个交易日**\n\n如果价差真会回家，大概要多久走完一半路程。数字很大，就是回得很慢。"
        if life not in {"—", "不收敛"}
        else "**半衰期还看不出来**\n\n这段价差不像会稳定地回家，所以没有一个有意义的“几天回到一半”。"
    )
    c3.markdown(life_line)
    st.markdown("**页面上其他词**")
    st.markdown(
        """
- **β（贝塔）**：H 股动 1%，A 股大约跟着动多少。用来决定两边各买多少，而不是两边买一样多。
- **z 分数**：价差离开自己最近平均水平有几个标准差。绝对值越大，偏得越厉害。
- **夏普**：收益相对波动有多划算。负数就是这段回测在亏，而且亏得不稳。
- **最大回撤**：净值从高点掉下去最深的那段。
- **北向资金**：外资通过沪深港通买 A 股的净额。这里只是和价差放在一起看，不是证明它造成了价差。
        """
    )
    st.markdown("**建议这样点**")
    st.markdown(
        f"""
1. 先留在这一页，看懂 {company} 的溢价和 p 值。
2. 打开「价差与信号」。浅色带是平常波动范围，三角是开仓，叉是平仓。
3. 打开「回测」，看扣掉费用之后的净值。旁边还有只买 A、只买 H、两边各买一半的对照。
4. 左侧可以换成工行、招行、石油。换完后，页顶会告诉你 p 值变大了还是变小了。想看这个关系换窗口、换子样本之后还在不在，打开「稳定性」。
        """
    )
    st.markdown(
        """
<div class="lesson">
  <article>
    <strong>不是无风险套利</strong>
    <p>A 股和 H 股不能换成彼此。价差可以高很久，也可以继续走远。</p>
  </article>
  <article>
    <strong>不是投资建议</strong>
    <p>这里的买卖是历史回测。亏钱的结果也会照实显示，不代表以后该怎么下单。</p>
  </article>
  <article>
    <strong>p 值小也不等于能赚钱</strong>
    <p>就算价差像会回家，印花税、佣金和借券费仍可能把利润吃掉。</p>
  </article>
</div>
        """,
        unsafe_allow_html=True,
    )
    _render_quick_compare("compare_from_easy")

with tab_px:
    st.plotly_chart(
        core_signal_figure(frame, params.z_window, params.entry_z, params.exit_z, params.stop_z),
        use_container_width=True,
        key="chart_core",
    )
    st.caption(
        "上图是交易价差：点线是滚动均值，三条色带从内到外是 ±1σ、±1.5σ、±2σ。"
        "上三角是买 A、卖 H，下三角是卖 A、买 H，叉是平仓。绿底是做多价差的持仓，红底是做空价差的持仓。"
        "下图是同一个 z-score：红虚线是开仓，灰点线是平仓，金点线是止损。"
    )
    st.plotly_chart(prices_figure(frame), use_container_width=True, key="chart_prices")
    st.caption("两条价格都从这段区间的第一天记为 100。深蓝是 A 股，暖橙是 H 股，H 股已经乘过汇率，换成人民币。")
    left, right = st.columns(2)
    left.plotly_chart(premium_figure(frame), use_container_width=True, key="chart_premium")
    right.plotly_chart(hedge_figure(frame, active=params.hedge), use_container_width=True, key="chart_hedge_signal")
    spread_bytes = spread_png(frame, params.z_window)
    if spread_bytes:
        st.download_button(
            "下载价差图 PNG",
            spread_bytes,
            file_name=f"{picked.a_code}_spread.png",
            mime="image/png",
            key="spread_png_signal",
        )

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
        st.plotly_chart(rolling_p_figure(rolling_p), use_container_width=True, key="chart_rolling_p_diag")
        st.caption("每个点用它之前 252 个同时开市日做 Engle–Granger，大约每 42 个交易日算一次。虚线是 0.05。")
    else:
        st.caption("评估区间不足 252 个交易日，没有画滚动协整 p 值。")
    st.plotly_chart(static_resid_figure(frame), use_container_width=True, key="chart_resid_diag")

with tab_stable:
    st.markdown(
        "这一页回答两件事：这段样本里的关系本身稳不稳，以及换窗口、换子样本、去掉极端日之后还剩多少。"
    )
    st.markdown("**滚动关系**")
    st.caption(
        f"半衰期来自 AR(1)：价差_t = c + φ × 价差_(t−1) + e。"
        f"全样本残差 φ = {_num(static_hl.get('phi'), 3)}，半衰期 = −ln(2) / ln(φ) = {_half_life_text(static_hl)}。"
        "φ 越接近 1，回复越慢。交易用的是另一条因果价差，半衰期写在诊断页右边。"
    )
    adf = result.get("residual_adf") or {}
    st.caption(
        f"残差 ADF 统计量 {_num(adf.get('adf_stat'), 2)}，p 值 {_pval(adf.get('adf_pvalue'))}。"
        + (adf.get("note") or "残差 ADF 不能代替 Engle–Granger p 值。")
    )
    rolling_p = result.get("rolling_p")
    if rolling_p is not None and len(rolling_p):
        st.plotly_chart(rolling_p_figure(rolling_p), use_container_width=True, key="chart_rolling_p_stable")
        st.caption("每个点只用它之前 252 个同时开市日做 Engle–Granger，大约每 42 个交易日算一次。虚线是 0.05。")
    else:
        st.caption("评估区间不足 252 个交易日，没有画滚动协整 p 值。")
    st.plotly_chart(hedge_figure(frame, active=params.hedge), use_container_width=True, key="chart_hedge_stable")
    used = "卡尔曼滤波" if params.hedge == "kalman" else "滚动 OLS"
    st.caption(f"两条 β 都只用当天以前的价格。现在仓位用的是{used}，图例里标了「交易用」。")
    st.plotly_chart(static_resid_figure(frame), use_container_width=True, key="chart_resid_stable")
    st.markdown("**扰动**")
    st.markdown(
        "不只做一次协整检验。系统会换窗口长度、换终点，再在全区间上切子样本、去掉极端日，"
        "看 5% 水平上还有多少情形能拒绝“没有协整”。"
    )
    stability_key = f"{picked.a_code}|{start_day}|{end_day}"
    if len(frame) < 200:
        st.info("这段评估区间太短，稳定性地图至少需要大约 200 个同时开市日。")
    elif st.button("计算稳定性地图", key="run_stability"):
        with st.spinner("正在换窗口重算协整，大约需要几秒到十几秒…"):
            grid = stability_grid(frame["log_a"], frame["log_h"], frame.index)
            st.session_state["stability"] = {
                "key": stability_key,
                "grid": grid,
                "report": fragility_report(grid),
                "by_length": length_summary(grid),
                "checks": slice_checks(frame),
            }
    saved_stability = st.session_state.get("stability")
    if isinstance(saved_stability, dict) and saved_stability.get("key") != stability_key:
        st.caption("股票或日期变了，需要重新计算。")
        saved_stability = None
    if isinstance(saved_stability, dict):
        report = saved_stability["report"]
        rate = report["reject_rate"]
        fragility = report["fragility"]
        if not np.isfinite(fragility):
            words = "算不出来"
        elif fragility < 0.25:
            words = "比较稳"
        elif fragility < 0.6:
            words = "不太稳"
        else:
            words = "很脆弱"
        _kpis(
            [
                ("成立比例", _pct(rate), "pos" if np.isfinite(rate) and rate >= 0.5 else "neg"),
                ("脆弱程度", _pct(fragility), "neg" if np.isfinite(fragility) and fragility >= 0.5 else ""),
                ("判断", words, ""),
                ("检验次数", f"{report['n_tests']}", ""),
            ],
            columns=4,
        )
        worst = report["worst"]
        if isinstance(worst, pd.DataFrame) and not worst.empty:
            top = worst.iloc[0]
            st.markdown(
                f"最危险的窗口是 **{pd.Timestamp(top['start']).date()} 至 {pd.Timestamp(top['end']).date()}**，"
                f"长度 {int(top['length'])} 个交易日，p 值 {_pval(top['pvalue'])}。"
            )
        st.plotly_chart(stability_heatmap(saved_stability["grid"]), use_container_width=True, key="chart_stability")
        by_length = saved_stability.get("by_length")
        if isinstance(by_length, pd.DataFrame) and not by_length.empty:
            st.plotly_chart(reject_rate_figure(by_length), use_container_width=True, key="chart_reject_rate")
            st.caption("柱子是该窗口长度上拒绝无协整的比例。暖橙虚线是 50%。")
        st.caption(
            "颜色越偏深蓝，p 值越小。0.05 以下才算在这个窗口里拒绝了无协整。"
            "这里为了重复计算，每个窗口的滞后都固定为 1 阶，和诊断页里用 AIC 选滞后的那一次检验不完全相同。"
        )
        if isinstance(worst, pd.DataFrame) and not worst.empty:
            show_worst = worst.copy()
            show_worst["开始"] = pd.to_datetime(show_worst["start"]).dt.date
            show_worst["结束"] = pd.to_datetime(show_worst["end"]).dt.date
            show_worst["p 值"] = show_worst["pvalue"].map(_pval)
            st.markdown("**p 值最高的窗口**")
            st.dataframe(
                show_worst[["开始", "结束", "length", "p 值"]].rename(columns={"length": "长度"}),
                hide_index=True,
                use_container_width=True,
            )
        checks = saved_stability["checks"].copy()
        checks["p 值"] = checks["pvalue"].map(_pval)
        checks["5% 结论"] = checks["reject_5"].map({True: "拒绝无协整", False: "不拒绝"})
        st.markdown("**子样本和极端日**")
        st.caption("下面每一行的滞后也固定为 1 阶，所以「全区间」的 p 值可以和页顶那次 AIC 选滞后的 EG p 值不一样。")
        st.dataframe(checks[["情形", "天数", "p 值", "5% 结论"]], hide_index=True, use_container_width=True)
        st.caption("这些检验共用同一段价格。成立比例高，只说明关系对换窗口不那么敏感，不代表样本外一定还在。")

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
            ("年化换手", _num(stats["turnover"], 2), ""),
            ("Calmar", _num(stats["calmar"], 2), _tone(stats["calmar"])),
        ],
        columns=4,
    )
    st.plotly_chart(equity_figure(frame), use_container_width=True, key="chart_equity")
    short_text = "允许做空 A 股。" if params.allow_short_a else "不允许做空 A 股：价差偏高时不交易。"
    cost_text = "已计入印花税、佣金、融券费和滑点。" if include_costs else "未扣交易成本，看到的是毛结果。"
    t1_text = "已按 T+1：下一个同时开市日收盘才成交。" if params.fill == "next_close" else "未按 T+1：信号当日收盘就成交，这比实盘更乐观。"
    st.caption(
        f"{frame.index.min().date()} 至 {frame.index.max().date()}，{stats['n_days']} 个同时开市日。"
        f"夏普按 252 天年化，无风险利率按 0。Calmar 是年化收益除以最大回撤的绝对值。{cost_text}{t1_text}{short_text}"
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
    trade_csv = show.to_csv(index=False).encode("utf-8-sig") if not trades.empty else "开仓,平仓\n".encode("utf-8-sig")
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
    st.download_button(
        "交易明细 CSV",
        trade_csv,
        file_name=f"{picked.a_code}_trades.csv",
        mime="text/csv",
    )
    if spread_bytes and equity_bytes:
        p1, p2 = st.columns(2)
        p1.download_button(
            "价差图 PNG",
            spread_bytes,
            file_name=f"{picked.a_code}_spread.png",
            mime="image/png",
            key="spread_png_backtest",
        )
        p2.download_button(
            "净值图 PNG",
            equity_bytes,
            file_name=f"{picked.a_code}_equity.png",
            mime="image/png",
            key="equity_png_backtest",
        )

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
        st.plotly_chart(northbound_figure(frame), use_container_width=True, key="chart_northbound")
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
    _render_quick_compare("compare_from_scan")
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

**协整。** 全样本 Engle–Granger 检验的原假设是「没有协整」。回归是 `log A = α + β log H + e`，含常数项，滞后用 AIC 选择。报告的 p 值是 MacKinnon 协整 p 值。残差上再做一次普通 ADF 会偏乐观，因为残差是同一段样本估出来的，不能拿来代替 EG p 值。Johansen 迹检验含常数项，差分滞后由 VAR 的 AIC 减 1。半衰期来自 AR(1)，`半衰期 = −ln 2 / ln φ`，只在 0 < φ < 1 时有限。

**交易和前视。** 全样本 β、EG、Johansen 和静态残差用了评估区间里的全部价格，只能当描述。回测用的 β 是滚动 OLS 或卡尔曼滤波，只用当天以前的价格；两条路径都会画出来，仓位只用侧栏选中的那一条。z 分数的滚动均值和标准差含当天的价差，成交放在信号之后。默认下一个同时开市日收盘才成交，所以 A 股不会在买入的同一根日 K 里卖出。选「未考虑：当日收盘」会更乐观。

**成本和日历。** 默认计入 A 股卖方印花税和佣金、H 股印花税和费用、融券费、汇率与成交滑点。印花税按日期分档，侧栏可以关掉整组成本。A、H 节假日不一致的日期整行删掉，股价不向前填充。汇率最多向前填 5 个交易日，并且要求中间价落在合理区间。

**数据来源。** 价格优先用 Yahoo 复权收盘价，取不到时用东财前复权。股票池快照来自东财 A+H 行情，腾讯没有对应 A 股，不在池子里。北向是两市合计净买入，单位亿元，不是这只股票自己的持仓；2024 年 8 月 16 日之后日频流向停止披露。

**稳定性。** 「稳定性」里换窗口长度、换终点、切半段、去掉极端日，看 5% 上拒绝无协整的比例。脆弱程度 = 1 − 成立比例。每个窗口的 Engle–Granger 用了该窗口里的全部价格，滞后固定为 1 阶，所以可以和页顶那次 AIC 选滞后的 p 值不一样。这些检验共用同一段价格，成立比例高只说明关系对换样本不那么敏感，不是样本外证据。

**多重比较。** 开仓阈值敏感性，以及「一组对照」里按 p 值排序，都是同一段样本上的多次试验。最小的 p 值或最好的夏普会被挑得偏乐观。

**不要读成什么。** 样本内协整不保证以后还在。A/H 不能转换，溢价可以在高位停很多年。做空 A 股取决于能不能借到券。

**权重。** 做多价差是买 A、按 β 卖 H，`|w_A| + |w_H| = 1`。持仓期间把开仓时的 β 锁住，只有 β 的相对变化超过再平衡带才调整。

**北向。** 回归只描述价差变化和全市场北向是否同一天同向。标准误是 HAC（5 阶）。
        """
    )
    changelog_path = Path(__file__).resolve().parent / "CHANGELOG.md"
    if changelog_path.exists():
        with st.expander("版本记录"):
            st.markdown(changelog_path.read_text(encoding="utf-8"))

st.markdown(
    '<p class="site-footer">Part of <a href="https://eugenefinance.tech">EF Lab · eugenefinance.tech</a> · V0.8 · 仅供学术研究与学习使用，不构成任何投资建议。历史回测不代表未来表现。</p>',
    unsafe_allow_html=True,
)
