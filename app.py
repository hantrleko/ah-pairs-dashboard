"""A+H 协整与动态配对交易看板。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

from ah_pairs.analysis import warmup_start
from ah_pairs.backtest import signal_token, trade_token
from ah_pairs.data import load_pair
from ah_pairs.freshness import cache_written_at, northbound_stops_before, price_is_stale
from ah_pairs.i18n import set_lang, tx
from ah_pairs.northbound import latest_valid
from ah_pairs.notes import parse_notes
from ui.cache import cached_alt_beta, cached_backtest, cached_northbound, cached_pair, cached_signals
from ui.format import kpis, pct, pval, span_frame, tone
from ui.sidebar import render_sidebar
from ui.style import inject_css
from ui.tabs import (
    render_backtest,
    render_diagnose,
    render_market,
    render_method,
    render_scan,
    render_signal,
    render_stability,
    render_summary,
)

st.set_page_config(page_title="EF Lab | A+H", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
if "lang" not in st.session_state:
    requested = st.query_params.get("lang")
    st.session_state["lang"] = requested if requested in {"zh", "zh-Hant", "en"} else "zh"
set_lang(st.session_state["lang"])
inject_css()
LOGO = str(Path(__file__).resolve().parent / "assets" / "eugene-finance-logo.jpg")

with st.sidebar:
    choice = render_sidebar(LOGO)

logo_col, title_col = st.columns([1, 5], vertical_alignment="center")
with logo_col:
    st.image(LOGO, width=156)
with title_col:
    st.markdown(
        tx(
            """
<div class="brand">
  <p class="kicker">V1.4 · EUGENE FINANCE</p>
  <h1>EF Lab · A+H 股协整配对交易研究看板</h1>
  <p class="subtitle">Interactive Cointegration &amp; Dynamic Pairs Trading Lab · 仅供学习与研究，不构成投资建议</p>
</div>
            """,
            """
<div class="brand">
  <p class="kicker">V1.4 · EUGENE FINANCE</p>
  <h1>EF Lab · A+H cointegration and pairs research</h1>
  <p class="subtitle">For study and research. Not investment advice.</p>
</div>
            """,
        ),
        unsafe_allow_html=True,
    )

params = choice["params"]
picked = choice["picked"]
start_day, end_day = choice["start_day"], choice["end_day"]
warm = warmup_start(start_day, params).strftime("%Y-%m-%d")
if choice["reload"]:
    try:
        load_pair(picked.a_code, picked.h_code, "2010-01-01", str(end_day), refresh=True)
        cached_pair.clear()
        cached_signals.clear()
        cached_backtest.clear()
        cached_alt_beta.clear()
    except Exception as exc:
        st.error(tx(f"重新下载失败：{exc}", f"Download failed: {exc}"))

north = None
north_error = ""
try:
    north = cached_northbound()
    north_key = str(pd.Timestamp(north.dropna().index.max()).date()) if north is not None and north.dropna().size else ""
except Exception as exc:
    north_error = str(exc)
    north_key = ""

try:
    with st.spinner(tx("读取行情并计算协整与回测…", "Loading prices and computing the backtest…")):
        signal_key = signal_token(params)
        notes, prepared = cached_signals(
            picked.a_code,
            picked.h_code,
            warm,
            str(end_day),
            str(pd.Timestamp(start_day).date()),
            signal_key,
            north_key,
        )
        frame_bt, trades, stats, sensitivity = cached_backtest(prepared["frame"], trade_token(params))
    result = dict(prepared)
    result.update(frame=frame_bt, trades=trades, stats=stats, sensitivity=sensitivity, params=params)
except Exception as exc:
    st.error(str(exc))
    st.stop()

frame = result["frame"]
eg = result["eg"]
stats = result["stats"]
parsed = parse_notes(notes)
sources = parsed["sources"]
note_meta = parsed["meta"]
last_px = frame.index.max()
price_stale = price_is_stale(last_px, end_day, pd.Timestamp.today())
nb_last = latest_valid(north) if north is not None else None
nb_stale = northbound_stops_before(last_px, nb_last)
nb_text = tx("没有载入", "not loaded") if nb_last is None else str(pd.Timestamp(nb_last).date())
cache_stamp = cache_written_at(picked.a_code, picked.h_code) or tx("无本地缓存", "no local cache")
common = frame.loc[frame["both_open"]] if "both_open" in frame.columns else frame
one_sided = int((~frame["both_open"]).sum()) if "both_open" in frame.columns else 0
premium_now = common["premium"].dropna() if "premium" in common.columns else frame["premium"].dropna()
premium_last = float(premium_now.iloc[-1]) if len(premium_now) else np.nan
price_tag = tx('<span class="tag bad">行情偏旧</span>', '<span class="tag bad">stale prices</span>') if price_stale else tx('<span class="tag">行情及时</span>', '<span class="tag">prices current</span>')
nb_tag = tx('<span class="tag bad">北向已停止披露</span>', '<span class="tag bad">northbound disclosure ended</span>') if nb_stale else tx('<span class="tag">北向覆盖到行情截止</span>', '<span class="tag">northbound covers the prices</span>')
st.markdown(
    tx(
        f"""
<div class="mast">
  <div>
    <p class="kicker">V1.4 · 这一对</p>
    <p class="pair-name">{picked["name"]}</p>
    <p class="meta">{picked.a_code} · {picked.h_code} · {len(common)} 个同时开市日，另有 {one_sided} 天只有一边开市。协整不用这些单边日；已有仓位仍计入开市一边的涨跌。</p>
  </div>
  <div class="pill"><span>最近溢价</span><strong>{pct(premium_last)}</strong></div>
</div>
<div class="toolbar">
  <strong>{picked["name"]}</strong>
  <span>评估 {start_day} — {end_day}</span>
  <span>A 股 {sources.get("a", "—")} · H 股 {sources.get("h", "—")} · 汇率 {sources.get("fx", "—")}</span>
  <span>行情 {frame.index.min().date()} — {last_px.date()}</span>
  {price_tag}
  <span>北向截至 {nb_text}</span>
  {nb_tag}
  <span>缓存写入 {cache_stamp}</span>
</div>
        """,
        f"""
<div class="mast">
  <div>
    <p class="kicker">V1.4 · this pair</p>
    <p class="pair-name">{picked["name"]}</p>
    <p class="meta">{picked.a_code} · {picked.h_code} · {len(common)} common sessions, plus {one_sided} one-sided days. Cointegration ignores those days; an open position still earns the side that traded.</p>
  </div>
  <div class="pill"><span>Premium</span><strong>{pct(premium_last)}</strong></div>
</div>
<div class="toolbar">
  <strong>{picked["name"]}</strong>
  <span>{start_day} — {end_day}</span>
  <span>A {sources.get("a", "—")} · H {sources.get("h", "—")} · FX {sources.get("fx", "—")}</span>
  <span>Prices {frame.index.min().date()} — {last_px.date()}</span>
  {price_tag}
  <span>Northbound through {nb_text}</span>
  {nb_tag}
  <span>Cache {cache_stamp}</span>
</div>
        """,
    ),
    unsafe_allow_html=True,
)
if price_stale:
    st.markdown(
        tx(
            f'<p class="health warn">行情最后一天是 {last_px.date()}，比所选结束日或今天落后超过 7 天。</p>',
            f'<p class="health warn">The last price is {last_px.date()}, more than 7 days behind the selected end or today.</p>',
        ),
        unsafe_allow_html=True,
    )
if note_meta.get("fx_lag"):
    lagged = int(note_meta["fx_lag"])
    st.caption(tx(f"汇率有 {lagged} 天沿用了此前最多 5 个交易日的牌价。", f"The exchange rate was carried forward for {lagged} days, at most five sessions."))
if "download_common" in note_meta:
    st.caption(
        tx(
            f"含预热的下载区间：同时开市 {note_meta['download_common']} 天，只有一边开市 {note_meta['download_one_sided']} 天。协整和信号只用同时开市日；已有仓位在单边开市日仍计算开市一边的收益，当天不调仓。",
            f"Download window, including warm-up: {note_meta['download_common']} common sessions and {note_meta['download_one_sided']} one-sided days. Cointegration and signals use common sessions. An open position still earns the side that traded, and that day does not rebalance.",
        )
    )
if note_meta.get("download_failed"):
    st.caption(tx("有一条行情下载失败，图上用的是本地缓存。", "One download failed, so the chart is using the local cache."))
if note_meta.get("adjust_mismatch"):
    st.caption(tx("两条腿的复权来源可能不一致。Yahoo 复权价含分红再投资，东财用前复权，价差水平要谨慎解读。", "The two legs may not use the same adjustment. Yahoo reinvests dividends; East Money uses a forward adjustment. Read the spread level with care."))
elif not any(str(note).startswith("meta:") for note in notes):
    for note in parsed["prose"]:
        st.caption(note)
if eg["reject_5"]:
    coint_words = tx(f"5% 水平上拒绝无协整（EG p = {pval(eg['pvalue'])}）", f"rejects no-cointegration at 5% (EG p = {pval(eg['pvalue'])})")
    note_class = "note ok"
else:
    coint_words = tx(f"目前未能在 5% 水平上拒绝无协整（EG p = {pval(eg['pvalue'])}）", f"does not reject no-cointegration at 5% (EG p = {pval(eg['pvalue'])})")
    note_class = "note"
if not np.isfinite(stats["total_return"]):
    pay_words = tx("此区间没有可计算的回测报酬", "this window has no computable backtest return")
elif stats["total_return"] < 0:
    pay_words = tx("此区间回测为负报酬", "the backtest return in this window is negative")
elif stats["total_return"] > 0:
    pay_words = tx("此区间回测为正报酬", "the backtest return in this window is positive")
else:
    pay_words = tx("此区间回测报酬接近零", "the backtest return in this window is about zero")
short_words = tx("本回测允许做空 A 股，多数标的不一定能融到券，这个假设偏乐观。", "This backtest allows short A-shares. Many names cannot be borrowed, so the assumption is optimistic.") if params.allow_short_a else tx("本回测不允许做空 A 股。", "This backtest does not short A-shares.")
hedge_name = tx("滚动 OLS", "rolling OLS") if params.hedge == "rolling" else tx("卡尔曼滤波", "Kalman")
fill_name = tx("次日收盘", "next close") if params.fill == "next_close" else tx("当日收盘", "same close")
cost_name = tx("法定成本", "statutory costs") if choice["include_costs"] else tx("零成本", "zero costs")
spec_line = tx(
    f"当前规格：{hedge_name}，估计窗口 {int(params.hedge_window)} 日，z 窗口 {int(params.z_window)} 日，开仓 {params.entry_z:g}，平仓 {params.exit_z:g}，{fill_name}，{cost_name}。",
    f"Spec: {hedge_name}, estimation window {int(params.hedge_window)} days, z window {int(params.z_window)} days, entry {params.entry_z:g}, exit {params.exit_z:g}, {fill_name}, {cost_name}.",
)
tail = tx("p 值不是获利概率，也不保证价差会收敛。", "The p-value is not the probability of profit, and it does not mean the spread will converge.")
st.markdown(
    f'<p class="{note_class}">{coint_words}{tx("；", "; ")}{pay_words}{tx("。", ". ")}{short_words}{tx("", " ")}{tail}</p>',
    unsafe_allow_html=True,
)
st.caption(spec_line)
sharpe_text = f"{stats['sharpe']:.2f}" if np.isfinite(stats["sharpe"]) else "—"
kpis(
    [
        (tx("EG p 值", "EG p-value"), pval(eg["pvalue"]), ""),
        (tx("净收益", "Net return"), pct(stats["total_return"]), tone(stats["total_return"])),
        (tx("夏普", "Sharpe"), sharpe_text, tone(stats["sharpe"])),
        (tx("最大回撤", "Max drawdown"), pct(stats["max_drawdown"]), "neg" if np.isfinite(stats["max_drawdown"]) else ""),
    ],
    columns=4,
)
st.caption(tx("EG p 值检验的是「没有协整」。净收益是期末净值减 1。半衰期、残差 ADF 和 Johansen 在「协整诊断」。", "The EG p-value tests “no cointegration”. Net return is ending equity minus 1. Half-life, residual ADF and Johansen are under Diagnostics."))
prev = st.session_state.get("pair_snapshot")
if prev and prev.get("code") != picked.a_code:
    st.caption(tx(f"刚从{prev['name']}换到{picked['name']}：EG p 从 {prev['p']} 变为 {pval(eg['pvalue'])}。p 更小只说明这段样本更接近协整。", f"Switched from {prev['name']} to {picked['name']}: EG p moved from {prev['p']} to {pval(eg['pvalue'])}. A smaller p-value only means this sample is closer to cointegration."))
st.session_state["pair_snapshot"] = {"code": picked.a_code, "name": picked["name"], "p": pval(eg["pvalue"])}

ctrl_left, ctrl_right = st.columns([3, 1])
with ctrl_left:
    chart_span = st.radio(
        tx("图表区间", "Chart span"),
        ["1y", "3y", "all"],
        format_func=lambda key: {"1y": tx("近 1 年", "1 year"), "3y": tx("近 3 年", "3 years"), "all": tx("全期间", "Full sample")}[key],
        horizontal=True,
        index=2,
        key="chart_span_code",
    )
with ctrl_right:
    show_marks = st.checkbox(tx("显示开平仓标记", "Show trade marks"), value=True)
show_alt = st.checkbox(
    tx("对照另一条 β", "Compare the other beta"),
    value=False,
    key="show_alt_beta",
    help=tx("默认只算侧栏选中的那条。勾上之后再补滚动 OLS 或卡尔曼里的另一条，第一次会多花一点时间。", "By default only the selected hedge is estimated. This adds the other one, which takes a little longer the first time."),
)
if show_alt:
    alt = cached_alt_beta(
        picked.a_code,
        picked.h_code,
        warm,
        str(end_day),
        str(pd.Timestamp(start_day).date()),
        signal_token(params),
    )
    for column in alt.columns:
        if column not in frame.columns:
            frame[column] = alt.reindex(frame.index)[column]
    result["frame"] = frame
focus = st.session_state.get("chart_focus")
if isinstance(focus, dict) and focus.get("start") and focus.get("end"):
    st.caption(tx(f"图表目前锁定在 {focus['start']} 至 {focus['end']}。", f"The chart is locked to {focus['start']} – {focus['end']}."))
    if st.button(tx("恢复图表区间", "Clear the chart lock"), key="clear_chart_focus"):
        st.session_state["chart_focus"] = None
        st.rerun()

page = SimpleNamespace(
    picked=picked,
    universe=choice["universe"],
    params=params,
    result=result,
    frame=frame,
    view=span_frame(frame, chart_span),
    common=common,
    sources=sources,
    north=north,
    north_error=north_error,
    include_costs=choice["include_costs"],
    chart_span=chart_span,
    show_marks=show_marks,
    cache_stamp=cache_stamp,
    warm=warm,
    start_day=start_day,
    end_day=end_day,
    last_px=last_px,
    premium_last=premium_last,
    eg=eg,
    stats=stats,
    static_hl=result["half_life_static"],
    traded_hl=result["half_life_traded"],
    johansen=result["johansen"],
    lang=st.session_state["lang"],
)

tabs = st.tabs(
    [
        tx("摘要", "Summary"),
        tx("协整诊断", "Diagnostics"),
        tx("信号", "Signal"),
        tx("稳定性", "Stability"),
        tx("回测", "Backtest"),
        tx("方法", "Method"),
        tx("市场对照", "Market"),
        tx("一组对照", "Pairs"),
    ]
)
with tabs[0]:
    render_summary(page)
with tabs[1]:
    render_diagnose(page)
with tabs[2]:
    render_signal(page)
with tabs[3]:
    render_stability(page)
with tabs[4]:
    render_backtest(page)
with tabs[5]:
    render_method()
with tabs[6]:
    render_market(page)
with tabs[7]:
    render_scan(page)

st.markdown(
    tx(
        '<p class="site-footer">Part of <a href="https://eugenefinance.tech">EF Lab · eugenefinance.tech</a> · V1.4 · 仅供学术研究与学习使用，不构成任何投资建议。历史回测不代表未来表现。</p>',
        '<p class="site-footer">Part of <a href="https://eugenefinance.tech">EF Lab · eugenefinance.tech</a> · V1.4 · Research and study only. Not investment advice. A backtest is not a forecast.</p>',
    ),
    unsafe_allow_html=True,
)
