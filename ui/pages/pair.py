"""单对研究页。抬头、规格条、结论卡、解释条、主图、分区切换。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

from ah_pairs.analysis import warmup_start
from ah_pairs.backtest import signal_token, trade_token
from ah_pairs.data import load_pair
from ah_pairs.freshness import cache_written_at, northbound_stops_before, price_is_stale
from ah_pairs.i18n import tx
from ah_pairs.northbound import latest_valid
from ah_pairs.notes import parse_notes
from ah_pairs.plots import core_signal_figure
from ui.cache import cached_alt_beta, cached_backtest, cached_northbound, cached_pair, cached_signals
from ui.explain import render_v14_live_card
from ui.format import kpis, num, pct, pval, show_chart, span_frame, tone, QUICK
from ui.spec_dialog import spec_dialog
from ui.state import current_pair, load_universe_filtered, network_allowed, spec_state, sync_query
from ui.tabs.backtest import render_backtest
from ui.tabs.diagnose import render_diagnose
from ui.tabs.market import render_market
from ui.tabs.scan import render_scan
from ui.tabs.signal import render_signal
from ui.tabs.stability import render_stability
from ui.tabs.summary import render_summary

from ah_pairs.presets import PRESETS, get_preset


def _spec_line(params, include_costs: bool) -> str:
    hedge_name = tx("滚动 OLS", "rolling OLS") if params.hedge == "rolling" else tx("卡尔曼滤波", "Kalman")
    fill_name = tx("次日收盘", "next close") if params.fill == "next_close" else tx("当日收盘", "same close")
    cost_name = tx("含成本", "costs on") if include_costs else tx("零成本", "zero costs")
    return tx(
        f"{hedge_name} · 估计 {int(params.hedge_window)} · z {int(params.z_window)} · 入 {params.entry_z:g} / 出 {params.exit_z:g} · {fill_name} · {cost_name}",
        f"{hedge_name} · est {int(params.hedge_window)} · z {int(params.z_window)} · in {params.entry_z:g} / out {params.exit_z:g} · {fill_name} · {cost_name}",
    )


def _is_frozen_v14(params, start_day, include_costs: bool) -> bool:
    try:
        from ah_pairs.desk import default_desk_params, is_default_spec

        return bool(include_costs) and is_default_spec(params, start_day)
    except Exception:
        return False


def render() -> None:
    universe = load_universe_filtered()
    if universe.empty:
        st.error(tx("股票池为空。", "The universe is empty."))
        return
    params, (start_day, end_day), include_costs = spec_state()

    # ---- 选择条：快捷 + 全名单 ----
    codes = universe["a_code"].tolist()
    a_code = current_pair(universe)
    top = st.columns([2.2, 1.2])
    with top[0]:
        pick_cols = st.columns(len(QUICK) + 1)
        for i, (code, name_zh, name_en) in enumerate(QUICK):
            if code in codes and pick_cols[i].button(tx(name_zh, name_en), key=f"pq_{code}", width="stretch"):
                st.session_state["pair_select"] = code
                st.rerun()
    with top[1]:
        labels = {row.a_code: f"{row.name}   {row.a_code} / {row.h_code}" for row in universe.itertuples(index=False)}
        st.selectbox(tx("公司", "Company"), codes, format_func=lambda code: labels.get(code, code), key="pair_select", label_visibility="collapsed")
    a_code = current_pair(universe)
    picked = universe.loc[universe["a_code"] == a_code].iloc[0]

    # ---- 计算（仅本页） ----
    engine = st.session_state.get("engine_version", "v14")
    warm = warmup_start(start_day, params).strftime("%Y-%m-%d")
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
                picked.a_code, picked.h_code, warm, str(end_day), str(pd.Timestamp(start_day).date()), signal_key, north_key,
            )
            borrow_token = ""
            if params.allow_short_a:
                from ah_pairs.borrow import borrow_token_for

                borrow_token = borrow_token_for(prepared["frame"].index, picked.a_code)
            frame_bt, trades, stats, sensitivity = cached_backtest(prepared["frame"], trade_token(params), borrow_token)
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
    last_px = frame.index.max()
    price_stale = price_is_stale(last_px, end_day, pd.Timestamp.today())
    nb_last = latest_valid(north) if north is not None else None
    common = frame.loc[frame["both_open"]] if "both_open" in frame.columns else frame
    premium_now = common["premium"].dropna() if "premium" in common.columns else frame["premium"].dropna()
    premium_last = float(premium_now.iloc[-1]) if len(premium_now) else np.nan

    # ---- 抬头 ----
    head = st.columns([4, 1.4])
    with head[0]:
        st.markdown(
            f'<p class="ef-pairname">{picked["name"]}</p><p class="ef-pairmeta">{picked.a_code} · {picked.h_code}</p>',
            unsafe_allow_html=True,
        )
    with head[1]:
        st.markdown(
            tx(
                f'<div class="ef-premium"><span class="lbl">最近溢价</span><span class="val">{pct(premium_last)}</span></div>',
                f'<div class="ef-premium"><span class="lbl">PREMIUM</span><span class="val">{pct(premium_last)}</span></div>',
            ),
            unsafe_allow_html=True,
        )

    # ---- 规格条 + 引擎 ----
    spec_cols = st.columns([3.4, 1.2])
    with spec_cols[0]:
        frozen = _is_frozen_v14(params, start_day, include_costs) and engine == "v14"
        chip = tx('<span class="ef-chip solid">冻结预设 v14-default</span>', '<span class="ef-chip solid">frozen preset v14-default</span>') if frozen else tx('<span class="ef-chip warn">探索规格</span>', '<span class="ef-chip warn">custom spec</span>')
        st.markdown(f'<div class="ef-spec">{chip}<span>{_spec_line(params, include_costs)}</span></div>', unsafe_allow_html=True)
        sbtns = st.columns([1, 1, 3])
        if sbtns[0].button(tx("调整规格…", "Adjust spec…"), key="open_spec"):
            spec_dialog()
        with sbtns[1].popover(tx("数据来源", "Data sources")):
            cache_stamp = cache_written_at(picked.a_code, picked.h_code) or tx("无本地缓存", "no local cache")
            nb_text = tx("没有载入", "not loaded") if nb_last is None else str(pd.Timestamp(nb_last).date())
            one_sided_n = int((~frame["both_open"]).sum()) if "both_open" in frame.columns else 0
            st.caption(tx(f"评估 {start_day} — {end_day}。A 股 {sources.get('a', '—')}，H 股 {sources.get('h', '—')}，汇率 {sources.get('fx', '—')}。行情 {frame.index.min().date()} — {last_px.date()}。北向截至 {nb_text}。缓存 {cache_stamp}。", f"{start_day} — {end_day}. A {sources.get('a', '—')}, H {sources.get('h', '—')}, FX {sources.get('fx', '—')}. Prices {frame.index.min().date()} — {last_px.date()}. Northbound through {nb_text}. Cache {cache_stamp}."))
            st.caption(tx(f"{len(common)} 个同时开市日，另有 {one_sided_n} 天只有一边开市。", f"{len(common)} common sessions, plus {one_sided_n} one-sided days."))
            if north_error:
                st.caption(tx(f"北向没有读到：{north_error}", f"Northbound was not read: {north_error}"))
    with spec_cols[1]:
        engine = st.segmented_control(
            tx("引擎", "Engine"), ["v14", "v15"], default=engine,
            format_func=lambda c: "V1.4" if c == "v14" else "V1.5",
            key="engine_version", label_visibility="collapsed",
        ) or "v14"

    if price_stale:
        st.markdown(
            tx(f'<div class="ef-notice warn"><p>行情最后一天是 {last_px.date()}，比所选结束日或今天落后超过 7 天。</p></div>', f'<div class="ef-notice warn"><p>The last price is {last_px.date()}, more than 7 days behind.</p></div>'),
            unsafe_allow_html=True,
        )
    if params.allow_short_a:
        st.markdown(
            tx('<div class="ef-notice warn"><p>当前规格允许做空 A 股。多数标的未必能融到券，这个假设偏乐观。</p></div>', '<div class="ef-notice warn"><p>This spec shorts A-shares. Many names cannot be borrowed, so the assumption is optimistic.</p></div>'),
            unsafe_allow_html=True,
        )

    # ---- 结论卡 ----
    sharpe_text = f"{stats['sharpe']:.2f}" if np.isfinite(stats["sharpe"]) else "—"
    coint_chip = '<span class="ef-chip gain">拒绝无协整</span>' if eg["reject_5"] else '<span class="ef-chip warn">未能拒绝</span>'
    if engine == "v14":
        ret = stats["total_return"]
        ret_cls = "gain" if np.isfinite(ret) and ret >= 0 else "loss"
        st.markdown(
            tx(
                f'''<div class="ef-verdict">
<div class="cell"><h4>协整 ⓘ</h4><div class="big">EG p {pval(eg["pvalue"])}</div><div class="sub">{"5% 水平上拒绝无协整" if eg["reject_5"] else "目前未能拒绝无协整"}</div></div>
<div class="cell"><h4>信号</h4><div class="big">z {num(frame["z"].iloc[-1], 2)}</div><div class="sub">入场 ±{params.entry_z:g} · 退出 ±{params.exit_z:g}</div></div>
<div class="cell"><h4>回测</h4><div class="big" style="color:var(--ef-{'gain' if ret_cls=='gain' else 'loss'})">{pct(ret)}</div><div class="sub">夏普 {sharpe_text} · 回撤 {pct(stats["max_drawdown"])}</div></div>
</div>''',
                f'''<div class="ef-verdict">
<div class="cell"><h4>COINTEGRATION</h4><div class="big">EG p {pval(eg["pvalue"])}</div><div class="sub">{"rejects no-cointegration at 5%" if eg["reject_5"] else "does not reject at 5%"}</div></div>
<div class="cell"><h4>SIGNAL</h4><div class="big">z {num(frame["z"].iloc[-1], 2)}</div><div class="sub">entry ±{params.entry_z:g} · exit ±{params.exit_z:g}</div></div>
<div class="cell"><h4>BACKTEST</h4><div class="big">{pct(ret)}</div><div class="sub">Sharpe {sharpe_text} · drawdown {pct(stats["max_drawdown"])}</div></div>
</div>''',
            ),
            unsafe_allow_html=True,
        )
    else:
        st.caption(tx("V1.5 的样本外指标在「回测与归因」分区计算。这里的 z 与诊断口径不变。", "V1.5 out-of-sample metrics are computed in the Backtest section. z and diagnostics are unchanged here."))

    # ---- 解释条（现阶段复用解释卡三栏） ----
    if engine == "v14":
        render_v14_live_card(SimpleNamespace(result=result, params=params))

    # ---- 主图 ----
    ctrl = st.columns([2.4, 1.4, 1.6])
    chart_span = ctrl[0].segmented_control(
        tx("图表区间", "Chart span"),
        ["1y", "3y", "all"],
        format_func=lambda key: {"1y": tx("近 1 年", "1 year"), "3y": tx("近 3 年", "3 years"), "all": tx("全期间", "Full sample")}[key],
        default="all",
        key="chart_span_code",
    ) or "all"
    show_marks = ctrl[1].checkbox(tx("开平仓标记", "Trade marks"), value=True)
    show_alt = ctrl[2].checkbox(tx("对照另一条 β", "Compare beta"), value=False, key="show_alt_beta")
    if show_alt:
        alt = cached_alt_beta(picked.a_code, picked.h_code, warm, str(end_day), str(pd.Timestamp(start_day).date()), signal_token(params))
        for column in alt.columns:
            if column not in frame.columns:
                frame[column] = alt.reindex(frame.index)[column]
        result["frame"] = frame
    focus = st.session_state.get("chart_focus")
    if isinstance(focus, dict) and focus.get("start") and focus.get("end"):
        st.markdown(tx(f'<span class="ef-chip">已锁定 {focus["start"]} — {focus["end"]}</span>', f'<span class="ef-chip">locked {focus["start"]} — {focus["end"]}</span>'), unsafe_allow_html=True)
        if st.button(tx("清除", "Clear"), key="clear_chart_focus"):
            st.session_state["chart_focus"] = None
            st.rerun()

    page = SimpleNamespace(
        picked=picked,
        universe=universe,
        params=params,
        result=result,
        frame=frame,
        view=span_frame(frame, chart_span),
        common=common,
        sources=sources,
        north=north,
        north_error=north_error,
        include_costs=include_costs,
        chart_span=chart_span,
        show_marks=show_marks,
        cache_stamp="",
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
        engine=engine,
    )
    show_chart(
        core_signal_figure(page.view, page.params.z_window, page.params.entry_z, page.params.exit_z, page.params.stop_z, show_marks=page.show_marks, height=440),
        "chart_core_home",
    )

    # ---- 分区切换 ----
    sections = [
        tx("信号", "Signal"),
        tx("诊断", "Diagnostics"),
        tx("稳定性", "Stability"),
        tx("回测与归因", "Backtest"),
        tx("市场对照", "Market"),
        tx("同组对照", "Peers"),
        tx("入门", "Basics"),
    ]
    pick = st.segmented_control(tx("分区", "Section"), sections, default=sections[0], key="pair_section", label_visibility="collapsed") or sections[0]
    st.divider()
    idx = sections.index(pick)
    if idx == 0:
        render_signal(page)
    elif idx == 1:
        render_diagnose(page)
    elif idx == 2:
        render_stability(page)
    elif idx == 3:
        render_backtest(page)
    elif idx == 4:
        render_market(page)
    elif idx == 5:
        render_scan(page)
    else:
        render_summary(page)

    # ---- 维护（本地模式） ----
    if network_allowed():
        with st.expander(tx("维护", "Maintenance")):
            if st.button(tx("重新下载这对行情", "Redownload this pair"), key="reload_pair"):
                try:
                    load_pair(picked.a_code, picked.h_code, "2010-01-01", str(end_day), refresh=True)
                    cached_pair.clear()
                    cached_signals.clear()
                    cached_backtest.clear()
                    cached_alt_beta.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(tx(f"重新下载失败：{exc}", f"Download failed: {exc}"))

    sync_query(a_code, start_day, end_day)
