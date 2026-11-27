"""今日 · 全池信号看板。默认首页：只读已发布快照，不做单对计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ah_pairs.board import classify
from ah_pairs.eflab_calendar import last_completed_session
from ah_pairs.i18n import tx
from ah_pairs.panorama import build_panorama, is_default_spec, write_panorama
from ah_pairs.paths import PUBLISHED_DIR
from ah_pairs.desk import default_desk_params
from ah_pairs.universe import liquid_subset, load_universe
from ah_pairs.workbench import build_workbench
from ui.explain import change_label, reason_label
from ui.format import follow_pair_selection, num, pct, show_table
from ui.state import load_universe_filtered, network_allowed

STATE_META = {
    "pending": ("待成交", "Pending", "warn"),
    "holding": ("持仓中", "Holding", "solid"),
    "blocked": ("受限", "Blocked", "warn"),
    "near_entry": ("接近入场", "Near entry", "gain"),
    "calm": ("平静", "Calm", ""),
    "unavailable": ("无法判断", "No reading", "loss"),
}
FILTER_ORDER = ["pending", "holding", "blocked", "near_entry", "calm", "unavailable"]
ENTRY_Z = 2.0


def _load_snapshot() -> tuple[pd.DataFrame | None, dict]:
    table_path = PUBLISHED_DIR / "panorama.parquet"
    meta: dict = {}
    meta_path = PUBLISHED_DIR / "panorama.json"
    if meta_path.exists():
        import json

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    if not table_path.exists():
        return None, meta
    try:
        return pd.read_parquet(table_path), meta
    except Exception:
        return None, meta


def _state_chip(state: str) -> str:
    zh, en, cls = STATE_META.get(state, (state, state, ""))
    label = tx(zh, en)
    return label


def _board(table: pd.DataFrame, meta: dict) -> None:
    session_day = last_completed_session()
    sample_end = pd.Timestamp(meta.get("sample_end")).date() if meta.get("sample_end") else None
    if sample_end is None:
        sample_end = pd.Timestamp(table["asof"].max()).date()
    if sample_end < session_day:
        st.markdown(
            tx(
                f'<div class="ef-notice warn"><p>快照样本截止 {sample_end}，落后于最近已收盘交易日 {session_day}。每日刷新通常在晚间更新。</p></div>',
                f'<div class="ef-notice warn"><p>The snapshot ends {sample_end}, behind the last completed session {session_day}. The daily refresh usually updates in the evening.</p></div>',
            ),
            unsafe_allow_html=True,
        )

    states = classify(table, sample_end, ENTRY_Z)
    work = table.copy()
    work["_state"] = states

    counts = {key: int((states == key).sum()) for key in FILTER_ORDER}
    total = len(work)
    options = ["all"] + [key for key in FILTER_ORDER if counts.get(key)]
    labels = {"all": tx(f"全部 {total}", f"All {total}")}
    for key in FILTER_ORDER:
        zh, en, _cls = STATE_META[key]
        labels[key] = tx(f"{zh} {counts.get(key, 0)}", f"{en} {counts.get(key, 0)}")
    pick = st.pills(
        tx("状态", "State"),
        options,
        default=["all"],
        selection_mode="single",
        format_func=lambda key: labels.get(key, key),
        label_visibility="collapsed",
        key="board_filter",
    ) or "all"
    view = work if pick == "all" else work.loc[work["_state"] == pick]

    # 距入场进度：|z| / 入场阈值，持仓或待成交时满格
    def _progress(row) -> float:
        if int(row.get("position") or 0) != 0 or bool(row.get("pending")):
            return 1.0
        try:
            return min(1.0, abs(float(row["z"])) / ENTRY_Z)
        except (TypeError, ValueError):
            return 0.0

    view = view.copy()
    view["_progress"] = view.apply(_progress, axis=1)
    has_tail = "z_tail" in view.columns and view["z_tail"].map(lambda v: isinstance(v, (list, tuple, np.ndarray)) and len(v) > 1).any()

    shown = pd.DataFrame(
        {
            tx("名称", "Name"): view["name"].astype(str),
            tx("状态", "State"): view["_state"].map(_state_chip),
            "z": view["z"].astype(float),
            tx("距入场", "To entry"): view["_progress"].astype(float),
            tx("溢价", "Premium"): (view["premium"].astype(float) * 100),
            tx("截止", "As of"): view["asof"].astype(str),
            tx("持仓", "Position"): view["position"].map(lambda v: {0: tx("空仓", "flat"), 1: tx("买A卖H", "long A/H-"), -1: tx("卖A买H", "short A/H+")}.get(int(v or 0), "—")),
            "A": view["a_code"].astype(str),
            "H": view["h_code"].astype(str),
        }
    )
    if has_tail:
        shown[tx("近 60 日 z", "z 60d")] = view["z_tail"].map(lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [])

    column_config = {
        "z": st.column_config.NumberColumn("z", format="%.2f", width="small"),
        tx("距入场", "To entry"): st.column_config.ProgressColumn(tx("距入场", "To entry"), min_value=0.0, max_value=1.0, width="medium"),
        tx("溢价", "Premium"): st.column_config.NumberColumn(tx("溢价 %", "Premium %"), format="%.1f", width="small"),
        tx("名称", "Name"): st.column_config.TextColumn(tx("名称", "Name"), pinned=True),
    }
    if has_tail:
        column_config[tx("近 60 日 z", "z 60d")] = st.column_config.LineChartColumn(tx("近 60 日 z", "z 60d"), width="medium")

    event = st.dataframe(
        shown,
        width="stretch",
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row",
        key="board_table",
    )
    rows = list(getattr(getattr(event, "selection", None), "rows", []) or [])
    if rows:
        code = str(view.iloc[int(rows[0])]["a_code"]).zfill(6)
        st.session_state["pair_select"] = code
        st.switch_page("app_pages/pair.py")

    st.caption(
        tx(
            "「距入场」是当前 |z| 占入场阈值的比例，持仓或待成交时显示满格。这是展示分类，不是交易信号。点一行进入单对页。",
            "'To entry' is |z| as a fraction of the entry threshold, full when holding or pending. It is a display grouping, not a trading signal. Click a row to open the pair.",
        )
    )


def _ledger_section() -> None:
    try:
        universe = load_universe()
    except Exception:
        universe = pd.DataFrame()
    table, missing = build_workbench(universe)
    if table.empty:
        st.markdown(
            tx(
                '<div class="ef-empty"><h3>前瞻账本尚未启账</h3><p>启账后，这里列出各预设相对上一决策日的变化：新触发、退出、成交与受限。</p></div>',
                '<div class="ef-empty"><h3>The forward ledger is not live yet</h3><p>Once live, this lists per-preset changes since the previous decision day.</p></div>',
            ),
            unsafe_allow_html=True,
        )
        return
    order = {"new_trigger": 0, "exit_signal": 1, "filled_entry": 2, "filled_exit": 3, "filled_flip": 4, "blocked": 5, "one_sided": 6, "backfilled": 7, "unchanged": 8}
    pos_labels = {0: tx("空仓", "Flat"), 1: tx("买A卖H", "Long A/H-"), -1: tx("卖A买H", "Short A/H+")}
    view = table.copy()
    view["_order"] = view["change"].map(lambda c: order.get(c, 9))
    view = view.sort_values(["_order", "engine", "preset_id", "a_code"])
    shown = pd.DataFrame(
        {
            tx("名称", "Name"): view["name"].astype(str),
            tx("引擎/预设", "Engine/preset"): view["engine"] + "/" + view["preset_id"].str.replace("v14-|v15-", "", regex=True),
            tx("截止", "As of"): view["date"].map(lambda v: str(v) if v else "—"),
            "z": view["z"].map(lambda v: num(v, 2)),
            tx("持仓", "Position"): view["position"].map(lambda v: pos_labels.get(int(v or 0), pos_labels[0])),
            tx("变化", "Change"): view["change"].map(change_label),
            tx("原因", "Reason"): view["reason"].map(lambda r: reason_label(r or "")),
            tx("观察时间", "Observed"): view["observed_at"].map(lambda v: str(v)[:16] if v else "—"),
        }
    )
    event = show_table(shown, key="today_ledger_table", selectable=True)
    follow_pair_selection(event, shown.assign(A=view["a_code"].astype(str)))


def _maintenance() -> None:
    if not network_allowed():
        return
    with st.expander(tx("维护", "Maintenance")):
        if st.button(tx("用本地缓存重算快照", "Recompute the snapshot from the local cache"), key="panorama_compute"):
            universe = load_universe()
            params = default_desk_params()
            with st.spinner(tx("只读本地行情…", "Reading local prices…")):
                fresh, fresh_meta = build_panorama(liquid_subset(universe), pd.Timestamp("2018-01-01").date(), pd.Timestamp.today().date(), params)
            if is_default_spec(params, pd.Timestamp("2018-01-01").date()):
                write_panorama(fresh, fresh_meta)
            st.rerun()


def render() -> None:
    st.markdown(
        tx(
            f'<p class="ef-kicker">EF LAB · A+H</p><h2 class="ef-title">今日 · 全池信号</h2><p class="ef-lede">最近已收盘交易日 {last_completed_session()}。默认规格 v14-default：滚动 OLS，估计 120 日，z 60 日，入场 ±2.0，次日收盘成交，含成本，不做空 A。</p>',
            f'<p class="ef-kicker">EF LAB · A+H</p><h2 class="ef-title">Today · the whole pool</h2><p class="ef-lede">Last completed session {last_completed_session()}. Frozen preset v14-default: rolling OLS, 120-day estimation, 60-day z, entry ±2.0, next-close fills, costs on, no shorting A.</p>',
        ),
        unsafe_allow_html=True,
    )
    table, meta = _load_snapshot()
    if table is None or table.empty:
        st.markdown(
            tx(
                '<div class="ef-empty"><h3>还没有全池快照</h3><p>每日刷新后自动生成。</p></div>',
                '<div class="ef-empty"><h3>No snapshot yet</h3><p>It is generated by the daily refresh.</p></div>',
            ),
            unsafe_allow_html=True,
        )
    else:
        _board(table, meta)
    _maintenance()
    st.divider()
    st.markdown(
        tx('<p class="ef-kicker">FORWARD LEDGER</p><h3 class="ef-title" style="font-size:1.2rem">前瞻账本 · 逐日变化</h3>', '<p class="ef-kicker">FORWARD LEDGER</p><h3 class="ef-title" style="font-size:1.2rem">Forward ledger · daily changes</h3>'),
        unsafe_allow_html=True,
    )
    _ledger_section()
