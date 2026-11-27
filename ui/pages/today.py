"""今日 · 全池信号。默认首页：只读已发布快照，不做单对计算。"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from ah_pairs.eflab_calendar import last_completed_session
from ah_pairs.i18n import tx
from ah_pairs.desk import default_desk_params
from ah_pairs.universe import liquid_subset, load_universe
from ah_pairs.workbench import build_workbench
from ui.explain import change_label, reason_label
from ui.format import follow_pair_selection, num, show_table
from ui.state import load_universe_filtered
from ui.tabs.panorama import render_panorama


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


def render() -> None:
    st.markdown(
        tx(
            f'<p class="ef-kicker">EF LAB · A+H</p><h2 class="ef-title">今日 · 全池信号</h2><p class="ef-lede">最近已收盘交易日 {last_completed_session()}。默认规格 v14-default：滚动 OLS，估计 120 日，z 60 日，入场 ±2.0，次日收盘成交，含成本，不做空 A。</p>',
            f'<p class="ef-kicker">EF LAB · A+H</p><h2 class="ef-title">Today · the whole pool</h2><p class="ef-lede">Last completed session {last_completed_session()}. Frozen preset v14-default: rolling OLS, 120-day estimation, 60-day z, entry ±2.0, next-close fills, costs on, no shorting A.</p>',
        ),
        unsafe_allow_html=True,
    )
    universe = load_universe_filtered()
    page = SimpleNamespace(
        params=default_desk_params(),
        start_day=date(2018, 1, 1),
        end_day=pd.Timestamp.today().date(),
        universe=liquid_subset(universe) if len(universe) else universe,
    )
    render_panorama(page)
    st.divider()
    st.markdown(
        tx('<p class="ef-kicker">FORWARD LEDGER</p><h3 class="ef-title" style="font-size:1.2rem">前瞻账本 · 逐日变化</h3>', '<p class="ef-kicker">FORWARD LEDGER</p><h3 class="ef-title" style="font-size:1.2rem">Forward ledger · daily changes</h3>'),
        unsafe_allow_html=True,
    )
    _ledger_section()
