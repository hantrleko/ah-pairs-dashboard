"""页头的 EF Lab 入口。只放已经公开的地址。"""

from __future__ import annotations

import streamlit as st

from ah_pairs.eflab_calendar import LAB_ENTRY, SESSION_READY, SITES, last_completed_session
from ah_pairs.i18n import tx


def render_lab_strip() -> None:
    session = last_completed_session()
    ready = SESSION_READY.strftime("%H:%M")
    links = " · ".join(f"[{site['name']}]({site['url']})" for site in SITES)
    st.caption(
        tx(
            f"入口 [EF Lab]({LAB_ENTRY}) · {links}。理财站点还没有公开地址。行情日期看到最近已收盘交易日 {session}，周末和 {ready} 以前不把当天算进去。",
            f"Entry [EF Lab]({LAB_ENTRY}) · {links}. The planning site has no public address. Market dates use the last completed session {session}. Weekends and times before {ready} do not count that day.",
        )
    )
