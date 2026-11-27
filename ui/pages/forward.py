"""前瞻记录页。"""

from __future__ import annotations

import streamlit as st

from ah_pairs.i18n import tx
from ui.tabs.forward import render_forward


def render() -> None:
    st.markdown(
        tx(
            '<p class="ef-kicker">FORWARD RECORD</p><h2 class="ef-title">前瞻记录</h2><p class="ef-lede">封存时间线：系统当时看到什么、决定什么、成交什么。只追加，不改写。</p>',
            '<p class="ef-kicker">FORWARD RECORD</p><h2 class="ef-title">Forward record</h2><p class="ef-lede">The sealed timeline: what the system saw, decided and filled. Append-only.</p>',
        ),
        unsafe_allow_html=True,
    )
    render_forward(None)
