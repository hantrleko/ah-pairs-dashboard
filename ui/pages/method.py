"""方法页。"""

from __future__ import annotations

import streamlit as st

from ah_pairs.i18n import tx
from ui.tabs.method import render_method


def render() -> None:
    st.markdown(
        tx(
            '<p class="ef-kicker">METHOD</p><h2 class="ef-title">方法</h2>',
            '<p class="ef-kicker">METHOD</p><h2 class="ef-title">Method</h2>',
        ),
        unsafe_allow_html=True,
    )
    col, _pad = st.columns([2.6, 1])
    with col:
        render_method()
