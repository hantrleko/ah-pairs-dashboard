"""应用外壳：顶部导航四页 + 上下文条 + 页脚。

页面在 ui/pages/。单对计算只在单对页发生。
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="A+H Pairs · EF Lab", layout="wide", initial_sidebar_state="collapsed")

from ah_pairs.i18n import tx
from ah_pairs.panorama import load_panorama
from ui.state import init_lang, lang_selector
from ui.style import inject_css

inject_css()
init_lang()

logo_path = Path("static/logo.png")
if logo_path.exists():
    st.logo(str(logo_path))


def _context_strip() -> None:
    asof = "—"
    try:
        import json

        from ah_pairs.panorama import META_NAME
        from ah_pairs.paths import PUBLISHED_DIR as CACHE_DIR

        meta_file = CACHE_DIR / META_NAME
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            asof = str(meta.get("sample_end") or "—")[:10]
    except Exception:
        pass
    lang = st.session_state.get("lang", "zh")
    left, right = st.columns([3.4, 0.9])
    with left:
        if asof != "—":
            st.markdown(
                tx(
                    f'<div class="ef-context"><span class="ef-dot">●</span><span>行情截止 {asof}</span></div>',
                    f'<div class="ef-context"><span class="ef-dot">●</span><span>data through {asof}</span></div>',
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                tx(
                    '<div class="ef-context"><span class="ef-dot stale">●</span><span>全景快照缺失</span></div>',
                    '<div class="ef-context"><span class="ef-dot stale">●</span><span>panorama snapshot missing</span></div>',
                ),
                unsafe_allow_html=True,
            )
    with right:
        lang_selector()


_context_strip()

pages = [
    st.Page("app_pages/today.py", title=tx("今日", "Today"), url_path="today", icon=":material/home:", default=True),
    st.Page("app_pages/pair.py", title=tx("单对", "Pair"), url_path="pair", icon=":material/candlestick_chart:"),
    st.Page("app_pages/forward.py", title=tx("前瞻", "Forward"), url_path="forward", icon=":material/schedule:"),
    st.Page("app_pages/method.py", title=tx("方法", "Method"), url_path="method", icon=":material/menu_book:"),
]
st.navigation(pages, position="top").run()

build_sha = ""
sha_file = Path("BUILD_SHA")
if sha_file.exists():
    build_sha = sha_file.read_text(encoding="utf-8").strip()[:7]
elif os.environ.get("BUILD_SHA"):
    build_sha = os.environ["BUILD_SHA"][:7]
build_part = f" · build {build_sha}" if build_sha else ""
st.markdown(
    tx(
        f'<div class="ef-footer"><a href="https://eugenefinance.tech" target="_blank">EF Lab</a> · V1.6{build_part} · 仅作研究展示，不构成投资建议</div>',
        f'<div class="ef-footer"><a href="https://eugenefinance.tech" target="_blank">EF Lab</a> · V1.6{build_part} · research demo, not investment advice</div>',
    ),
    unsafe_allow_html=True,
)
