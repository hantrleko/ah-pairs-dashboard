"""页面样式。"""

from __future__ import annotations

import streamlit as st

CSS = """
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
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { font-size: 0.86rem; }
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stDateInput div[data-baseweb="input"] {
  background: #ffffff;
  border-color: #d5dee8;
}
.stTabs [data-baseweb="tab-list"] { gap: 0.15rem; border-bottom: 1px solid #d5dee8; }
.stTabs [data-baseweb="tab"] {
  background: transparent;
  color: #5a6a7a;
  border-radius: 0;
  padding: 0.45rem 0.7rem;
  font-size: 0.92rem;
}
.stTabs [aria-selected="true"] { color: #1f4e79; border-bottom: 2px solid #1f4e79; }
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
.kpi-row { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 0.55rem; margin: 0 0 0.65rem; }
.kpi-row.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.kpi-row.five { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.kpi {
  background: #ffffff;
  border: 1px solid #d5dee8;
  border-radius: 8px;
  padding: 0.62rem 0.75rem 0.55rem;
  box-shadow: 0 1px 2px rgba(22, 50, 79, 0.05), 0 6px 16px rgba(22, 50, 79, 0.04);
}
.kpi span { display: block; font-size: 0.72rem; letter-spacing: 0.04em; color: #5a6a7a; }
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
.lesson { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.7rem; margin: 0.35rem 0 0.9rem; }
.lesson article {
  background: #ffffff;
  border: 1px solid #d5dee8;
  border-radius: 8px;
  padding: 0.85rem 0.95rem 0.8rem;
  box-shadow: 0 1px 2px rgba(22, 50, 79, 0.05);
}
.lesson strong { display: block; color: #1f4e79; margin-bottom: 0.35rem; font-size: 0.95rem; }
.lesson p { margin: 0; color: #3d4c5c; font-size: 0.92rem; line-height: 1.6; }
.toolbar, .health {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 0.75rem;
  align-items: baseline;
  margin: 0.15rem 0 0.55rem;
  color: #3d4c5c;
  font-size: 0.86rem;
}
.health.warn {
  display: block;
  background: #fff6f2;
  border: 1px solid #e7c3b2;
  border-left: 3px solid #b4532a;
  border-radius: 8px;
  padding: 0.55rem 0.75rem;
  color: #7a3418;
}
.tag { display: inline-block; padding: 0.05rem 0.4rem; border-radius: 999px; background: #e7f2ea; color: #1d6b45; font-size: 0.75rem; }
.tag.bad { background: #f8e8e4; color: #9b3a3a; }
@media (max-width: 900px) {
  .kpi-row, .kpi-row.four, .kpi-row.five { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .lesson { grid-template-columns: 1fr; }
  .mast { flex-direction: column; align-items: flex-start; }
  .brand h1 { font-size: 1.25rem; }
  .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; overflow-x: visible; }
  .stTabs [data-baseweb="tab"] { padding: 0.4rem 0.5rem; font-size: 0.82rem; }
  [data-testid="stRadio"] [role="radiogroup"] { flex-wrap: wrap; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
