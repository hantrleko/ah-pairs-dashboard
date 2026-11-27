"""页面样式。颜色 token 与 .streamlit/config.toml 保持一致；

自定义 CSS 只作用于 ef-* 组件，Streamlit 内部选择器尽量少，每条注明原因。
"""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
/* ---- 内部选择器（必要项，升级时核对） ---- */
/* 隐藏汉堡菜单、Deploy 按钮与页脚装饰；顶部导航在 stToolbar 内，不能整条隐藏 */
#MainMenu, [data-testid="stMainMenu"], footer, [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] { display: none; }
div[data-testid="stDecoration"] { display: none; }
/* 版心宽度与紧凑节奏 */
.block-container { padding-top: 3.4rem; padding-bottom: 2.2rem; max-width: 1240px; }
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }

/* ---- 设计 token ---- */
:root {
  --ef-paper: #F4EEE1;
  --ef-card: #F9F6F1;
  --ef-ink: #0C3123;
  --ef-ink-muted: #4E7466;
  --ef-accent: #0E624D;
  --ef-gold: #BB973A;
  --ef-line: #B9D5CB;
  --ef-line-soft: #D8D0BE;
  --ef-gain: #1D6B45;
  --ef-loss: #9B3A3A;
  --ef-warn: #8F4F0C;
  --ef-warn-bg: #F5E9D4;
  --ef-serif: "Libre Baskerville", "Noto Serif SC", "Songti SC", "STSong", "SimSun", serif;
  --ef-sans: "IBM Plex Sans", "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
  --ef-mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}

/* ---- 提示框统一为纸面家族样式（语义靠图标和文字） ---- */
[data-testid="stAlert"] {
  background: var(--ef-card);
  border: 1px solid var(--ef-line-soft);
  border-left: 3px solid var(--ef-gold);
  border-radius: 6px;
  color: var(--ef-ink);
}
[data-testid="stAlert"] p { color: var(--ef-ink); }

/* ---- 眉题 ---- */
.ef-kicker {
  font-family: var(--ef-mono);
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ef-gold);
  margin: 0 0 0.15rem;
}
.ef-kicker::after {
  content: "";
  display: block;
  width: 2.2rem;
  border-bottom: 1.5px solid var(--ef-gold);
  margin-top: 0.3rem;
}

/* ---- 页面标题 ---- */
.ef-title {
  font-family: var(--ef-serif);
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--ef-ink);
  margin: 0 0 0.1rem;
  letter-spacing: -0.01em;
}
.ef-lede { color: var(--ef-ink-muted); font-size: 0.92rem; margin: 0.1rem 0 0.9rem; }

/* ---- 上下文条 ---- */
.ef-context {
  display: flex;
  align-items: baseline;
  gap: 0.9rem;
  font-family: var(--ef-mono);
  font-size: 0.78rem;
  color: var(--ef-ink-muted);
  margin: 0 0 0.4rem;
  flex-wrap: wrap;
}
.ef-context .ef-dot { color: var(--ef-gain); }
.ef-context .ef-dot.stale { color: var(--ef-warn); }

/* ---- 状态签 ---- */
.ef-chip {
  display: inline-block;
  font-family: var(--ef-mono);
  font-size: 0.72rem;
  padding: 0.08rem 0.45rem;
  border-radius: 999px;
  border: 1px solid var(--ef-line-soft);
  color: var(--ef-ink-muted);
  background: transparent;
  white-space: nowrap;
}
.ef-chip.gain { color: var(--ef-gain); border-color: var(--ef-gain); }
.ef-chip.loss { color: var(--ef-loss); border-color: var(--ef-loss); }
.ef-chip.warn { color: var(--ef-warn); border-color: var(--ef-warn); background: var(--ef-warn-bg); }
.ef-chip.long { color: var(--ef-accent); border-color: var(--ef-accent); background: rgba(14, 98, 77, 0.10); }
.ef-chip.short { color: #7a611f; border-color: var(--ef-gold); background: rgba(187, 151, 58, 0.14); }
.ef-chip.solid { background: var(--ef-accent); color: var(--ef-card); border-color: var(--ef-accent); }

/* ---- 提示 ---- */
.ef-notice {
  border-left: 3px solid var(--ef-gold);
  background: var(--ef-card);
  padding: 0.55rem 0.8rem;
  color: var(--ef-ink);
  font-size: 0.9rem;
  margin: 0.3rem 0 0.6rem;
}
.ef-notice.warn { border-left-color: var(--ef-warn); background: var(--ef-warn-bg); }
.ef-notice p { margin: 0; }

/* ---- 结论卡 ---- */
.ef-verdict { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.7rem; margin: 0.2rem 0 0.9rem; }
.ef-verdict .cell {
  background: var(--ef-card);
  border: 1px solid var(--ef-line-soft);
  border-top: 2px solid var(--ef-gold);
  border-radius: 6px;
  padding: 0.7rem 0.85rem 0.65rem;
}
.ef-verdict .cell h4 {
  font-family: var(--ef-mono);
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  color: var(--ef-ink-muted);
  margin: 0 0 0.3rem;
  font-weight: 500;
}
.ef-verdict .cell .big { font-size: 1.25rem; font-variant-numeric: tabular-nums; font-weight: 600; color: var(--ef-ink); }
.ef-verdict .cell .sub { font-size: 0.82rem; color: var(--ef-ink-muted); margin-top: 0.2rem; }

/* ---- 三步解释条 ---- */
.ef-flow { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; align-items: stretch; gap: 0.5rem; margin: 0.2rem 0 0.9rem; }
.ef-flow .step {
  background: var(--ef-card);
  border: 1px solid var(--ef-line-soft);
  border-radius: 6px;
  padding: 0.6rem 0.8rem;
}
.ef-flow .step h5 {
  font-family: var(--ef-mono);
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  color: var(--ef-gold);
  margin: 0 0 0.25rem;
  font-weight: 500;
}
.ef-flow .step p { margin: 0; font-size: 0.88rem; color: var(--ef-ink); }
.ef-flow .arrow { align-self: center; color: var(--ef-gold); font-family: var(--ef-mono); }

/* ---- 规格条 ---- */
.ef-spec {
  font-family: var(--ef-mono);
  font-size: 0.78rem;
  color: var(--ef-ink-muted);
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin: 0.15rem 0 0.3rem;
}

/* ---- 单对抬头 ---- */
.ef-pairname {
  font-family: var(--ef-serif);
  font-size: 2.1rem;
  font-weight: 700;
  color: var(--ef-ink);
  margin: 0;
  line-height: 1.1;
}
.ef-pairmeta { font-family: var(--ef-mono); font-size: 0.8rem; color: var(--ef-ink-muted); margin: 0.15rem 0 0; }
.ef-premium { text-align: right; }
.ef-premium .lbl { font-family: var(--ef-mono); font-size: 0.7rem; letter-spacing: 0.12em; color: var(--ef-ink-muted); display: block; }
.ef-premium .val { font-size: 1.5rem; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--ef-ink); }

/* ---- 数字磁贴 ---- */
.ef-tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.7rem; margin: 0.2rem 0 0.8rem; }
.ef-tiles .tile {
  background: var(--ef-card);
  border: 1px solid var(--ef-line-soft);
  border-radius: 6px;
  padding: 0.55rem 0.75rem 0.5rem;
}
.ef-tiles .tile span { display: block; font-family: var(--ef-mono); font-size: 0.7rem; letter-spacing: 0.06em; color: var(--ef-ink-muted); }
.ef-tiles .tile strong { display: block; margin-top: 0.15rem; font-size: 1.2rem; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--ef-ink); }

/* ---- 空状态 ---- */
.ef-empty {
  border: 1px dashed var(--ef-line);
  border-radius: 8px;
  padding: 1.6rem 1.4rem;
  text-align: center;
  color: var(--ef-ink-muted);
  background: transparent;
}
.ef-empty h3 { font-family: var(--ef-serif); color: var(--ef-ink); margin: 0 0 0.4rem; font-size: 1.2rem; }
.ef-empty p { margin: 0.2rem 0; font-size: 0.9rem; }

/* ---- 页脚 ---- */
.ef-footer {
  margin-top: 2rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--ef-line-soft);
  color: var(--ef-ink-muted);
  font-size: 0.8rem;
  font-family: var(--ef-mono);
}
.ef-footer a { color: var(--ef-accent); text-decoration: none; }

/* ---- 课程卡 ---- */
.lesson { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.7rem; margin: 0.35rem 0 0.9rem; }
.lesson article {
  background: var(--ef-card);
  border: 1px solid var(--ef-line-soft);
  border-radius: 6px;
  padding: 0.85rem 0.95rem 0.8rem;
}
.lesson strong { display: block; color: var(--ef-accent); margin-bottom: 0.35rem; font-size: 0.95rem; }
.lesson p { margin: 0; color: var(--ef-ink); font-size: 0.9rem; line-height: 1.6; }

/* ---- 窄屏 ---- */
@media (max-width: 900px) {
  .block-container { padding-top: 3.2rem; }
  .ef-verdict { grid-template-columns: 1fr; }
  .ef-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ef-flow { grid-template-columns: 1fr; }
  .ef-flow .arrow { display: none; }
  .lesson { grid-template-columns: 1fr; }
  .ef-pairname { font-size: 1.6rem; }
  .ef-title { font-size: 1.4rem; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
