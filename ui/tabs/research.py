"""单对研究。首屏是解释卡，下方按主题分七个区。

解释卡优先引用冻结预设的封存账本；账本尚未启账时，
v14 退化为按当前侧栏参数的实时探索口径，并如实标注。
"""

from __future__ import annotations

import streamlit as st

from ah_pairs.forward import FORWARD_DIR, read_rows, rows_path
from ah_pairs.i18n import tx
from ah_pairs.presets import PRESETS
from ui.explain import render_ledger_card, render_v14_live_card
from ui.tabs.backtest import render_backtest
from ui.tabs.diagnose import render_diagnose
from ui.tabs.market import render_market
from ui.tabs.scan import render_scan
from ui.tabs.signal import render_signal
from ui.tabs.stability import render_stability
from ui.tabs.summary import render_summary


def _ledger_card_section(page) -> None:
    a_code = str(page.picked.a_code).zfill(6)
    available = []
    for preset_id in PRESETS:
        rows = read_rows(rows_path(FORWARD_DIR, preset_id, a_code))
        if rows:
            available.append((preset_id, rows))
    if available:
        st.caption(
            tx(
                "这张卡来自冻结预设的封存账本：理由由引擎在决策时产生，不是页面反推。侧栏探索参数不改变这些记录。",
                "This card comes from the sealed ledger of a frozen preset: reasons are produced by the engine at decision time, not inferred by the page. Sidebar exploration does not change these records.",
            )
        )
        ids = [preset_id for preset_id, _rows in available]
        pick = st.selectbox(tx("预设", "Preset"), ids, key="research_preset")
        rows = dict(available)[pick]
        render_ledger_card(rows[-1], title=f"{page.picked['name']} · {pick}")
        return
    if page.engine == "v14":
        st.caption(
            tx(
                "前瞻账本尚未启账。下面这张卡按当前侧栏参数实时重算，是探索口径，不是冻结预设的封存记录。",
                "The forward ledger is not live yet. The card below is recomputed from the current sidebar spec — exploration, not a sealed record of a frozen preset.",
            )
        )
        render_v14_live_card(page)
    else:
        st.caption(
            tx(
                "v15 的规则按折锁定，解释以样本外折结果和前瞻账本为准；账本尚未启账，当前没有可展示的逐日决策。",
                "v15 rules are locked per fold; its explanation lives in the out-of-sample folds and the forward ledger. The ledger is not live yet, so there is no daily decision to show.",
            )
        )


def render_research(page) -> None:
    st.markdown(tx("**信号解释卡**", "**Signal explanation**"))
    _ledger_card_section(page)
    st.divider()
    subtabs = st.tabs(
        [
            tx("摘要", "Summary"),
            tx("信号", "Signal"),
            tx("协整诊断", "Diagnostics"),
            tx("稳定性", "Stability"),
            tx("回测与归因", "Backtest"),
            tx("市场对照", "Market"),
            tx("一组对照", "Pairs"),
        ]
    )
    with subtabs[0]:
        render_summary(page)
    with subtabs[1]:
        render_signal(page)
    with subtabs[2]:
        render_diagnose(page)
    with subtabs[3]:
        render_stability(page)
    with subtabs[4]:
        render_backtest(page)
    with subtabs[5]:
        render_market(page)
    with subtabs[6]:
        render_scan(page)
