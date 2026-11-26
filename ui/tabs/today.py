"""今日工作台。首屏回答“今天有什么变化”，不是“谁的 z 最大”。

上半部分是前瞻账本的逐日变化（正式启账后有数据）；下半部分是
定时任务写下的当前快照。两部分都是只读的。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ah_pairs.i18n import tx
from ah_pairs.universe import load_universe
from ah_pairs.workbench import build_workbench
from ui.explain import change_label, reason_label
from ui.format import follow_pair_selection, num, show_table
from ui.tabs.panorama import render_panorama

_ORDER = {
    "new_trigger": 0,
    "exit_signal": 1,
    "filled_entry": 2,
    "filled_exit": 3,
    "filled_flip": 4,
    "blocked": 5,
    "one_sided": 6,
    "backfilled": 7,
    "unchanged": 8,
}


def _coverage_caption(table: pd.DataFrame, missing: list[str]) -> None:
    if table.empty:
        return
    total = len(table)
    changed = int((table["change"] != "unchanged").sum())
    asofs = sorted({str(x) for x in table["data_asof"].dropna()})
    latest_asof = asofs[-1] if asofs else "—"
    observed = sorted({str(x) for x in table["observed_at"].dropna()})
    latest_obs = observed[-1][:16] if observed else "—"
    pairs = table["a_code"].nunique()
    st.caption(
        tx(
            f"账本覆盖 {pairs} 只股票 × {total // max(pairs, 1)} 个预设，共 {total} 条；{changed} 条有变化或受限。"
            f"行情截止 {latest_asof}，最近实际观察 {latest_obs}。"
            + (f"{len(missing)} 条账本缺失。" if missing else ""),
            f"{pairs} pairs x {total // max(pairs, 1)} presets, {total} ledgers; {changed} changed or blocked. "
            f"Data as of {latest_asof}, last observed {latest_obs}."
            + (f" {len(missing)} ledgers missing." if missing else ""),
        )
    )


def render_today(page) -> None:
    st.caption(
        tx(
            "这张表比较的是账本里最近两个决策日，不是两次刷新时间。信号变化和模拟成交分开列：先出信号，按预设规则在下一次执行时点成交。连续越阈只在首次触发日标新。",
            "Each row compares the last two decision days in the ledger, not two refresh runs. Signal events and simulated fills are separate columns: the signal comes first, the fill follows the preset rule. A repeated threshold crossing is not reported as new.",
        )
    )
    try:
        universe = load_universe()
    except Exception:
        universe = pd.DataFrame()
    table, missing = build_workbench(universe)
    if table.empty:
        st.info(
            tx(
                "前瞻账本尚未启账。正式启用后，每个工作日收盘后这里会列出各预设的变化；本地演练记录不在这里展示。下面仍是定时任务写下的当前快照。",
                "The forward ledger is not live yet. Once enabled, each weekday close adds the per-preset changes here; local rehearsal records are not shown. The snapshot written by the scheduled job is still below.",
            )
        )
    else:
        _coverage_caption(table, missing)
        col1, col2, col3 = st.columns(3)
        engines = sorted(table["engine"].unique())
        presets = sorted(table["preset_id"].unique())
        changes = sorted(table["change"].unique(), key=lambda c: _ORDER.get(c, 9))
        with col1:
            pick_engine = st.multiselect(tx("引擎", "Engine"), engines, default=engines, key="today_engine")
        with col2:
            pick_preset = st.multiselect(tx("预设", "Preset"), presets, default=presets, key="today_preset")
        with col3:
            only_changed = st.checkbox(tx("只看有变化或受限", "Only changed or blocked"), value=True, key="today_only_changed")
        view = table[table["engine"].isin(pick_engine) & table["preset_id"].isin(pick_preset)]
        if only_changed:
            view = view[view["change"] != "unchanged"]
        if view.empty:
            st.caption(tx("筛选后没有行。取消“只看有变化”可以看到全部账本。", "No rows after filtering. Uncheck the filter to see every ledger."))
        else:
            view = view.copy()
            view["_order"] = view["change"].map(lambda c: _ORDER.get(c, 9))
            view = view.sort_values(["_order", "engine", "preset_id", "a_code"])
            pos_labels = {0: tx("空仓", "Flat"), 1: tx("买A卖H", "Long A/H-"), -1: tx("卖A买H", "Short A/H+")}
            shown = pd.DataFrame(
                {
                    tx("名称", "Name"): view["name"].astype(str),
                    tx("引擎/预设", "Engine/preset"): view["engine"] + "/" + view["preset_id"].str.replace("v14-|v15-", "", regex=True),
                    tx("截止", "As of"): view["date"].map(lambda v: str(v) if v else "—"),
                    "z": view["z"].map(lambda v: num(v, 2)),
                    tx("持仓", "Position"): view["position"].map(lambda v: pos_labels.get(int(v or 0), pos_labels[0])),
                    tx("变化", "Change"): view["change"].map(change_label),
                    tx("原因", "Reason"): view["reason"].map(lambda r: reason_label(r or "")),
                    tx("段内净值", "Seg. equity"): view["equity"].map(lambda v: num(v, 4)),
                    tx("观察时间", "Observed"): view["observed_at"].map(lambda v: str(v)[:16] if v else "—"),
                }
            )
            event = show_table(shown, key="today_table", selectable=True)
            follow_pair_selection(event, shown)
    st.divider()
    st.markdown(tx("**当前快照（默认规格预计算）**", "**Current snapshot (default spec, precomputed)**"))
    render_panorama(page)
