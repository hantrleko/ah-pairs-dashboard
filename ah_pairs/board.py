"""全池看板的状态分类。纯函数，不碰网络。"""

from __future__ import annotations

from datetime import date

import pandas as pd

BLOCKED_REASONS = {"short_a_blocked", "beta_blocked"}

STATES = ("unavailable", "pending", "holding", "blocked", "near_entry", "calm")

NEAR_ENTRY_MARGIN = 0.5


def board_state(
    asof,
    position: int,
    pending: bool,
    reason: str,
    z: float,
    entry_z: float,
    reference_day: date,
) -> str:
    """按优先级分类：无法判断 > 待成交 > 持仓中 > 受限 > 接近入场 > 平静。

    - 无法判断：这只股票自己的截止日落后于快照整体的样本截止日
      （个股层面的行情缺失）。快照整体落后于最近收盘日时，页面顶部
      另有提示，不逐行标。
    - 待成交：引擎已给信号但还没成交。
    - 受限：引擎理由属于受限集合（需做空 A、β 越界）。只在持仓时统计；
      空仓被拦本身就是"接近入场"的信号，不算受限。
    - 接近入场：空仓且 |z| 距入场阈值不到 NEAR_ENTRY_MARGIN。这是展示分类，不是信号。
    """
    try:
        cutoff = pd.Timestamp(asof).date()
    except (TypeError, ValueError):
        return "unavailable"
    if cutoff < reference_day:
        return "unavailable"
    pos = int(position or 0)
    if bool(pending):
        return "pending"
    if pos != 0:
        if (reason or "") in BLOCKED_REASONS:
            return "blocked"
        return "holding"
    try:
        zval = float(z)
    except (TypeError, ValueError):
        return "unavailable"
    if abs(zval) >= float(entry_z) - NEAR_ENTRY_MARGIN:
        return "near_entry"
    return "calm"


def classify(table: pd.DataFrame, reference_day: date, entry_z: float) -> pd.Series:
    """逐行分类，返回与 table 对齐的 Series。"""
    if table is None or table.empty:
        return pd.Series(dtype=str)
    reasons = table["reason"] if "reason" in table.columns else pd.Series([""] * len(table), index=table.index)
    return pd.Series(
        [
            board_state(
                row.asof,
                int(row.position or 0),
                bool(row.pending),
                str(reasons.iloc[i] or ""),
                row.z,
                entry_z,
                reference_day,
            )
            for i, row in enumerate(table.itertuples(index=False))
        ],
        index=table.index,
        dtype=str,
    )
