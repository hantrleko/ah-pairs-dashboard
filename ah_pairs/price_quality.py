"""收盘价序列清洗：上市日前数据、异常跳变。

写入缓存或发布副本前调用。已发布目录里的文件也应通过同一规则，避免脏点进入协整和滚动 β。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# 写入前自动删除的单日涨跌阈值。61% 这类日内暴涨先保留，由人工判断是否真实。
MAX_ABS_DAILY_MOVE = 1.0
# 测试与扫描用的更严阈值。
SCAN_MOVE_THRESHOLD = 0.50


def parse_list_date(value) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat"}:
        return None
    ts = pd.Timestamp(text)
    if pd.isna(ts):
        return None
    return ts.date()


def _drop_low_prefix(series: pd.Series) -> tuple[pd.Series, list[str]]:
    """Yahoo 有时在正式行情前混入极低价格的孤立点（例如另一只股票的历史）。"""
    notes: list[str] = []
    out = series.dropna().sort_index()
    if len(out) < 3:
        return out, notes
    median = float(out.median())
    if not np.isfinite(median) or median <= 0:
        return out, notes
    floor = median * 0.25
    low = out < floor
    if not bool(low.any()):
        return out, notes
    # 只砍头部连续低价，不碰中间异常点（中间留给跳变规则处理）。
    cut = 0
    for value in out.to_numpy():
        if value >= floor:
            break
        cut += 1
    if cut > 0:
        notes.append(f"去掉头部 {cut} 根远低于中位价的数据")
        out = out.iloc[cut:]
    return out, notes


def sanitize_close_series(
    series: pd.Series,
    *,
    list_date: date | None = None,
    max_abs_move: float = MAX_ABS_DAILY_MOVE,
) -> tuple[pd.Series, list[str]]:
    """去掉上市日前的点、头部低价噪声和单日异常跳变。返回 (清洗后序列, 说明)。"""
    notes: list[str] = []
    out = pd.Series(series, dtype=float).dropna().sort_index()
    if out.empty:
        return out, notes
    if list_date is not None:
        cut = pd.Timestamp(list_date).normalize()
        before = int((out.index < cut).sum())
        out = out.loc[out.index >= cut]
        if before:
            notes.append(f"上市日 {cut.date()} 前去掉 {before} 根")
    if out.empty:
        return out, notes
    out, prefix_notes = _drop_low_prefix(out)
    notes.extend(prefix_notes)
    if len(out) < 2:
        return out[~out.index.duplicated(keep="last")].sort_index(), notes
    for _ in range(8):
        move = out.pct_change().abs()
        bad = move > max_abs_move
        if not bool(bad.any()):
            break
        drop_count = int(bad.sum())
        notes.append(f"去掉 {drop_count} 根单日涨跌超过 {max_abs_move:.0%} 的数据")
        out = out.loc[~bad]
        if len(out) < 2:
            break
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out, notes


def scan_large_moves(series: pd.Series, max_abs_move: float = MAX_ABS_DAILY_MOVE) -> list[tuple[pd.Timestamp, float]]:
    """返回 (日期, 单日涨跌幅) 列表，供测试和人工核对。"""
    out = series.dropna().sort_index()
    if len(out) < 2:
        return []
    move = out.pct_change()
    hits = move[move.abs() > max_abs_move].dropna()
    return [(pd.Timestamp(ix), float(val)) for ix, val in hits.items()]