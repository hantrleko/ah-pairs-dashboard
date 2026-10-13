"""行情和北向资金的截止日期。

北向日频在 2024-08-16 之后停止披露。价格如果比所选结束日或今天落后超过一周，也算过期。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ah_pairs.paths import CACHE_DIR
from ah_pairs.universe import a_to_yahoo, h_to_yahoo

NORTHBOUND_DISCLOSURE_END = pd.Timestamp("2024-08-16")
PRICE_LAG_DAYS = 7


def price_is_stale(last_bar, requested_end, today) -> bool:
    """最后一根 K 线是否落后于「今天和所选结束日里较早的那天」超过一周。"""
    if last_bar is None or requested_end is None or today is None:
        return True
    last = pd.Timestamp(last_bar).normalize()
    horizon = min(pd.Timestamp(requested_end).normalize(), pd.Timestamp(today).normalize())
    return (horizon - last).days > PRICE_LAG_DAYS


def northbound_stops_before(price_end, northbound_end) -> bool:
    """价差还在走，北向日频已经断了。"""
    if price_end is None or northbound_end is None:
        return True
    return pd.Timestamp(price_end).normalize() > pd.Timestamp(northbound_end).normalize()


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("=", "_").replace("^", "_")
    return CACHE_DIR / f"{safe}.csv"


def cache_written_at(a_code: str, h_code: str) -> str | None:
    """本地行情缓存里最晚的写入时间。没有缓存文件时返回 None。"""
    stamps = []
    for ticker in (a_to_yahoo(a_code), h_to_yahoo(h_code), "HKDCNY=X"):
        path = _cache_path(ticker)
        if path.exists():
            stamps.append(path.stat().st_mtime)
    if not stamps:
        return None
    return datetime.fromtimestamp(max(stamps)).strftime("%Y-%m-%d %H:%M")
