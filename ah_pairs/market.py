"""市场层面的对照：南向资金，以及恒生沪深港通 AH 股溢价指数（HSAHP）。

北向日频在 2024-08 之后停止。南向和 AH 溢价指数仍可能更新，但下载失败时页面只说明原因，不拿别的序列冒充。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ah_pairs.northbound import _as_dates, describe_northbound
from ah_pairs.paths import CACHE_DIR


def _store(path: Path, series: pd.Series, column: str) -> pd.Series:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = series.to_frame(column)
    frame.index.name = "date"
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp)
    tmp.replace(path)
    return series


def _load_cached(path: Path, name: str) -> pd.Series | None:
    legacy = path.with_suffix(".csv")
    source = path if path.exists() else legacy if legacy.exists() else None
    if source is None:
        return None
    if source.suffix == ".parquet":
        frame = pd.read_parquet(source)
        if "date" in frame.columns:
            frame = frame.set_index("date")
    else:
        frame = pd.read_csv(source, index_col=0, parse_dates=True)
    series = frame.iloc[:, 0].astype(float)
    series.index = _as_dates(series.index)
    series.name = name
    return series.sort_index()


def load_southbound(cache_dir: Path = CACHE_DIR, refresh: bool = False) -> pd.Series:
    """港股通合计净买入，单位亿元。不是这一只 H 股的南向持仓。"""
    path = cache_dir / "southbound.parquet"
    if not refresh:
        cached = _load_cached(path, "southbound")
        if cached is not None and not cached.empty:
            return cached
    import akshare as ak

    raw = ak.stock_hsgt_hist_em(symbol="南向资金")
    series = pd.Series(
        pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(),
        index=_as_dates(raw.iloc[:, 0]),
        name="southbound",
    )
    series = series[~series.index.duplicated(keep="last")].dropna().sort_index()
    return _store(path, series, "southbound")


def load_ah_premium_index(cache_dir: Path = CACHE_DIR, refresh: bool = False) -> pd.Series:
    """恒生沪深港通 AH 股溢价指数收盘。100 表示样本内 A/H 平均平价，高于 100 表示 A 股更贵。"""
    path = cache_dir / "hsahp.parquet"
    if not refresh:
        cached = _load_cached(path, "hsahp")
        if cached is not None and not cached.empty:
            return cached
    import akshare as ak

    raw = ak.stock_hk_index_daily_em(symbol="HSAHP")
    close_col = "latest" if "latest" in raw.columns else raw.columns[1]
    series = pd.Series(
        pd.to_numeric(raw[close_col], errors="coerce").to_numpy(),
        index=_as_dates(raw["date"] if "date" in raw.columns else raw.iloc[:, 0]),
        name="hsahp",
    )
    series = series[series > 0]
    series = series[~series.index.duplicated(keep="last")].dropna().sort_index()
    if series.empty:
        raise RuntimeError("没有下载到恒生 AH 溢价指数")
    return _store(path, series, "hsahp")


def describe_market(spread: pd.Series, factor: pd.Series) -> dict:
    """和北向同一套 HAC 回归，只是把解释变量换成别的市场序列。"""
    result = describe_northbound(spread, factor)
    return result
