"""行情下载与缓存。价格优先用 Yahoo 复权收盘，失败时用东财前复权。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ah_pairs.paths import CACHE_DIR
from ah_pairs.universe import a_to_yahoo, h_to_yahoo

HISTORY_START = "2010-01-01"


def _cache_file(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("=", "_").replace("^", "_")
    return cache_dir / f"{safe}.csv"


def _normalize_index(index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(idx).normalize()


def _write_series(path: Path, series: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = series.dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    frame = out.rename("close").to_frame()
    tmp = path.with_suffix(".tmp.csv")
    frame.to_csv(tmp, encoding="utf-8")
    tmp.replace(path)


def _read_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    series = frame.iloc[:, 0].astype(float)
    series.index = _normalize_index(series.index)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = "close"
    return series.dropna()


def _cache_covers(series: pd.Series, today: pd.Timestamp | None = None) -> bool:
    if series.empty:
        return False
    today = pd.Timestamp.today().normalize() if today is None else today.normalize()
    last = series.index.max()
    return last >= today - pd.Timedelta(days=5)


def _yahoo_close(ticker: str) -> pd.Series:
    import yfinance as yf

    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    downloaded = yf.download(
        ticker,
        start=HISTORY_START,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if downloaded is None or downloaded.empty:
        return pd.Series(dtype=float)
    if isinstance(downloaded.columns, pd.MultiIndex):
        close = downloaded["Close"]
        series = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
    else:
        series = downloaded["Close"]
    series = pd.Series(series.to_numpy(dtype=float), index=_normalize_index(series.index))
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.dropna()


def _eastmoney_close(kind: str, code: str) -> pd.Series:
    import akshare as ak

    end = pd.Timestamp.today().strftime("%Y%m%d")
    if kind == "a":
        raw = ak.stock_zh_a_hist(
            symbol=str(code).zfill(6),
            period="daily",
            start_date="20100101",
            end_date=end,
            adjust="qfq",
        )
    elif kind == "h":
        raw = ak.stock_hk_hist(
            symbol=str(code).zfill(5),
            period="daily",
            start_date="20100101",
            end_date=end,
            adjust="qfq",
        )
    else:
        raise ValueError(kind)
    if raw is None or raw.empty or "收盘" not in raw.columns:
        return pd.Series(dtype=float)
    series = pd.Series(
        pd.to_numeric(raw["收盘"], errors="coerce").to_numpy(),
        index=_normalize_index(raw["日期"]),
    )
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.dropna()


def get_close(
    ticker: str,
    cache_dir: Path = CACHE_DIR,
    kind: str | None = None,
    code: str | None = None,
    refresh: bool = False,
) -> tuple[pd.Series, str]:
    """返回 (收盘价, 来源)。来源是 cache、yahoo 或 eastmoney。"""
    path = _cache_file(cache_dir, ticker)
    cached = _read_series(path) if path.exists() else pd.Series(dtype=float)
    if not refresh and _cache_covers(cached):
        return cached, "cache"

    yahoo_error = ""
    try:
        fresh = _yahoo_close(ticker)
    except Exception as exc:
        fresh = pd.Series(dtype=float)
        yahoo_error = str(exc)
    source = "yahoo"
    if fresh.empty and kind in {"a", "h"} and code:
        try:
            fresh = _eastmoney_close(kind, code)
            source = "eastmoney"
        except Exception:
            fresh = pd.Series(dtype=float)
    if fresh.empty:
        if not cached.empty:
            note = "沿用本地缓存" if not yahoo_error else "行情下载失败，沿用本地缓存"
            return cached, note
        raise RuntimeError(f"下载 {ticker} 失败。{yahoo_error}".strip())
    _write_series(path, fresh)
    return fresh, source


def get_fx(cache_dir: Path = CACHE_DIR, refresh: bool = False) -> tuple[pd.Series, str]:
    """1 港元兑人民币。Yahoo 代码 HKDCNY=X。"""
    series, source = get_close("HKDCNY=X", cache_dir=cache_dir, refresh=refresh)
    if series.empty:
        raise RuntimeError("没有下载到港元兑人民币汇率")
    median = float(series.median())
    if median > 2:
        series = 1.0 / series
        source = source + "+inverted"
    elif not (0.6 <= median <= 1.3):
        raise RuntimeError(f"港元汇率中位数 {median:.3f} 不在合理区间")
    series.name = "fx"
    return series, source


def load_pair(
    a_code: str,
    h_code: str,
    start,
    end,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """对齐两地都开市的交易日，并把 H 股换成人民币。

    start/end 含首尾。两地节假日不一致的日期删掉，不用前值填充股价。
    """
    warnings: list[str] = []
    a_ticker = a_to_yahoo(a_code)
    h_ticker = h_to_yahoo(h_code)
    a_px, a_src = get_close(
        a_ticker, cache_dir, kind="a", code=a_code, refresh=refresh
    )
    h_px, h_src = get_close(
        h_ticker, cache_dir, kind="h", code=h_code, refresh=refresh
    )
    fx, fx_src = get_fx(cache_dir, refresh=refresh)
    if a_src != "cache":
        warnings.append(f"A 股来源：{a_src}")
    if h_src != "cache":
        warnings.append(f"H 股来源：{h_src}")
    if fx_src != "cache":
        warnings.append(f"汇率来源：{fx_src}")
    if "eastmoney" in {a_src, h_src} or "沿用" in a_src or "沿用" in h_src:
        warnings.append("两条腿的复权来源可能不一致，价差水平要谨慎解读。")

    px = pd.concat({"a_cny": a_px, "h_hkd": h_px}, axis=1, join="inner").sort_index()
    px = px[(px["a_cny"] > 0) & (px["h_hkd"] > 0)]
    fx_aligned = fx.reindex(px.index).ffill(limit=5)
    px["fx"] = fx_aligned
    px = px.dropna()
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    px = px.loc[(px.index >= start_ts) & (px.index <= end_ts)]
    if px.empty:
        raise RuntimeError("这个区间里没有 A/H 同时开市的交易日")
    px["h_cny"] = px["h_hkd"] * px["fx"]
    px["premium"] = px["a_cny"] / px["h_cny"] - 1.0
    px["log_a"] = np.log(px["a_cny"])
    px["log_h"] = np.log(px["h_cny"])
    px["ret_a"] = px["a_cny"].pct_change()
    px["ret_h"] = px["h_cny"].pct_change()
    px.index.name = "date"
    px.attrs["a_code"] = str(a_code).zfill(6)
    px.attrs["h_code"] = str(h_code).zfill(5)
    return px, warnings
