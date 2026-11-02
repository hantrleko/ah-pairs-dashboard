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
    return cache_dir / f"{safe}.parquet"


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
    frame.index.name = "date"
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp)
    tmp.replace(path)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if "date" in frame.columns and not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.set_index("date")
    return frame


def _read_series(path: Path) -> pd.Series:
    frame = _read_frame(path)
    series = frame.iloc[:, 0].astype(float)
    series.index = _normalize_index(series.index)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = "close"
    return series.dropna()


def _cached_series(path: Path) -> pd.Series:
    legacy = path.with_suffix(".csv")
    if path.exists():
        return _read_series(path)
    if legacy.exists():
        return _read_series(legacy)
    return pd.Series(dtype=float)


def _cache_covers(series: pd.Series, today: pd.Timestamp | None = None) -> bool:
    if series.empty:
        return False
    today = pd.Timestamp.today().normalize() if today is None else today.normalize()
    last = series.index.max()
    return last >= today - pd.Timedelta(days=5)


def _yahoo_close(ticker: str) -> pd.Series:
    import socket

    import yfinance as yf

    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(20)
    try:
        downloaded = yf.download(
            ticker,
            start=HISTORY_START,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=20,
        )
    finally:
        socket.setdefaulttimeout(old_timeout)
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
    cached = _cached_series(path)
    if not refresh and _cache_covers(cached):
        return cached, "cache"

    yahoo_error = ""
    try:
        fresh = _yahoo_close(ticker)
    except Exception as exc:
        fresh = pd.Series(dtype=float)
        yahoo_error = str(exc)
    source = "yahoo"
    if fresh.empty and not cached.empty and not refresh:
        return cached, "行情下载失败，沿用本地缓存"
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


def assemble_pair(a_px, h_px, fx, start, end) -> tuple[pd.DataFrame, list[str]]:
    """把两条价格和汇率拼成回测表。

    协整和信号只用两边都有成交的日子。只有一边开市时，已有仓位仍计入开市那边的涨跌，
    收盘那边的收益记 0，当天不新开仓、也不调 β。股价只为了记账向前填充，不拿填充价重新回归。
    """
    warnings: list[str] = []
    a = pd.Series(a_px, dtype=float).copy()
    h = pd.Series(h_px, dtype=float).copy()
    fx_s = pd.Series(fx, dtype=float).copy()
    a.index = _normalize_index(a.index)
    h.index = _normalize_index(h.index)
    fx_s.index = _normalize_index(fx_s.index)
    a = a[np.isfinite(a) & (a > 0)]
    h = h[np.isfinite(h) & (h > 0)]
    a = a[~a.index.duplicated(keep="last")].sort_index()
    h = h[~h.index.duplicated(keep="last")].sort_index()
    px = pd.concat({"a_raw": a, "h_raw": h}, axis=1).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    px = px.loc[(px.index >= start_ts) & (px.index <= end_ts)]
    if px.empty:
        raise RuntimeError("这个区间里没有行情")
    px["both_open"] = px["a_raw"].notna() & px["h_raw"].notna()
    px["a_cny"] = px["a_raw"].ffill()
    px["h_hkd"] = px["h_raw"].ffill()
    fx_raw = fx_s.reindex(px.index)
    fx_aligned = fx_raw.ffill(limit=5)
    lagged = int((fx_raw.isna() & fx_aligned.notna()).sum())
    if lagged:
        warnings.append(f"meta:fx_lag:{lagged}")
        warnings.append(f"汇率有 {lagged} 天沿用了此前最多 5 个交易日的牌价。")
    px["fx"] = fx_aligned
    px = px.dropna(subset=["a_cny", "h_hkd", "fx"])
    if px.empty or not bool(px["both_open"].any()):
        raise RuntimeError("这个区间里没有 A/H 同时开市的交易日")
    common = int(px["both_open"].sum())
    one_sided = int((~px["both_open"]).sum())
    warnings.append(f"meta:sessions:{common}:{one_sided}")
    warnings.append(
        f"含预热的下载区间：同时开市 {common} 天，只有一边开市 {one_sided} 天。"
        "协整和信号只用同时开市日；已有仓位在单边开市日仍计算开市一边的收益，当天不调仓。"
    )
    px["h_cny"] = px["h_hkd"] * px["fx"]
    px["premium"] = px["a_cny"] / px["h_cny"] - 1.0
    px["log_a"] = np.log(px["a_cny"])
    px["log_h"] = np.log(px["h_cny"])
    px["ret_a"] = px["a_cny"].pct_change().where(px["a_raw"].notna(), 0.0)
    px["ret_h"] = px["h_cny"].pct_change().where(px["h_raw"].notna(), 0.0)
    px = px.drop(columns=["a_raw", "h_raw"])
    px.index.name = "date"
    return px, warnings


def load_pair(
    a_code: str,
    h_code: str,
    start,
    end,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """下载一对 A+H，并把 H 股换成人民币。start/end 含首尾。"""
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
    warnings.append(f"meta:src:a:{a_src}")
    warnings.append(f"meta:src:h:{h_src}")
    warnings.append(f"meta:src:fx:{fx_src}")
    warnings.append(f"A股来源：{a_src}")
    warnings.append(f"H股来源：{h_src}")
    warnings.append(f"汇率来源：{fx_src}")
    if "失败" in a_src or "失败" in h_src or "失败" in fx_src:
        warnings.append("meta:download_failed:1")
        warnings.append("有一条行情下载失败，图上用的是本地缓存。")
    if "eastmoney" in {a_src, h_src} or "沿用" in a_src or "沿用" in h_src:
        warnings.append("meta:adjust_mismatch:1")
        warnings.append("两条腿的复权来源可能不一致，价差水平要谨慎解读。")
    px, align_notes = assemble_pair(a_px, h_px, fx, start, end)
    warnings.extend(align_notes)
    px.attrs["a_code"] = str(a_code).zfill(6)
    px.attrs["h_code"] = str(h_code).zfill(5)
    return px, warnings


def unadjusted_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("=", "_").replace("^", "_")
    return cache_dir / f"{safe}.unadj.parquet"


def load_unadjusted_close(
    ticker: str,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> pd.Series:
    """未复权收盘，只给稳定性对照。主回测仍用复权价。"""
    path = unadjusted_path(cache_dir, ticker)
    if path.exists() and not refresh:
        return _read_series(path)
    import yfinance as yf

    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    downloaded = yf.download(
        ticker,
        start=HISTORY_START,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=20,
    )
    if downloaded is None or downloaded.empty or "Close" not in downloaded.columns:
        raise RuntimeError(f"没有下载到 {ticker} 的未复权收盘")
    close = downloaded["Close"]
    series = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
    series = pd.Series(series.to_numpy(dtype=float), index=_normalize_index(series.index))
    series = series[~series.index.duplicated(keep="last")].sort_index().dropna()
    if series.empty:
        raise RuntimeError(f"没有下载到 {ticker} 的未复权收盘")
    _write_series(path, series)
    return series


def unadjusted_log_columns(
    a_code: str,
    h_code: str,
    fx,
    index,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """按主表的日期对齐未复权对数价格。H 股仍乘同一条汇率。"""
    a = load_unadjusted_close(a_to_yahoo(a_code), cache_dir, refresh=refresh)
    h = load_unadjusted_close(h_to_yahoo(h_code), cache_dir, refresh=refresh)
    frame = pd.DataFrame(index=pd.DatetimeIndex(index))
    a_px = a.reindex(frame.index).ffill()
    h_px = h.reindex(frame.index).ffill()
    fx_s = pd.Series(fx, dtype=float)
    fx_s.index = _normalize_index(fx_s.index)
    fx_aligned = fx_s.reindex(frame.index).ffill()
    h_cny = h_px * fx_aligned
    return pd.DataFrame(
        {
            "log_a_raw": np.log(a_px.where(a_px > 0)),
            "log_h_raw": np.log(h_cny.where(h_cny > 0)),
        },
        index=frame.index,
    )
