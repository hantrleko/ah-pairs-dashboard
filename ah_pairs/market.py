"""市场层面的对照：南向资金，以及恒生沪深港通 AH 股溢价指数（HSAHP）。

北向日频在 2024-08 之后停止。南向和 AH 溢价指数打开页面时只读本地缓存。
用户点更新才访问网络。HSAHP 先试 Yahoo，再试东财；都失败时沿用已有缓存。
"""

from __future__ import annotations

import socket
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from ah_pairs.northbound import _as_dates, describe_northbound
from ah_pairs.paths import CACHE_DIR


@contextmanager
def _timeout(seconds: float):
    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(old)


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


def read_southbound(cache_dir: Path = CACHE_DIR) -> pd.Series | None:
    """只读本地文件，不访问网络。"""
    return _load_cached(cache_dir / "southbound.parquet", "southbound")


def read_ah_premium_index(cache_dir: Path = CACHE_DIR) -> pd.Series | None:
    return _load_cached(cache_dir / "hsahp.parquet", "hsahp")


def _fetch_southbound() -> pd.Series:
    import akshare as ak

    raw = ak.stock_hsgt_hist_em(symbol="南向资金")
    series = pd.Series(
        pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(),
        index=_as_dates(raw.iloc[:, 0]),
        name="southbound",
    )
    return series[~series.index.duplicated(keep="last")].dropna().sort_index()


def _yahoo_index(ticker: str) -> pd.Series:
    import yfinance as yf

    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    downloaded = yf.download(
        ticker,
        start="2010-01-01",
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=20,
    )
    if downloaded is None or downloaded.empty or "Close" not in getattr(downloaded, "columns", []):
        return pd.Series(dtype=float, name="hsahp")
    close = downloaded["Close"]
    series = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
    out = pd.Series(series.to_numpy(dtype=float), index=_as_dates(series.index), name="hsahp")
    out = out[out > 0].dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if out.empty or float(out.median()) < 30:
        return pd.Series(dtype=float, name="hsahp")
    return out


def _eastmoney_hsahp() -> pd.Series:
    import akshare as ak

    raw = ak.stock_hk_index_daily_em(symbol="HSAHP")
    close_col = "latest" if "latest" in raw.columns else raw.columns[1]
    series = pd.Series(
        pd.to_numeric(raw[close_col], errors="coerce").to_numpy(),
        index=_as_dates(raw["date"] if "date" in raw.columns else raw.iloc[:, 0]),
        name="hsahp",
    )
    series = series[series > 0]
    return series[~series.index.duplicated(keep="last")].dropna().sort_index()


def _pull_hsahp() -> tuple[pd.Series, str]:
    """先 Yahoo，再东财。返回 (序列, 来源)。"""
    errors = []
    for ticker in ("HSAHP.HK", "^HSAHP"):
        try:
            with _timeout(20):
                series = _yahoo_index(ticker)
            if series is not None and not series.empty:
                return series, f"yahoo:{ticker}"
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
    try:
        with _timeout(20):
            series = _eastmoney_hsahp()
        if series is not None and not series.empty:
            return series, "eastmoney"
    except Exception as exc:
        errors.append(str(exc))
    detail = "; ".join(errors) if errors else "空序列"
    raise RuntimeError(f"没有下载到恒生 AH 溢价指数。{detail}")


def _write_source(path: Path, source: str) -> None:
    path.with_suffix(".source").write_text(source, encoding="utf-8")


def read_source(path: Path) -> str:
    note = path.with_suffix(".source")
    if note.exists():
        return note.read_text(encoding="utf-8").strip()
    return ""


def refresh_or_keep(cached: pd.Series | None, fetch, store) -> tuple[pd.Series, str]:
    """拉取失败但本地有数时，返回旧序列和错误说明，不把别的指数拿来冒充。"""
    try:
        fresh = fetch()
        if isinstance(fresh, tuple):
            series, source = fresh
        else:
            series, source = fresh, ""
        if series is None or series.empty:
            raise RuntimeError("下载结果是空的")
        stored = store(series)
        return stored, source
    except Exception as exc:
        if cached is not None and not cached.dropna().empty:
            end = pd.Timestamp(cached.dropna().index.max()).date()
            return cached, f"沿用截至 {end} 的缓存。更新失败：{exc}"
        raise


def load_southbound(cache_dir: Path = CACHE_DIR, refresh: bool = False) -> pd.Series:
    """港股通合计净买入，单位亿元。不是这一只 H 股的南向持仓。

    refresh 为假时只读缓存。没有文件就抛出 FileNotFoundError，由页面提示再拉取。
    """
    path = cache_dir / "southbound.parquet"
    cached = _load_cached(path, "southbound")
    if not refresh:
        if cached is not None and not cached.empty:
            return cached
        raise FileNotFoundError("还没有南向资金缓存")
    def _pull_south():
        with _timeout(20):
            return _fetch_southbound(), "eastmoney"

    series, note = refresh_or_keep(
        cached,
        _pull_south,
        lambda fresh: _store(path, fresh, "southbound"),
    )
    if note and not note.startswith("沿用"):
        _write_source(path, note)
    if note.startswith("沿用"):
        raise RuntimeError(note)
    return series


def load_ah_premium_index(cache_dir: Path = CACHE_DIR, refresh: bool = False) -> pd.Series:
    """恒生沪深港通 AH 股溢价指数收盘。100 表示样本内 A/H 平均平价，高于 100 表示 A 股更贵。"""
    path = cache_dir / "hsahp.parquet"
    cached = _load_cached(path, "hsahp")
    if not refresh:
        if cached is not None and not cached.empty:
            return cached
        raise FileNotFoundError("还没有恒生 AH 溢价指数缓存")
    series, note = refresh_or_keep(cached, _pull_hsahp, lambda fresh: _store(path, fresh, "hsahp"))
    if note and not note.startswith("沿用"):
        _write_source(path, note)
    if note.startswith("沿用"):
        raise RuntimeError(note)
    return series


def describe_market(spread: pd.Series, factor: pd.Series) -> dict:
    """和北向同一套 HAC 回归，只是把解释变量换成别的市场序列。"""
    result = describe_northbound(spread, factor)
    return result
