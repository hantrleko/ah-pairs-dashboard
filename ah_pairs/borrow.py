"""A 股融券标的名单。

名单来自融资融券标的证券及保证金比例，不是券商的融券年化费率。
只有名单覆盖到的日期才限制做空；没覆盖的日子仍按侧栏开关，不把今天的资格套到整个历史。
打开页面只读本地文件。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ah_pairs.paths import CACHE_DIR, PUBLISHED_DIR
from ah_pairs.store import prefer, publish_file, same_dir


def _path(directory: Path) -> Path:
    return directory / "borrow.parquet"


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    frame = pd.read_parquet(path)
    if frame.empty or "a_code" not in frame.columns or "date" not in frame.columns:
        return None
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["a_code"] = frame["a_code"].astype(str).str.zfill(6)
    return frame


def read_borrow(cache_dir: Path | None = None) -> pd.DataFrame | None:
    directory = CACHE_DIR if cache_dir is None else cache_dir
    path = prefer(_path(directory), _path(PUBLISHED_DIR), directory, CACHE_DIR)
    if path is None:
        return None
    return _read(path)


def _code_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        text = str(column)
        if "代码" in text or text.lower() in {"code", "symbol"}:
            return column
    return frame.columns[0]


def _ratio_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if "融券" in str(column):
            return column
    return None


def _as_ratio(value) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return np.nan
    number = float(number)
    if number > 1.5:
        return number / 100.0
    return number


def _parse_margin_list(raw: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "a_code", "eligible", "margin_ratio"])
    codes = raw[_code_column(raw)].astype(str).str.extract(r"(\d{6})", expand=False)
    ratio_col = _ratio_column(raw)
    if ratio_col is None:
        ratios = np.full(len(raw), np.nan)
    else:
        ratios = raw[ratio_col].map(_as_ratio).to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "date": pd.Timestamp(day).normalize(),
            "a_code": codes,
            "eligible": True,
            "margin_ratio": ratios,
        }
    )
    return out.dropna(subset=["a_code"]).drop_duplicates(["date", "a_code"])


def _download_snapshot() -> pd.DataFrame:
    import akshare as ak

    errors = []
    today = pd.Timestamp.today().normalize()
    for shift in range(0, 8):
        day = today - pd.Timedelta(days=shift)
        frames = []
        for market in ("沪市", "深市"):
            try:
                raw = ak.stock_margin_ratio_pa(symbol=market, date=day.strftime("%Y%m%d"))
            except Exception as exc:
                errors.append(f"{market} {day.date()}: {exc}")
                continue
            part = _parse_margin_list(raw, day)
            if not part.empty:
                frames.append(part)
        if frames:
            return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "a_code"])
    detail = "；".join(errors[:4]) if errors else "空名单"
    raise RuntimeError(f"没有下载到融券标的名单。{detail}")


def _write(frame: pd.DataFrame, directory: Path) -> Path:
    path = _path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def _merge(old: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return fresh
    fresh_days = set(pd.to_datetime(fresh["date"]).dt.normalize())
    keep = old.loc[~pd.to_datetime(old["date"]).dt.normalize().isin(fresh_days)]
    return pd.concat([keep, fresh], ignore_index=True)


def load_borrow(cache_dir: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    directory = CACHE_DIR if cache_dir is None else cache_dir
    cached = read_borrow(directory)
    if not refresh:
        if cached is not None and not cached.empty:
            return cached
        raise FileNotFoundError("还没有融券标的名单")
    fresh = _download_snapshot()
    merged = _merge(cached, fresh)
    src = _write(merged, directory)
    if same_dir(directory, CACHE_DIR):
        publish_file(src, _path(PUBLISHED_DIR))
    return merged


def borrow_token_for(index, a_code: str, cache_dir: Path | None = None) -> str:
    """覆盖到的日期写成 date:0 或 date:1。没有名单时返回空字符串，回测保持原样。"""
    calendar = read_borrow(cache_dir)
    if calendar is None or calendar.empty:
        return ""
    code = str(a_code).zfill(6)
    days = pd.to_datetime(calendar["date"]).dt.normalize()
    covered = set(days)
    eligible = set(zip(days, calendar["a_code"].astype(str).str.zfill(6)))
    parts = []
    for day in pd.DatetimeIndex(pd.to_datetime(index)).normalize():
        if day not in covered:
            continue
        bit = "1" if (day, code) in eligible else "0"
        parts.append(f"{day.date().isoformat()}:{bit}")
    return ",".join(parts)


def mask_from_borrow_token(index, token: str) -> np.ndarray:
    """没写进 token 的日子保持允许。token 为空时不应调用。"""
    mask = np.ones(len(index), dtype=bool)
    lookup = {}
    for part in token.split(","):
        if ":" not in part:
            continue
        day, bit = part.split(":", 1)
        lookup[pd.Timestamp(day).normalize()] = bit == "1"
    for i, day in enumerate(pd.DatetimeIndex(pd.to_datetime(index)).normalize()):
        if day in lookup:
            mask[i] = lookup[day]
    return mask


def borrow_status(a_code: str, cache_dir: Path | None = None) -> dict:
    calendar = read_borrow(cache_dir)
    if calendar is None or calendar.empty:
        return {"ok": False}
    code = str(a_code).zfill(6)
    days = pd.to_datetime(calendar["date"]).dt.normalize()
    latest = days.max()
    current = calendar.loc[days == latest]
    codes = set(current["a_code"].astype(str).str.zfill(6))
    ratio = np.nan
    if code in codes and "margin_ratio" in current.columns:
        hit = current.loc[current["a_code"].astype(str).str.zfill(6) == code, "margin_ratio"]
        if len(hit):
            ratio = float(hit.iloc[-1])
    return {
        "ok": True,
        "asof": pd.Timestamp(latest).date().isoformat(),
        "n_dates": int(days.nunique()),
        "n_names": int(current["a_code"].nunique()),
        "eligible": code in codes,
        "margin_ratio": ratio,
        "covered_days": int(days.nunique()),
    }
