"""股息日历只放在 data/dividends。打开页面只读磁盘。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ah_pairs.paths import DATA_DIR
from ah_pairs.universe import a_to_yahoo, h_to_yahoo
from ah_pairs.v15.net_premium import clean_dividend_calendar

DIVIDEND_DIR = DATA_DIR / "dividends"


def calendar_path(a_code: str, h_code: str, folder: Path = DIVIDEND_DIR) -> Path:
    return folder / f"{a_code}_{h_code}.parquet"


def read_dividend_calendar(a_code: str, h_code: str, folder: Path = DIVIDEND_DIR) -> pd.DataFrame | None:
    path = calendar_path(a_code, h_code, folder)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "ex_date" in frame.columns:
        frame["ex_date"] = pd.to_datetime(frame["ex_date"])
    return frame


def write_dividend_calendar(frame: pd.DataFrame, a_code: str, h_code: str, folder: Path = DIVIDEND_DIR) -> Path:
    path = calendar_path(a_code, h_code, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def download_dividend_calendar(a_code: str, h_code: str, a_dates, h_dates, folder: Path = DIVIDEND_DIR) -> tuple[pd.DataFrame, list[str]]:
    """按钮才会调用。Yahoo 的现金股息写成日历，再按开市日质检。"""
    import socket

    import yfinance as yf

    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(20)
    rows = []
    try:
        for leg, ticker in (("A", a_to_yahoo(a_code)), ("H", h_to_yahoo(h_code))):
            series = yf.Ticker(ticker).dividends
            if series is None or len(series) == 0:
                continue
            for stamp, value in series.items():
                rows.append({"leg": leg, "ex_date": pd.Timestamp(stamp), "dps": float(value), "record_date": None, "source": "yahoo"})
    finally:
        socket.setdefaulttimeout(previous)
    raw = pd.DataFrame(rows)
    cleaned, log = clean_dividend_calendar(raw, a_dates, h_dates)
    if not cleaned.empty:
        write_dividend_calendar(cleaned, a_code, h_code, folder)
    return cleaned, log
