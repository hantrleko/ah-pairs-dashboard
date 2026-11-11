"""净溢价。第一层用后复权价，第二层用税后股息差移动锚点。

两腿都是收益率时，换成人民币会把汇率约掉，所以楔子不再乘一次汇率。
窗口里的平均楔子留在滚动均值里；再减一次会把 z 的均值推离 0。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIOS = {
    "optimistic": (0.00, 0.10),
    "base": (0.10, 0.20),
    "conservative": (0.20, 0.20),
}
WEDGE_ANNUAL_LIMIT = 0.05
YEAR_DAYS = 244


def log_premium(a_adj, h_adj, fx) -> pd.Series:
    a = np.asarray(a_adj, dtype=np.float64)
    h = np.asarray(h_adj, dtype=np.float64)
    rate = np.asarray(fx, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log(a) - np.log(h) - np.log(rate)
    index = getattr(a_adj, "index", None)
    return pd.Series(values, index=index, dtype="float64", name="premium_log")


def tau_a_for_holding_days(days: int) -> float:
    """持股天数用日历天。满 30 天仍算不超过一个月。"""
    held = int(days)
    if held <= 30:
        return 0.20
    if held <= 365:
        return 0.10
    return 0.0


def wedge_annual(d_a: float, d_h: float, tau_a: float, tau_h: float) -> float:
    """税后股息率之差，单位是每年的对数溢价。d 是近 12 个月 DPS / 价格。"""
    return float(d_a) * (1.0 - float(tau_a)) - float(d_h) * (1.0 - float(tau_h))


def wedge_is_implausible(annual: float) -> bool:
    return bool(np.isfinite(annual) and abs(float(annual)) > WEDGE_ANNUAL_LIMIT)


def wedge_over_hold(annual: float, half_life: float, year_days: int = YEAR_DAYS) -> float:
    if not np.isfinite(half_life) or half_life <= 0:
        return 0.0
    return float(annual) * (float(half_life) / float(year_days))


def net_z(premium, wedge, window: int) -> pd.Series:
    """z = (premium - μ - (wedge - μ_wedge)) / σ。常数楔子不再重复计入。"""
    prem = pd.Series(np.asarray(premium, dtype=np.float64), index=getattr(premium, "index", None), dtype="float64")
    wed = pd.Series(np.asarray(wedge, dtype=np.float64), index=prem.index, dtype="float64")
    span = int(window)
    mu = prem.rolling(span, min_periods=span).mean()
    sigma = prem.rolling(span, min_periods=span).std(ddof=1)
    wed_mu = wed.rolling(span, min_periods=span).mean()
    sigma = sigma.where(sigma > 0)
    values = (prem - mu - (wed - wed_mu)) / sigma
    return pd.Series(values.to_numpy(dtype=np.float64), index=prem.index, dtype="float64", name="z_net")


def clean_dividend_calendar(calendar: pd.DataFrame, a_dates, h_dates) -> tuple[pd.DataFrame, list[str]]:
    """丢掉非正股息和没有落在该腿开市日上的除净日。丢弃原因写进日志。"""
    empty = pd.DataFrame(columns=["leg", "ex_date", "dps", "record_date", "source"])
    if calendar is None or len(calendar) == 0:
        return empty, ["股息日历是空的"]
    a_open = set(pd.DatetimeIndex(pd.to_datetime(a_dates)).normalize())
    h_open = set(pd.DatetimeIndex(pd.to_datetime(h_dates)).normalize())
    kept = []
    log: list[str] = []
    for row in calendar.itertuples(index=False):
        leg = str(getattr(row, "leg", "")).upper()
        dps = getattr(row, "dps", np.nan)
        ex = pd.Timestamp(getattr(row, "ex_date"))
        ex = ex.tz_convert("UTC").tz_localize(None).normalize() if ex.tzinfo else ex.normalize()
        if leg not in {"A", "H"}:
            log.append(f"丢弃 {ex.date()}：腿 {leg} 无法识别")
            continue
        if not np.isfinite(dps) or float(dps) <= 0:
            log.append(f"丢弃 {leg} {ex.date()}：股息不是正数")
            continue
        sessions = a_open if leg == "A" else h_open
        if ex not in sessions:
            log.append(f"丢弃 {leg} {ex.date()}：除净日不在该腿开市日")
            continue
        record = getattr(row, "record_date", None)
        kept.append(
            {
                "leg": leg,
                "ex_date": ex,
                "dps": float(dps),
                "record_date": None if record is None or (isinstance(record, float) and not np.isfinite(record)) else pd.Timestamp(record),
                "source": str(getattr(row, "source", "")),
            }
        )
    if not kept:
        log.append("质检后没有可用股息")
        return empty, log
    out = pd.DataFrame(kept)
    out["dps"] = out["dps"].astype("float64")
    return out, log


def trailing_yield(calendar: pd.DataFrame, leg: str, asof, price: float, lookback_days: int = 365) -> float:
    if calendar is None or calendar.empty or not np.isfinite(price) or price <= 0:
        return np.nan
    end = pd.Timestamp(asof).normalize()
    start = end - pd.Timedelta(days=int(lookback_days))
    rows = calendar[calendar["leg"] == leg]
    if rows.empty:
        return np.nan
    ex = pd.to_datetime(rows["ex_date"]).dt.normalize()
    picked = rows.loc[(ex > start) & (ex <= end), "dps"]
    if picked.empty:
        return 0.0
    return float(picked.astype("float64").sum() / float(price))


def dividend_gap_note(calendar: pd.DataFrame | None, start, end) -> list[str]:
    """没有日历时整段都要写明，不能当成零股息。"""
    if calendar is None or calendar.empty:
        return [f"{pd.Timestamp(start).date()} 至 {pd.Timestamp(end).date()} 没有股息日历"]
    return []
