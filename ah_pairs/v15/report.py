"""页面上用的 V1.5 快照。这里不算样本外折。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ah_pairs.v15.adaptive import half_life_tail, latest_half_life
from ah_pairs.v15.net_premium import SCENARIOS, dividend_gap_note, net_z, trailing_yield, wedge_annual, wedge_is_implausible


def result_key(a_code: str, start, end, allow_short_a: bool) -> str:
    return f"v15|{a_code}|{pd.Timestamp(start).date()}|{pd.Timestamp(end).date()}|{int(bool(allow_short_a))}"


def preview(frame: pd.DataFrame, calendar: pd.DataFrame | None = None) -> dict:
    log_a = frame["log_a"].astype("float64")
    log_h = frame["log_h"].astype("float64")
    premium = (log_a - log_h).astype("float64")
    gaps = dividend_gap_note(calendar, frame.index.min(), frame.index.max())
    wedge = np.zeros(len(frame), dtype=np.float64)
    annual = np.nan
    if calendar is not None and not calendar.empty and "a_cny" in frame.columns:
        last = frame.index[-1]
        price_a = float(frame["a_cny"].iloc[-1])
        price_h = float(frame["h_hkd"].iloc[-1]) if "h_hkd" in frame.columns else np.nan
        d_a = trailing_yield(calendar, "A", last, price_a)
        d_h = trailing_yield(calendar, "H", last, price_h)
        if np.isfinite(d_a) and np.isfinite(d_h):
            tau_a, tau_h = SCENARIOS["base"]
            annual = wedge_annual(d_a, d_h, tau_a, tau_h)
            wedge[:] = annual
            gaps = []
    z = net_z(premium, wedge, 60).dropna()
    life = latest_half_life(log_a, log_h, 120)
    path = half_life_tail(log_a, log_h, window=120, tail=90)
    finite = path[np.isfinite(path)]
    return {
        "z": float(z.iloc[-1]) if len(z) else np.nan,
        "half_life": float(life) if np.isfinite(life) else np.nan,
        "half_life_low": float(np.min(finite)) if len(finite) else np.nan,
        "half_life_high": float(np.max(finite)) if len(finite) else np.nan,
        "wedge_annual": float(annual) if np.isfinite(annual) else np.nan,
        "wedge_implausible": wedge_is_implausible(annual) if np.isfinite(annual) else False,
        "gaps": gaps,
    }
