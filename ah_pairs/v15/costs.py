"""V1.5 成本。费率按成交日取，不把一整段历史用成一个数。

点差和冲击是估计。印花税、过户费按公布的生效日。股息税不进成交成本。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

A_TRANSFER = 0.00001
A_TRANSFER_FROM = pd.Timestamp("2022-04-29")
A_STAMP_HALVED = pd.Timestamp("2023-08-28")
H_STAMP_DOWN = pd.Timestamp("2023-11-17")
H_STAMP_UP = pd.Timestamp("2021-08-01")
H_LEVY_DEFAULT = 0.0000565
FX_COST = 0.0005
IMPACT_K = 0.5
A_SPREAD_FALLBACK = 0.0005
H_SPREAD_FALLBACK = 0.0010


@dataclass(frozen=True)
class CostSchedule:
    a_commission: float = 0.00025
    h_commission: float = 0.00025
    a_commission_min: float = 5.0
    h_commission_min_hkd: float = 100.0
    h_levy: float = H_LEVY_DEFAULT
    fx_cost: float = FX_COST
    impact_k: float = IMPACT_K


def _day(day) -> pd.Timestamp:
    stamp = pd.Timestamp(day)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


def a_stamp_rate(day) -> float:
    """A 股卖出印花税。买入为 0。"""
    if _day(day) >= A_STAMP_HALVED:
        return 0.0005
    return 0.001


def a_transfer_rate(day) -> float:
    if _day(day) >= A_TRANSFER_FROM:
        return A_TRANSFER
    return 0.0


def h_stamp_rate(day) -> float:
    if _day(day) >= H_STAMP_DOWN:
        return 0.001
    if _day(day) >= H_STAMP_UP:
        return 0.0013
    return 0.001


def h_stamp_hkd(notional_hkd: float, day) -> float:
    """港股印花税向上取整到 1 港元。"""
    raw = float(notional_hkd) * h_stamp_rate(day)
    if raw <= 0:
        return 0.0
    return float(math.ceil(raw - 1e-9))


def _commission(notional: float, rate: float, minimum: float) -> float:
    if notional <= 0:
        return 0.0
    return float(max(notional * rate, minimum))


def fill_cost(
    day,
    leg: str,
    side: str,
    notional_cny: float,
    fx: float,
    schedule: CostSchedule | None = None,
    half_spread: float | None = None,
    adv_cny: float | None = None,
    charge_fx: bool = True,
) -> dict:
    """一笔成交的成本，金额是人民币。total 等于各分项之和。"""
    schedule = schedule or CostSchedule()
    notional = float(notional_cny)
    rate_fx = float(fx) if fx and np.isfinite(fx) and fx > 0 else np.nan
    stamp = transfer = commission = levies = spread = impact = fx_cost = 0.0
    if notional > 0 and np.isfinite(notional):
        if leg == "A":
            if side == "sell":
                stamp = notional * a_stamp_rate(day)
            transfer = notional * a_transfer_rate(day)
            commission = _commission(notional, schedule.a_commission, schedule.a_commission_min)
            spread_rate = A_SPREAD_FALLBACK if half_spread is None or not np.isfinite(half_spread) else float(half_spread)
        else:
            if not np.isfinite(rate_fx) or rate_fx <= 0:
                raise ValueError("H 股成交需要港元汇率")
            notional_hkd = notional / rate_fx
            stamp = h_stamp_hkd(notional_hkd, day) * rate_fx
            commission = _commission(
                notional,
                schedule.h_commission,
                schedule.h_commission_min_hkd * rate_fx,
            )
            levies = notional * schedule.h_levy
            spread_rate = H_SPREAD_FALLBACK if half_spread is None or not np.isfinite(half_spread) else float(half_spread)
        spread = notional * max(spread_rate, 0.0)
        if adv_cny is not None and np.isfinite(adv_cny) and adv_cny > 0:
            impact = notional * (schedule.impact_k * math.sqrt(notional / adv_cny)) / 10_000.0
        if charge_fx and leg == "H":
            fx_cost = notional * schedule.fx_cost
    total = stamp + transfer + commission + levies + spread + impact + fx_cost
    return {
        "date": _day(day),
        "leg": leg,
        "side": side,
        "notional": notional,
        "stamp": float(stamp),
        "transfer": float(transfer),
        "commission": float(commission),
        "levies": float(levies),
        "spread": float(spread),
        "impact": float(impact),
        "fx": float(fx_cost),
        "total": float(total),
        "total_bp": float(total / notional * 10_000.0) if notional else 0.0,
    }


def explicit_round_trip_bp(day="2024-06-03", notional: float = 1_000_000.0, fx: float = 0.9) -> float:
    """两腿各进出一次的印花税、过户费、佣金、征费和换汇，占单腿名义的 bps。不含点差和冲击。"""
    schedule = CostSchedule()
    total = 0.0
    for side in ("buy", "sell"):
        row = fill_cost(day, "A", side, notional, fx, schedule, half_spread=0.0, charge_fx=False)
        total += row["stamp"] + row["transfer"] + row["commission"] + row["levies"]
        row_h = fill_cost(day, "H", side, notional, fx, schedule, half_spread=0.0, charge_fx=True)
        total += row_h["stamp"] + row_h["transfer"] + row_h["commission"] + row_h["levies"] + row_h["fx"]
    return float(total / notional * 10_000.0)


def cost_parts_ok(row: dict) -> bool:
    parts = ("stamp", "transfer", "commission", "levies", "spread", "impact", "fx")
    return abs(row["total"] - sum(row[name] for name in parts)) <= 1e-8 * max(1.0, abs(row["total"]))
