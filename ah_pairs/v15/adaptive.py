"""自适应闸门和次日成交。

半衰期不在 10 到 120 日，或折点 EG p 大于 0.10，当天只许平仓。
信号在共同开市日收盘产生，两腿一起顺延到下一个共同开市日收盘。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ah_pairs.v15.costs import CostSchedule, fill_cost

STOP_Z = 3.5
BOOK_NOTIONAL = 1_000_000.0


def ar1_phi(resid) -> float:
    values = np.asarray(resid, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return np.nan
    centered = values - float(np.mean(values))
    left = centered[:-1]
    right = centered[1:]
    denom = float(np.dot(left, left))
    if denom <= 0:
        return np.nan
    return float(np.dot(left, right) / denom)


def half_life_from_phi(phi: float) -> float:
    if not np.isfinite(phi) or phi <= 0.0 or phi >= 1.0:
        return math.inf
    return float(-math.log(2.0) / math.log(phi))


def synthetic_ar1(half_life: float, n: int, seed: int, scale: float = 1.0) -> np.ndarray:
    phi = math.exp(-math.log(2.0) / float(half_life))
    rng = np.random.default_rng(seed)
    shock = rng.normal(0.0, scale, size=int(n)).astype(np.float64)
    out = np.zeros(int(n), dtype=np.float64)
    for i in range(1, int(n)):
        out[i] = phi * out[i - 1] + shock[i]
    return out


def gate_open(half_life: float, eg_p: float | None) -> tuple[bool, str]:
    if not np.isfinite(half_life) or not (10.0 <= float(half_life) <= 120.0):
        return False, "half_life_out_of_range"
    if eg_p is None:
        return True, ""
    if not np.isfinite(eg_p) or float(eg_p) > 0.10:
        return False, "eg_gate_fail"
    return True, ""


def rolling_no_trade_share(spread, window: int = 1200) -> float:
    """逐日半衰期闸门。半衰期落进区间之后，才用 ADF 作为第二道闸门。"""
    values = np.asarray(spread, dtype=np.float64)
    span = int(window)
    checked = 0
    blocked = 0
    for end in range(span, len(values)):
        checked += 1
        sample = values[end - span : end]
        life = half_life_from_phi(ar1_phi(sample))
        if not (10.0 <= life <= 120.0):
            blocked += 1
            continue
        if _eg_pvalue(sample) > 0.10:
            blocked += 1
    if checked == 0:
        return np.nan
    return float(blocked / checked)


def _eg_pvalue(sample: np.ndarray) -> float:
    """单序列过不了 Engle-Granger。这里用残差对常数的 ADF p 值作闸门的第二道。"""
    from statsmodels.tsa.stattools import adfuller

    try:
        return float(adfuller(sample, maxlag=1, autolag=None)[1])
    except Exception:
        return 1.0


def latest_half_life(log_a, log_h, window: int = 120) -> float:
    left = np.asarray(log_a, dtype=np.float64)
    right = np.asarray(log_h, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    span = int(window)
    if len(left) < span:
        return math.inf
    y = left[-span:]
    x = right[-span:]
    var_x = float(np.var(x))
    beta = float(np.cov(y, x, ddof=0)[0, 1] / var_x) if var_x > 0 else 1.0
    return half_life_from_phi(ar1_phi(y - beta * x))


def half_life_tail(log_a, log_h, window: int = 120, tail: int = 90) -> np.ndarray:
    left = np.asarray(log_a, dtype=np.float64)
    right = np.asarray(log_h, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    span = int(window)
    if len(left) < span:
        return np.asarray([], dtype=np.float64)
    start = max(span, len(left) - int(tail) + 1)
    out = np.empty(len(left) - start + 1, dtype=np.float64)
    for i, end in enumerate(range(start, len(left) + 1)):
        y = left[end - span : end]
        x = right[end - span : end]
        var_x = float(np.var(x))
        beta = float(np.cov(y, x, ddof=0)[0, 1] / var_x) if var_x > 0 else 1.0
        out[i] = half_life_from_phi(ar1_phi(y - beta * x))
    return out


def pair_no_trade_share(log_a, log_h, window: int = 120) -> float:
    """两条价格都在时，用窗口 OLS 残差的半衰期，必要时再算 Engle-Granger。"""
    from statsmodels.tsa.stattools import coint

    left = np.asarray(log_a, dtype=np.float64)
    right = np.asarray(log_h, dtype=np.float64)
    span = int(window)
    checked = 0
    blocked = 0
    for end in range(span, len(left)):
        checked += 1
        y = left[end - span : end]
        x = right[end - span : end]
        var_x = float(np.var(x))
        beta = float(np.cov(y, x, ddof=0)[0, 1] / var_x) if var_x > 0 else 1.0
        resid = y - beta * x
        life = half_life_from_phi(ar1_phi(resid))
        opened, _reason = gate_open(life, 1.0)
        if not opened:
            blocked += 1
            continue
        try:
            p_value = float(coint(y, x, trend="c", autolag=None, maxlag=1)[1])
        except Exception:
            p_value = 1.0
        if p_value > 0.10:
            blocked += 1
    if checked == 0:
        return np.nan
    return float(blocked / checked)


def execute(
    frame: pd.DataFrame,
    z: np.ndarray,
    z_in: float,
    z_out: float,
    max_hold: int,
    allow_short_a: bool = False,
    gate: np.ndarray | None = None,
    schedule: CostSchedule | None = None,
) -> dict:
    """持仓用前一日权重赚当日收益。单边日不成交。"""
    schedule = schedule or CostSchedule()
    n = len(frame)
    signal = np.asarray(z, dtype=np.float64)
    both = frame["both_open"].fillna(False).to_numpy(dtype=bool) if "both_open" in frame.columns else np.ones(n, dtype=bool)
    ret_a = frame["ret_a"].to_numpy(dtype=np.float64) if "ret_a" in frame.columns else np.zeros(n)
    ret_h = frame["ret_h"].to_numpy(dtype=np.float64) if "ret_h" in frame.columns else np.zeros(n)
    dates = pd.DatetimeIndex(frame.index)
    allowed = np.ones(n, dtype=bool) if gate is None else np.asarray(gate, dtype=bool)
    position = np.zeros(n, dtype=np.int64)
    cost = np.zeros(n, dtype=np.float64)
    gross = np.zeros(n, dtype=np.float64)
    rows: list[dict] = []
    held = 0
    pending = 0
    age = 0
    blocked = 0
    signals = 0
    stop_exit = 0
    for t in range(n):
        prev = held
        if both[t]:
            held = pending
        if both[t] and prev != 0:
            age += 1
        if held == 0:
            age = 0
        position[t] = held
        w_a = 0.5 * held
        w_h = -0.5 * held
        ra = ret_a[t] if np.isfinite(ret_a[t]) else 0.0
        rh = ret_h[t] if np.isfinite(ret_h[t]) else 0.0
        gross[t] = (0.5 * prev) * ra + (-0.5 * prev) * rh
        if both[t] and held != prev:
            traded = abs(held - prev)
            fx = float(frame["fx"].iloc[t]) if "fx" in frame.columns else 0.9
            for _step in range(int(traded)):
                a_side = "buy" if held > prev else "sell"
                h_side = "sell" if held > prev else "buy"
                for leg, side in (("A", a_side), ("H", h_side)):
                    row = fill_cost(
                        dates[t],
                        leg,
                        side,
                        0.5 * BOOK_NOTIONAL,
                        fx,
                        schedule,
                        half_spread=0.0,
                        charge_fx=(leg == "H"),
                    )
                    rows.append(row)
                    cost[t] += row["total"] / BOOK_NOTIONAL
        if not both[t]:
            continue
        z_now = signal[t]
        desired = held
        if np.isfinite(z_now) and held != 0 and abs(z_now) >= STOP_Z:
            desired = 0
            stop_exit += 1
        elif np.isfinite(z_now) and held != 0 and age >= int(max_hold):
            desired = 0
        elif np.isfinite(z_now) and held != 0 and abs(z_now) <= float(z_out):
            desired = 0
        elif held == 0 and np.isfinite(z_now) and allowed[t]:
            if z_now <= -float(z_in):
                desired = 1
            elif z_now >= float(z_in):
                signals += 1
                if allow_short_a:
                    desired = -1
                else:
                    blocked += 1
                    desired = 0
        pending = desired
    net = gross - cost
    return {
        "position": position,
        "gross": gross,
        "cost": cost,
        "net": net,
        "rows": rows,
        "n_blocked": int(blocked),
        "n_short_a_signals": int(signals),
        "n_stop": int(stop_exit),
        "blocked_share": float(blocked / signals) if signals else np.nan,
    }
