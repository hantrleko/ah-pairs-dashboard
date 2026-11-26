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


def new_exec_state() -> dict:
    """逐日推进的执行状态。空仓、无挂单、exit_z 在开仓时锁定。"""
    return {"held": 0, "pending": 0, "age": 0, "exit_z": None}


def exec_step(
    state: dict,
    *,
    date,
    both_open: bool,
    ret_a: float = 0.0,
    ret_h: float = 0.0,
    fx: float = 0.9,
    z_now: float = np.nan,
    allowed: bool = True,
    z_in: float = 2.0,
    z_out: float = 0.5,
    max_hold: int = 60,
    allow_short_a: bool = False,
    stop_z: float = STOP_Z,
    schedule: CostSchedule | None = None,
) -> dict:
    """推进一天。持仓用前一日权重赚当日收益；单边日不成交、不读 z。

    出场阈值用 state 里开仓时锁定的 exit_z；空仓时没有出场决策。
    返回当天的持仓、收益、成交明细和机器可读的原因代码。
    """
    schedule = schedule or CostSchedule()
    prev = int(state["held"])
    held = int(state["held"])
    pending = int(state["pending"])
    age = int(state["age"])
    if both_open:
        held = pending
    if both_open and prev != 0:
        age += 1
    if held == 0:
        age = 0
    ra = float(ret_a) if np.isfinite(ret_a) else 0.0
    rh = float(ret_h) if np.isfinite(ret_h) else 0.0
    gross = (0.5 * prev) * ra + (-0.5 * prev) * rh
    cost = 0.0
    rows: list[dict] = []
    if both_open and held != prev:
        traded = abs(held - prev)
        for _step in range(int(traded)):
            a_side = "buy" if held > prev else "sell"
            h_side = "sell" if held > prev else "buy"
            for leg, side in (("A", a_side), ("H", h_side)):
                row = fill_cost(
                    date,
                    leg,
                    side,
                    0.5 * BOOK_NOTIONAL,
                    fx,
                    schedule,
                    half_spread=0.0,
                    charge_fx=(leg == "H"),
                )
                rows.append(row)
                cost += row["total"] / BOOK_NOTIONAL
        if held != 0 and prev == 0:
            state["exit_z"] = float(z_out)
        elif held == 0:
            state["exit_z"] = None
    state["held"] = held
    state["age"] = age
    blocked = False
    short_signal = False
    stop_hit = False
    if not both_open:
        # 单边日不读 z、不产生新信号，也不覆盖还没成交的挂单。
        reason = "one_sided"
    else:
        exit_z = state["exit_z"] if state["exit_z"] is not None else float(z_out)
        desired = held
        reason = "hold" if held != 0 else "flat"
        if np.isfinite(z_now) and held != 0 and abs(z_now) >= float(stop_z):
            desired = 0
            stop_hit = True
            reason = "exit_stop"
        elif np.isfinite(z_now) and held != 0 and age >= int(max_hold):
            desired = 0
            reason = "exit_max_hold"
        elif np.isfinite(z_now) and held != 0 and abs(z_now) <= float(exit_z):
            desired = 0
            reason = "exit_z_out"
        elif held == 0 and np.isfinite(z_now):
            if abs(z_now) >= float(z_in) and not allowed:
                reason = "gate_blocked"
                blocked = True
            elif allowed:
                if z_now <= -float(z_in):
                    desired = 1
                    reason = "enter_long_spread"
                elif z_now >= float(z_in):
                    short_signal = True
                    if allow_short_a:
                        desired = -1
                        reason = "enter_short_spread"
                    else:
                        blocked = True
                        desired = 0
                        reason = "short_a_blocked"
        state["pending"] = int(desired)
    return {
        "position": int(held),
        "signal": int(state["pending"]),
        "pending": int(state["pending"]) != int(held),
        "gross": float(gross),
        "cost": float(cost),
        "net": float(gross - cost),
        "rows": rows,
        "reason": reason,
        "blocked": bool(blocked),
        "short_signal": bool(short_signal),
        "stop": bool(stop_hit),
        "age": int(age),
    }


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
    state = new_exec_state()
    blocked = 0
    signals = 0
    stop_exit = 0
    for t in range(n):
        day = exec_step(
            state,
            date=dates[t],
            both_open=bool(both[t]),
            ret_a=ret_a[t],
            ret_h=ret_h[t],
            fx=float(frame["fx"].iloc[t]) if "fx" in frame.columns else 0.9,
            z_now=signal[t],
            allowed=bool(allowed[t]),
            z_in=z_in,
            z_out=z_out,
            max_hold=max_hold,
            allow_short_a=allow_short_a,
            stop_z=STOP_Z,
            schedule=schedule,
        )
        position[t] = day["position"]
        cost[t] = day["cost"]
        gross[t] = day["gross"]
        rows.extend(day["rows"])
        if day["reason"] == "short_a_blocked":
            blocked += 1
        if day["short_signal"]:
            signals += 1
        if day["stop"]:
            stop_exit += 1
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
