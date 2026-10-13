"""z-score 均值回归回测。

默认在信号出现的下一交易日收盘成交。A 股 T+1 下，当日买入最早下一交易日才能卖出；
日频、次日收盘成交自然满足这个限制。H 股虽然可以 T+0，两条腿用同一个成交时点，
避免只对冲了一边。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ah_pairs.costs import CostParams, rebalance_cost


@dataclass
class StrategyParams:
    hedge: str = "rolling"
    hedge_window: int = 120
    z_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    kalman_delta: float = 1e-4
    allow_short_a: bool = True
    require_positive_beta: bool = True
    fill: str = "next_close"
    beta_band: float = 0.10
    costs: CostParams = field(default_factory=CostParams)

    def __post_init__(self) -> None:
        if self.hedge not in {"rolling", "kalman"}:
            raise ValueError("hedge 只能是 rolling 或 kalman")
        if self.fill not in {"next_close", "same_close"}:
            raise ValueError("fill 只能是 next_close 或 same_close")
        if self.hedge_window < 20 or self.z_window < 10:
            raise ValueError("估计窗口至少 20 日，z 窗口至少 10 日")
        if self.entry_z <= 0 or self.exit_z < 0 or self.exit_z >= self.entry_z:
            raise ValueError("需要满足 0 <= 平仓阈值 < 开仓阈值")
        if self.stop_z != 0 and self.stop_z <= self.entry_z:
            raise ValueError("止损阈值需要大于开仓阈值；0 表示关闭止损")
        if self.kalman_delta <= 0:
            raise ValueError("卡尔曼状态噪声系数需要为正")
        if self.beta_band < 0:
            raise ValueError("对冲再平衡阈值不能为负")


def update_position(
    position: int,
    z: float,
    entry: float,
    exit_: float,
    stop: float,
    allow_short_a: bool,
) -> int:
    """+1 做多价差（多 A、空 H），-1 做空价差（空 A、多 H），0 空仓。"""
    if not np.isfinite(z):
        return int(position)
    if position == 0:
        if z >= entry and allow_short_a:
            return -1
        if z <= -entry:
            return 1
        return 0
    if position == 1:
        if stop > 0 and z <= -stop:
            return 0
        if z >= -exit_:
            return 0
        return 1
    if stop > 0 and z >= stop:
        return 0
    if z <= exit_:
        return 0
    return -1


def target_weights(direction: float, beta: float) -> tuple[float, float]:
    """按 |w_A| + |w_H| = 1 归一。H 的相对仓位仍是回归 beta。"""
    if direction == 0 or not np.isfinite(beta):
        return 0.0, 0.0
    gross = 1.0 + abs(float(beta))
    w_a = float(direction) / gross
    w_h = -float(direction) * float(beta) / gross
    return w_a, w_h


def positions_from_z(
    z: np.ndarray,
    beta: np.ndarray,
    params: StrategyParams,
) -> np.ndarray:
    """每个时点收盘后想持有的方向。还没有推迟到成交日。"""
    n = len(z)
    target = np.zeros(n, dtype=int)
    held = 0
    for t in range(n):
        held = update_position(
            held,
            float(z[t]),
            params.entry_z,
            params.exit_z,
            params.stop_z,
            params.allow_short_a,
        )
        if params.require_positive_beta and not (
            np.isfinite(beta[t]) and beta[t] > 0
        ):
            held = 0
        target[t] = held
    return target


def _held_weights(
    direction: np.ndarray, beta: np.ndarray, beta_band: float
) -> tuple[np.ndarray, np.ndarray]:
    """方向不变时锁定开仓 beta。偏离超过 beta_band 才调整对冲比率。"""
    n = len(direction)
    w_a = np.zeros(n)
    w_h = np.zeros(n)
    held_a = held_h = 0.0
    held_beta = np.nan
    prev_dir = 0
    for t in range(n):
        dir_t = int(direction[t])
        b = float(beta[t])
        if dir_t == 0 or not np.isfinite(b) or b <= 0:
            held_a, held_h, held_beta = 0.0, 0.0, np.nan
        elif dir_t != prev_dir or not np.isfinite(held_beta):
            held_a, held_h = target_weights(dir_t, b)
            held_beta = b
        elif abs(b / held_beta - 1.0) > beta_band:
            held_a, held_h = target_weights(dir_t, b)
            held_beta = b
        w_a[t] = held_a
        w_h[t] = held_h
        prev_dir = dir_t
    return w_a, w_h


def _executed_direction(target: np.ndarray, fill: str) -> np.ndarray:
    n = len(target)
    executed = np.zeros(n, dtype=int)
    if fill == "same_close":
        executed[:] = target
        return executed
    if n > 1:
        executed[1:] = target[:-1]
    return executed


def run_backtest(
    frame: pd.DataFrame,
    params: StrategyParams,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """在已经算好 z 和 beta 的表上回测。

    frame 需要列：z, beta, ret_a, ret_h。索引是交易日。
    ret 是当天收盘相对前一日收盘的简单收益。
    """
    work = frame.copy()
    z = work["z"].to_numpy(dtype=float)
    beta = work["beta"].to_numpy(dtype=float)
    ret_a = work["ret_a"].to_numpy(dtype=float)
    ret_h = work["ret_h"].to_numpy(dtype=float)
    dates = work.index
    n = len(work)

    target = positions_from_z(z, beta, params)
    direction = _executed_direction(target, params.fill)
    for t in range(n):
        use_beta = beta[t]
        if params.require_positive_beta and not (np.isfinite(use_beta) and use_beta > 0):
            direction[t] = 0
    w_a, w_h = _held_weights(direction, beta, params.beta_band)

    pnl = np.zeros(n)
    cost = np.zeros(n)
    borrow = np.zeros(n)
    gross = np.zeros(n)
    missing = 0
    ruined = False
    day_count = params.costs.trading_days

    for t in range(n):
        if t == 0:
            prev_a = prev_h = 0.0
            gross_t = 0.0
            borrow_t = 0.0
        else:
            prev_a, prev_h = w_a[t - 1], w_h[t - 1]
            ra, rh = ret_a[t], ret_h[t]
            if not (np.isfinite(ra) and np.isfinite(rh)):
                gross_t = 0.0
                missing += 1
            else:
                gross_t = prev_a * ra + prev_h * rh
            borrow_t = 0.0
            if prev_a < 0:
                borrow_t += -prev_a * params.costs.a_borrow_annual / day_count
            if prev_h < 0:
                borrow_t += -prev_h * params.costs.h_borrow_annual / day_count
        cost_t = rebalance_cost(prev_a, prev_h, w_a[t], w_h[t], dates[t], params.costs)
        gross[t] = gross_t
        borrow[t] = borrow_t
        cost[t] = cost_t
        pnl[t] = gross_t - borrow_t - cost_t

    equity = np.ones(n)
    for t in range(n):
        step = 1.0 + pnl[t]
        base = 1.0 if t == 0 else equity[t - 1]
        if step <= 0:
            equity[t] = 0.0
            if t + 1 < n:
                equity[t + 1 :] = 0.0
            ruined = True
            break
        equity[t] = base * step

    peak = np.maximum.accumulate(np.where(np.isfinite(equity), equity, 0.0))
    peak = np.where(peak <= 0, np.nan, peak)
    drawdown = equity / peak - 1.0

    work["target"] = target
    work["position"] = direction
    work["w_a"] = w_a
    work["w_h"] = w_h
    work["ret_gross"] = gross
    work["borrow"] = borrow
    work["cost"] = cost
    work["ret_strat"] = pnl
    work["equity"] = equity
    work["drawdown"] = drawdown

    trades = _extract_trades(work, params.fill)
    stats = _performance_stats(work, trades, missing, ruined)
    return work, trades, stats


def _extract_trades(frame: pd.DataFrame, fill: str) -> pd.DataFrame:
    position = frame["position"].to_numpy(dtype=int)
    equity = frame["equity"].to_numpy(dtype=float)
    z = frame["z"].to_numpy(dtype=float)
    dates = frame.index
    rows = []
    entry = None
    direction = 0
    for t, pos in enumerate(position):
        if entry is None and pos != 0:
            entry = t
            direction = int(pos)
            continue
        if entry is not None and pos != direction:
            rows.append(
                _trade_row(dates, equity, z, entry, t, direction, fill, closed=True)
            )
            entry = None
            direction = 0
            if pos != 0:
                entry = t
                direction = int(pos)
    if entry is not None:
        rows.append(
            _trade_row(
                dates,
                equity,
                z,
                entry,
                len(position) - 1,
                direction,
                fill,
                closed=False,
            )
        )
    columns = [
        "entry_date",
        "exit_date",
        "direction",
        "entry_z",
        "exit_z",
        "holding_days",
        "trade_return",
        "closed",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def _signal_index(index: int, fill: str) -> int:
    if fill == "next_close":
        return max(index - 1, 0)
    return index


def _trade_row(dates, equity, z, entry, exit_, direction, fill: str, closed: bool) -> dict:
    base = 1.0 if entry == 0 else float(equity[entry - 1])
    end_equity = float(equity[exit_])
    trade_return = np.nan if base <= 0 else end_equity / base - 1.0
    z_entry = z[_signal_index(entry, fill)]
    z_exit = z[_signal_index(exit_, fill)]
    return {
        "entry_date": dates[entry],
        "exit_date": dates[exit_],
        "direction": int(direction),
        "entry_z": float(z_entry) if np.isfinite(z_entry) else np.nan,
        "exit_z": float(z_exit) if np.isfinite(z_exit) else np.nan,
        "holding_days": int(exit_ - entry),
        "trade_return": float(trade_return) if np.isfinite(trade_return) else np.nan,
        "closed": bool(closed),
    }


def _performance_stats(
    frame: pd.DataFrame,
    trades: pd.DataFrame,
    missing_returns: int,
    ruined: bool,
) -> dict:
    rets = frame["ret_strat"].to_numpy(dtype=float)
    rets = rets[np.isfinite(rets)]
    equity = frame["equity"].to_numpy(dtype=float)
    n = len(frame)
    years = n / 252.0 if n else np.nan
    end_equity = float(equity[-1]) if n else np.nan
    if n > 1 and end_equity > 0 and years > 0:
        cagr = float(end_equity ** (1.0 / years) - 1.0)
    else:
        cagr = np.nan
    vol = float(np.std(rets, ddof=1)) if len(rets) > 2 else np.nan
    sharpe = float(np.mean(rets) / vol * np.sqrt(252.0)) if vol and vol > 0 else np.nan
    max_dd = float(np.nanmin(frame["drawdown"])) if n else np.nan
    dw = frame["w_a"].diff().abs().fillna(frame["w_a"].abs())
    dw = dw + frame["w_h"].diff().abs().fillna(frame["w_h"].abs())
    turnover = float(dw.sum() / years) if years and years > 0 else np.nan
    closed = trades[trades["closed"]] if len(trades) else trades
    if len(closed):
        wins = closed["trade_return"] > 0
        win_rate = float(wins.mean())
        avg_trade = float(closed["trade_return"].mean())
        avg_hold = float(closed["holding_days"].mean())
    else:
        win_rate = avg_trade = avg_hold = np.nan
    return {
        "n_days": int(n),
        "years": float(years) if np.isfinite(years) else np.nan,
        "total_return": float(end_equity - 1.0) if np.isfinite(end_equity) else np.nan,
        "cagr": cagr,
        "vol": float(vol * np.sqrt(252.0)) if np.isfinite(vol) else np.nan,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if np.isfinite(cagr) and np.isfinite(max_dd) and max_dd < 0 else np.nan,
        "turnover": turnover,
        "n_trades": int(len(closed)),
        "n_open": int((~trades["closed"]).sum()) if len(trades) else 0,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "avg_holding_days": avg_hold,
        "exposure": float((frame["position"] != 0).mean()) if n else np.nan,
        "total_cost": float(frame["cost"].sum()),
        "total_borrow": float(frame["borrow"].sum()),
        "missing_returns": int(missing_returns),
        "ruined": bool(ruined),
        "start": frame.index.min(),
        "end": frame.index.max(),
    }
