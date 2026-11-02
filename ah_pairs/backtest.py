"""z-score 均值回归回测。

默认在信号出现的下一交易日收盘成交。A 股 T+1 下，当日买入最早下一交易日才能卖出；
日频、次日收盘成交自然满足这个限制。H 股虽然可以 T+0，两条腿用同一个成交时点，
避免只对冲了一边。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

import numpy as np
import pandas as pd

from ah_pairs.costs import CostParams, rebalance_cost_parts


@dataclass
class StrategyParams:
    hedge: str = "rolling"
    hedge_window: int = 120
    z_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    kalman_delta: float = 1e-4
    kalman_delta_auto: bool = False
    allow_short_a: bool = False
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


def params_to_json(params: StrategyParams) -> str:
    return json.dumps(asdict(params), sort_keys=True)


def params_from_json(payload: str) -> StrategyParams:
    raw = json.loads(payload)
    costs = CostParams(**raw.pop("costs"))
    return StrategyParams(costs=costs, **raw)


def signal_token(params: StrategyParams) -> str:
    """只包含会改变 β 和 z 的字段。开仓阈值和成本不在这里。"""
    payload = {
        "hedge": params.hedge,
        "hedge_window": int(params.hedge_window),
        "z_window": int(params.z_window),
        "kalman_delta": float(params.kalman_delta),
        "kalman_delta_auto": bool(params.kalman_delta_auto),
    }
    return json.dumps(payload, sort_keys=True)


def params_from_signal_token(payload: str) -> StrategyParams:
    raw = json.loads(payload)
    return StrategyParams(
        hedge=raw["hedge"],
        hedge_window=int(raw["hedge_window"]),
        z_window=int(raw["z_window"]),
        kalman_delta=float(raw["kalman_delta"]),
        kalman_delta_auto=bool(raw["kalman_delta_auto"]),
        costs=CostParams(),
    )


def trade_token(params: StrategyParams) -> str:
    """开平仓、成交时点和成本。β 已经算好之后，只改这些不必重估协整。"""
    payload = {
        "entry_z": float(params.entry_z),
        "exit_z": float(params.exit_z),
        "stop_z": float(params.stop_z),
        "allow_short_a": bool(params.allow_short_a),
        "require_positive_beta": bool(params.require_positive_beta),
        "fill": params.fill,
        "beta_band": float(params.beta_band),
        "costs": asdict(params.costs),
    }
    return json.dumps(payload, sort_keys=True)


def params_from_trade_token(payload: str) -> StrategyParams:
    raw = json.loads(payload)
    costs = CostParams(**raw.pop("costs"))
    return StrategyParams(costs=costs, **raw)


def sample_clock(index) -> dict:
    """年数用首尾日历跨度，不用「行数 / 252」。

    并集日历一年大约 260 多行。若仍除以 252，年数会被抬高，年化收益和换手会被压低。
    """
    idx = pd.DatetimeIndex(index)
    n = len(idx)
    if n < 2:
        return {"years": np.nan, "bars_per_year": 252.0, "span_days": 0}
    span_days = max(int((idx.max().normalize() - idx.min().normalize()).days), 1)
    years = span_days / 365.25
    return {
        "years": float(years),
        "bars_per_year": float(n / years),
        "span_days": int(span_days),
    }


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
    tradable: np.ndarray | None = None,
) -> np.ndarray:
    """每个时点收盘后想持有的方向。还没有推迟到成交日。

    tradable 为假的日子不重新读 z，方向保持上一个可交易日的决定。
    """
    n = len(z)
    target = np.zeros(n, dtype=int)
    held = 0
    can_trade = np.ones(n, dtype=bool) if tradable is None else np.asarray(tradable, dtype=bool)
    for t in range(n):
        if not can_trade[t]:
            target[t] = held
            continue
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
    direction: np.ndarray,
    beta: np.ndarray,
    beta_band: float,
    tradable: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """方向不变时锁定开仓 beta。偏离超过 beta_band 才调整对冲比率。单边开市日不调仓。"""
    n = len(direction)
    w_a = np.zeros(n)
    w_h = np.zeros(n)
    held_a = held_h = 0.0
    held_beta = np.nan
    prev_dir = 0
    can_trade = np.ones(n, dtype=bool) if tradable is None else np.asarray(tradable, dtype=bool)
    for t in range(n):
        if not can_trade[t]:
            w_a[t] = held_a
            w_h[t] = held_h
            continue
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


def _executed_direction(
    target: np.ndarray,
    fill: str,
    tradable: np.ndarray | None = None,
) -> np.ndarray:
    """成交只落在可交易日。next_close 等到下一个两边都开市的收盘，不在单边日成交。"""
    n = len(target)
    executed = np.zeros(n, dtype=int)
    can_trade = np.ones(n, dtype=bool) if tradable is None else np.asarray(tradable, dtype=bool)
    if fill == "same_close":
        held = 0
        for t in range(n):
            if can_trade[t]:
                held = int(target[t])
            executed[t] = held
        return executed
    pending = 0
    held = 0
    for t in range(n):
        if not can_trade[t]:
            executed[t] = held
            continue
        held = pending
        executed[t] = held
        pending = int(target[t])
    return executed


def _tradable(frame: pd.DataFrame) -> np.ndarray:
    if "both_open" not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    return frame["both_open"].fillna(False).to_numpy(dtype=bool)


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

    tradable = _tradable(work)
    target = positions_from_z(z, beta, params, tradable=tradable)
    direction = _executed_direction(target, params.fill, tradable=tradable)
    for t in range(n):
        if not tradable[t]:
            continue
        use_beta = beta[t]
        if params.require_positive_beta and not (np.isfinite(use_beta) and use_beta > 0):
            direction[t] = 0
    w_a, w_h = _held_weights(direction, beta, params.beta_band, tradable)

    pnl = np.zeros(n)
    cost = np.zeros(n)
    borrow = np.zeros(n)
    gross = np.zeros(n)
    part_names = ("a_commission", "a_stamp", "h_commission", "h_stamp", "slippage")
    part_arr = {name: np.zeros(n) for name in part_names}
    cost_rebalance = np.zeros(n)
    cost_turnover = np.zeros(n)
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
        part = rebalance_cost_parts(prev_a, prev_h, w_a[t], w_h[t], dates[t], params.costs)
        cost_t = float(sum(part.values()))
        for name in part_names:
            part_arr[name][t] = part[name]
        same_direction = t > 0 and int(direction[t]) == int(direction[t - 1]) and int(direction[t]) != 0
        if same_direction:
            cost_rebalance[t] = cost_t
        else:
            cost_turnover[t] = cost_t
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
    for name in part_names:
        work[f"cost_{name}"] = part_arr[name]
    work["cost_rebalance"] = cost_rebalance
    work["cost_turnover"] = cost_turnover
    work["ret_strat"] = pnl
    work["equity"] = equity
    work["drawdown"] = drawdown

    trades = _extract_trades(work, params.fill)
    stats = _performance_stats(work, trades, missing, ruined)
    return work, trades, stats


def current_state(frame: pd.DataFrame, trades: pd.DataFrame, fill: str) -> dict:
    """最后一根 K 线收盘时的 z、持仓，以及 T+1 下有没有还没成交的信号。"""
    held = int(frame["position"].iloc[-1]) if len(frame) else 0
    signal = int(frame["target"].iloc[-1]) if len(frame) and "target" in frame.columns else held
    z_value = float(frame["z"].iloc[-1]) if len(frame) else np.nan
    pending = fill == "next_close" and signal != held
    last_entry = None
    last_exit = None
    if trades is not None and len(trades) and "entry_date" in trades.columns:
        last_entry = pd.Timestamp(pd.to_datetime(trades["entry_date"]).max())
        if "closed" in trades.columns:
            closed = trades[trades["closed"].astype(bool)]
            if len(closed) and "exit_date" in closed.columns:
                last_exit = pd.Timestamp(pd.to_datetime(closed["exit_date"]).max())
    return {
        "asof": frame.index.max() if len(frame) else None,
        "z": z_value,
        "position": held,
        "signal": signal,
        "pending": pending,
        "last_entry": last_entry,
        "last_exit": last_exit,
    }


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
    clock = sample_clock(frame.index)
    years = clock["years"]
    ann = clock["bars_per_year"] if np.isfinite(clock["bars_per_year"]) and clock["bars_per_year"] > 0 else 252.0
    end_equity = float(equity[-1]) if n else np.nan
    if n > 1 and end_equity > 0 and np.isfinite(years) and years > 0:
        cagr = float(end_equity ** (1.0 / years) - 1.0)
    else:
        cagr = np.nan
    vol = float(np.std(rets, ddof=1)) if len(rets) > 2 else np.nan
    sharpe = float(np.mean(rets) / vol * np.sqrt(ann)) if vol and vol > 0 else np.nan
    sharpe_common = sharpe
    if "both_open" in frame.columns and np.isfinite(years) and years > 0:
        mask = frame["both_open"].fillna(False).to_numpy(dtype=bool)
        common_rets = frame["ret_strat"].to_numpy(dtype=float)[mask]
        common_rets = common_rets[np.isfinite(common_rets)]
        if len(common_rets) > 2:
            common_vol = float(np.std(common_rets, ddof=1))
            common_ann = len(common_rets) / float(years)
            if common_vol > 0 and common_ann > 0:
                sharpe_common = float(np.mean(common_rets) / common_vol * np.sqrt(common_ann))
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
        "vol": float(vol * np.sqrt(ann)) if np.isfinite(vol) else np.nan,
        "sharpe": sharpe,
        "sharpe_common": sharpe_common,
        "bars_per_year": float(ann),
        "span_days": int(clock["span_days"]),
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
        "cost_a_commission": float(frame["cost_a_commission"].sum()) if "cost_a_commission" in frame.columns else 0.0,
        "cost_a_stamp": float(frame["cost_a_stamp"].sum()) if "cost_a_stamp" in frame.columns else 0.0,
        "cost_h_commission": float(frame["cost_h_commission"].sum()) if "cost_h_commission" in frame.columns else 0.0,
        "cost_h_stamp": float(frame["cost_h_stamp"].sum()) if "cost_h_stamp" in frame.columns else 0.0,
        "cost_slippage": float(frame["cost_slippage"].sum()) if "cost_slippage" in frame.columns else 0.0,
        "cost_turnover": float(frame["cost_turnover"].sum()) if "cost_turnover" in frame.columns else 0.0,
        "cost_rebalance": float(frame["cost_rebalance"].sum()) if "cost_rebalance" in frame.columns else 0.0,
        "missing_returns": int(missing_returns),
        "ruined": bool(ruined),
        "n_common": int(frame["both_open"].fillna(False).sum()) if "both_open" in frame.columns else int(n),
        "n_one_sided": int((~frame["both_open"].fillna(False)).sum()) if "both_open" in frame.columns else 0,
        "start": frame.index.min(),
        "end": frame.index.max(),
    }


def later_window_performance(frame: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """后半段沿用已经算好的收益，不重新选择参数。"""
    if "ret_strat" not in frame.columns or len(frame) < 40:
        return {"ok": False}
    part = frame.iloc[len(frame) // 2 :].copy()
    rets = part["ret_strat"].fillna(0.0)
    equity = (1.0 + rets).cumprod()
    end_equity = float(equity.iloc[-1])
    years = sample_clock(part.index)["years"]
    cagr = float(end_equity ** (1.0 / years) - 1.0) if np.isfinite(years) and years > 0 and end_equity > 0 else np.nan
    start = part.index.min()
    closed = trades[trades["closed"]] if len(trades) else trades
    if len(closed):
        closed = closed[pd.to_datetime(closed["entry_date"]) >= start]
    return {
        "ok": True,
        "start": start,
        "end": part.index.max(),
        "n_days": int(len(part)),
        "total_return": float(end_equity - 1.0),
        "cagr": cagr,
        "n_trades": int(len(closed)),
    }


def contrast_costs(frame: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """同一条信号下，当前成本、零成本和滑点再加 10 个基点。"""
    from dataclasses import replace

    zero = replace(
        params.costs,
        a_commission=0.0,
        h_commission=0.0,
        a_borrow_annual=0.0,
        h_borrow_annual=0.0,
        use_statutory_stamp=False,
        a_stamp_sell=0.0,
        h_stamp=0.0,
        slippage=0.0,
    )
    slip = replace(params.costs, slippage=float(params.costs.slippage) + 0.001)
    rows = []
    cases = (
        ("current", "当前成本", params.costs),
        ("zero", "不计成本", zero),
        ("slip10", "滑点再加 10 bps", slip),
    )
    for code, name, costs in cases:
        _work, _trades, stats = run_backtest(frame, replace(params, costs=costs))
        rows.append(
            {
                "code": code,
                "情形": name,
                "净收益": stats["total_return"],
                "夏普": stats["sharpe"],
                "交易次数": stats["n_trades"],
                "成本合计": stats["total_cost"],
                "融券费合计": stats["total_borrow"],
            }
        )
    return pd.DataFrame(rows)
