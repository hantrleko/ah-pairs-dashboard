"""已经决定持有这家公司时的三种做法：永远持 A、永远持 H、按溢价切换。

不做空。切换规则只用当天已经知道的 z：z < 0 持有 A，z ≥ 0 持有 H。
成交和主回测一样，要等到下一个两边都开市的收盘，当天涨跌仍算旧仓位。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ah_pairs.backtest import sample_clock
from ah_pairs.costs import CostParams, rebalance_cost


def _summarize(index, returns: np.ndarray) -> dict:
    series = pd.Series(np.asarray(returns, dtype=float), index=index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = np.ones(len(series))
    for i, step in enumerate(series.to_numpy(dtype=float)):
        nxt = (1.0 if i == 0 else equity[i - 1]) * (1.0 + step)
        if nxt <= 0:
            equity[i:] = 0.0
            break
        equity[i] = nxt
    clock = sample_clock(index)
    years = clock["years"]
    ann = clock["bars_per_year"] if np.isfinite(clock["bars_per_year"]) and clock["bars_per_year"] > 0 else 252.0
    end_equity = float(equity[-1]) if len(equity) else np.nan
    cagr = float(end_equity ** (1.0 / years) - 1.0) if years and years > 0 and end_equity > 0 else np.nan
    vol = float(series.std(ddof=1)) if len(series) > 2 else np.nan
    sharpe = float(series.mean() / vol * np.sqrt(ann)) if vol and vol > 0 else np.nan
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.where(peak <= 0, np.nan, peak) - 1.0
    return {
        "total_return": float(end_equity - 1.0) if np.isfinite(end_equity) else np.nan,
        "sharpe": sharpe,
        "max_drawdown": float(np.nanmin(drawdown)) if len(drawdown) else np.nan,
        "cagr": cagr,
        "total_cost": float("nan"),
        "switches": 0,
    }


def _end_weights(want_a: np.ndarray, tradable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """收盘后的权重。want_a 为 1 持 A、0 持 H，nan 表示维持上一次决定。"""
    n = len(want_a)
    end_a = np.zeros(n)
    end_h = np.zeros(n)
    pending_a = 0.0
    pending_h = 0.0
    held_a = 0.0
    held_h = 0.0
    decided = np.nan
    for t in range(n):
        if not tradable[t]:
            end_a[t] = held_a
            end_h[t] = held_h
            continue
        held_a, held_h = pending_a, pending_h
        end_a[t] = held_a
        end_h[t] = held_h
        if np.isfinite(want_a[t]):
            decided = float(want_a[t])
        if np.isfinite(decided):
            pending_a = decided
            pending_h = 1.0 - decided
    return end_a, end_h


def _path(frame: pd.DataFrame, want_a: np.ndarray, costs: CostParams) -> dict:
    tradable = frame["both_open"].fillna(False).to_numpy(dtype=bool) if "both_open" in frame.columns else np.ones(len(frame), dtype=bool)
    end_a, end_h = _end_weights(want_a, tradable)
    ret_a = frame["ret_a"].to_numpy(dtype=float)
    ret_h = frame["ret_h"].to_numpy(dtype=float)
    pnl = np.zeros(len(frame))
    spent = 0.0
    switches = 0
    for t in range(len(frame)):
        prev_a = 0.0 if t == 0 else end_a[t - 1]
        prev_h = 0.0 if t == 0 else end_h[t - 1]
        if t > 0 and np.isfinite(ret_a[t]) and np.isfinite(ret_h[t]):
            gross = prev_a * ret_a[t] + prev_h * ret_h[t]
        else:
            gross = 0.0
        cost = rebalance_cost(prev_a, prev_h, end_a[t], end_h[t], frame.index[t], costs)
        if abs(end_a[t] - prev_a) + abs(end_h[t] - prev_h) > 1e-12:
            switches += 1
        spent += cost
        pnl[t] = gross - cost
    out = _summarize(frame.index, pnl)
    out["total_cost"] = float(spent)
    out["switches"] = int(switches)
    return out


def holding_comparison(frame: pd.DataFrame, costs: CostParams | None = None) -> pd.DataFrame:
    """三行比较。列是代号，页面再翻译。"""
    costs = costs or CostParams()
    n = len(frame)
    if n < 5 or "ret_a" not in frame.columns or "ret_h" not in frame.columns:
        return pd.DataFrame()
    z = frame["z"].to_numpy(dtype=float) if "z" in frame.columns else np.full(n, np.nan)
    want = np.full(n, np.nan)
    for t in range(n):
        if np.isfinite(z[t]):
            want[t] = 1.0 if z[t] < 0 else 0.0
    rows = [
        {"code": "always_a", **_path(frame, np.ones(n), costs)},
        {"code": "always_h", **_path(frame, np.zeros(n), costs)},
        {"code": "switch", **_path(frame, want, costs)},
    ]
    return pd.DataFrame(rows)
