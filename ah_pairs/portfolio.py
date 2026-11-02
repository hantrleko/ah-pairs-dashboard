"""多对策略收益的等风险组合。

t 日的权重只用 t 之前的滚动波动，缺波动的日子权重为 0。
这不是无风险组合，只是把若干对的历史日收益按波动倒数加在一起。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_vol_weights(returns: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    frame = returns.apply(pd.to_numeric, errors="coerce")
    vol = frame.rolling(lookback, min_periods=lookback).std(ddof=1).shift(1)
    inv = 1.0 / vol.where(vol > 1e-8)
    inv = inv.where(np.isfinite(inv), 0.0).fillna(0.0)
    gross = inv.sum(axis=1)
    weights = inv.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)
    return weights


def portfolio_returns(returns: pd.DataFrame, lookback: int = 60) -> pd.Series:
    clean = returns.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    weights = inverse_vol_weights(clean, lookback=lookback)
    out = (weights * clean).sum(axis=1)
    out.name = "equal_risk"
    # 还没有任何一对走出波动窗口时，权重是 0，收益也记 0。调用方可以按权重大于 0 再截取。
    return out


def portfolio_summary(returns: pd.DataFrame, lookback: int = 60) -> dict:
    if returns is None or returns.empty or returns.shape[1] == 0:
        return {"ok": False, "message": "没有可用于组合的日收益。"}
    series = portfolio_returns(returns, lookback=lookback)
    active = series.loc[inverse_vol_weights(returns.fillna(0.0), lookback).sum(axis=1) > 0]
    if len(active) < 30:
        return {"ok": False, "message": "有效重叠不足 30 天，没有计算等风险组合。"}
    from ah_pairs.backtest import sample_clock

    equity = (1.0 + active).cumprod()
    end_equity = float(equity.iloc[-1])
    clock = sample_clock(active.index)
    years = clock["years"]
    ann = clock["bars_per_year"] if np.isfinite(clock["bars_per_year"]) else 252.0
    vol = float(active.std(ddof=1))
    sharpe = float(active.mean() / vol * np.sqrt(ann)) if vol > 0 else np.nan
    cagr = float(end_equity ** (1.0 / years) - 1.0) if years > 0 and end_equity > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    return {
        "ok": True,
        "n_pairs": int(returns.shape[1]),
        "n_days": int(len(active)),
        "start": active.index.min(),
        "end": active.index.max(),
        "total_return": float(end_equity - 1.0),
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "equity": equity,
        "returns": active,
    }
