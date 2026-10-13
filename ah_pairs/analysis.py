"""把行情、协整、对冲和回测串成一次分析。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ah_pairs.backtest import StrategyParams, run_backtest
from ah_pairs.cointegration import (
    engle_granger,
    johansen_test,
    ou_half_life,
    select_vecm_lags,
)
from ah_pairs.hedge import kalman_hedge, rolling_ols, rolling_zscore


def warmup_start(start, params: StrategyParams) -> pd.Timestamp:
    """评估区间之前多取的自然日，用来准备 beta 和 z。"""
    days = int((params.hedge_window + params.z_window) * 2.2) + 20
    return pd.Timestamp(start).normalize() - pd.Timedelta(days=days)


def build_signals(prices: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    y = prices["log_a"].to_numpy(dtype=float)
    x = prices["log_h"].to_numpy(dtype=float)
    if params.hedge == "kalman":
        alpha, beta = kalman_hedge(
            y, x, delta=params.kalman_delta, init_window=params.hedge_window
        )
    else:
        alpha, beta = rolling_ols(y, x, params.hedge_window)
    spread = y - alpha - beta * x
    out = prices.copy()
    out["alpha"] = alpha
    out["beta"] = beta
    out["spread"] = spread
    out["z"] = rolling_zscore(spread, params.z_window)
    out["spread_beta1"] = prices["log_a"] - prices["log_h"]
    return out


def analyze_frame(
    prices: pd.DataFrame,
    params: StrategyParams,
    eval_start=None,
    northbound: pd.Series | None = None,
) -> dict:
    """prices 可以包含评估区间之前的预热样本。

    协整检验和回测净值只使用 eval_start 及之后的日期。
    预热样本只用来生成评估区间第一天就已经可用的 beta 和 z。
    """
    signals = build_signals(prices, params)
    if eval_start is not None:
        start = pd.Timestamp(eval_start).normalize()
        view = signals.loc[signals.index >= start].copy()
    else:
        view = signals.copy()
    if view.empty:
        raise ValueError("评估区间里没有数据")
    usable = int(np.isfinite(view["z"].to_numpy(dtype=float)).sum())
    if usable < 30:
        raise ValueError(
            f"有效 z-score 只有 {usable} 天。把起始日往前移，或缩短估计窗口。"
        )

    eg = engle_granger(view["log_a"], view["log_h"])
    view["static_resid"] = eg["resid"].reindex(view.index)
    try:
        lags = select_vecm_lags(view["log_a"], view["log_h"])
        johansen = johansen_test(view["log_a"], view["log_h"], k_ar_diff=lags)
    except Exception as exc:
        johansen = {"ok": False, "message": str(exc)}
    else:
        johansen["ok"] = True

    static_hl = ou_half_life(eg["resid"])
    traded = view["spread"].replace([np.inf, -np.inf], np.nan).dropna()
    traded_hl = ou_half_life(traded) if len(traded) >= 30 else None

    result, trades, stats = run_backtest(view, params)
    north = None
    if northbound is not None and not northbound.empty:
        from ah_pairs.northbound import describe_northbound

        north = describe_northbound(result["spread"], northbound)
        result["northbound"] = northbound.reindex(result.index)

    return {
        "frame": result,
        "trades": trades,
        "stats": stats,
        "eg": {k: v for k, v in eg.items() if k != "resid"},
        "half_life_static": static_hl,
        "half_life_traded": traded_hl,
        "johansen": johansen,
        "northbound": north,
        "params": params,
    }


def screen_one(prices: pd.DataFrame, params: StrategyParams, eval_start) -> dict:
    result = analyze_frame(prices, params, eval_start=eval_start)
    eg = result["eg"]
    hl = result["half_life_static"]
    stats = result["stats"]
    premium = result["frame"]["premium"].dropna()
    return {
        "nobs": eg["nobs"],
        "eg_pvalue": eg["pvalue"],
        "eg_beta": eg["beta"],
        "eg_t": eg["t_stat"],
        "half_life": hl["half_life"],
        "sharpe": stats["sharpe"],
        "cagr": stats["cagr"],
        "max_drawdown": stats["max_drawdown"],
        "n_trades": stats["n_trades"],
        "latest_premium": float(premium.iloc[-1]) if len(premium) else np.nan,
    }
