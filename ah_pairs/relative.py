"""个股溢价减去恒生 AH 溢价指数。

检验的是这只股票相对市场溢价的偏离会不会回到常数附近。
这是一条已经指定系数的价差，普通 ADF 不能代替 Engle–Granger 的 p 值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from ah_pairs.cointegration import ou_half_life


def index_to_premium(level: pd.Series) -> pd.Series:
    """恒生 AH 溢价指数 100 表示平价。已经是溢价小数时不再除。"""
    series = pd.Series(level, dtype=float).dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        raise ValueError("恒生 AH 溢价指数是空的")
    median = float(series.median())
    if median > 30:
        out = series / 100.0 - 1.0
    elif -0.2 <= median <= 1.5:
        out = series
    else:
        raise ValueError(f"指数中位数 {median:.2f} 对不上溢价")
    out.name = "index_premium"
    return out


def relative_premium(stock_premium: pd.Series, index_level: pd.Series | None) -> dict:
    """个股溢价减市场溢价，再做带常数的 ADF。

    直接相减把市场溢价的系数定成 1。原假设是相对溢价有单位根。
    """
    if index_level is None or len(getattr(index_level, "index", [])) == 0:
        return {"ok": False, "message": "还没有恒生 AH 溢价指数。这一项先不算。"}
    try:
        market = index_to_premium(index_level)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}
    stock = pd.Series(stock_premium, dtype=float).dropna()
    stock.index = pd.DatetimeIndex(pd.to_datetime(stock.index)).normalize()
    market.index = pd.DatetimeIndex(pd.to_datetime(market.index)).normalize()
    both = pd.concat({"stock": stock, "market": market}, axis=1).dropna()
    if len(both) < 40:
        return {"ok": False, "message": f"个股溢价和指数重叠只有 {len(both)} 天，没有做相对溢价检验。"}
    residual = both["stock"] - both["market"]
    residual.name = "relative_premium"
    stat, pvalue, usedlag, nobs, crit, _ic = adfuller(residual.to_numpy(dtype=float), autolag="aic")
    try:
        half = ou_half_life(residual)
    except ValueError:
        half = {"half_life": np.nan, "mean_reverting": False}
    return {
        "ok": True,
        "nobs": int(nobs),
        "adf_stat": float(stat),
        "adf_pvalue": float(pvalue),
        "adf_lag": int(usedlag),
        "crit_5": float(crit["5%"]),
        "reject_5": bool(pvalue < 0.05),
        "half_life": half.get("half_life", np.nan),
        "mean_reverting": bool(half.get("mean_reverting")),
        "latest": float(residual.iloc[-1]),
        "mean": float(residual.mean()),
        "start": both.index.min(),
        "end": both.index.max(),
    }
