"""Engle-Granger、Johansen 与 OU 半衰期。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def engle_granger(y, x, trend: str = "c") -> dict:
    """log(A) 对 log(H) 的 Engle-Granger 检验。

    返回的 p 值是协整检验的 MacKinnon p 值，不是残差 ADF 的普通 p 值。
    这是整段样本的描述，回归系数用到了样本内的全部观测。
    """
    y_s = pd.Series(y, dtype=float).dropna()
    x_s = pd.Series(x, dtype=float).reindex(y_s.index).dropna()
    y_s = y_s.reindex(x_s.index)
    if len(y_s) < 40:
        raise ValueError("协整检验至少需要 40 个重叠交易日")

    stat, pvalue, crit = coint(
        y_s.to_numpy(), x_s.to_numpy(), trend=trend, autolag="aic"
    )
    design = sm.add_constant(x_s.to_numpy())
    fit = sm.OLS(y_s.to_numpy(), design).fit()
    alpha = float(fit.params[0])
    beta = float(fit.params[1])
    resid = pd.Series(fit.resid, index=y_s.index, name="static_resid")
    crit_arr = np.asarray(crit, dtype=float).ravel()
    return {
        "nobs": int(fit.nobs),
        "alpha": alpha,
        "beta": beta,
        "r2": float(fit.rsquared),
        "t_stat": float(stat),
        "pvalue": float(pvalue),
        "crit_1": float(crit_arr[0]),
        "crit_5": float(crit_arr[1]),
        "crit_10": float(crit_arr[2]),
        "reject_5": bool(pvalue < 0.05),
        "resid": resid,
    }


def residual_adf(resid) -> dict:
    """对协整回归残差再做一次普通 ADF。

    残差是用同一段样本估计出来的，这个 p 值会偏小，不能代替 Engle-Granger 的 p 值。
    """
    series = pd.Series(resid, dtype=float).dropna()
    if len(series) < 40:
        raise ValueError("残差 ADF 至少需要 40 个观测")
    stat, pvalue, usedlag, nobs, crit, _ic = adfuller(series.to_numpy(), autolag="aic")
    return {
        "adf_stat": float(stat),
        "adf_pvalue": float(pvalue),
        "adf_lag": int(usedlag),
        "adf_nobs": int(nobs),
        "adf_crit_5": float(crit["5%"]),
        "note": "普通 ADF。残差来自同一样本的回归，p 值偏乐观，不能代替 Engle-Granger p 值。",
    }


def select_vecm_lags(y, x, maxlags: int = 4) -> int:
    """用 VAR 的 AIC 选择 VECM 差分滞后。失败时退回 1 阶。"""
    try:
        from statsmodels.tsa.api import VAR

        data = pd.DataFrame({"y": np.asarray(y, float), "x": np.asarray(x, float)}).dropna()
        if len(data) < maxlags * 8:
            return 1
        order = VAR(data).select_order(maxlags=maxlags)
        level_lag = int(order.aic or 1)
        return max(level_lag - 1, 0)
    except Exception:
        return 1


def johansen_test(y, x, k_ar_diff: int = 1, det_order: int = 0) -> dict:
    """两变量 Johansen 迹检验。det_order=0 表示协整关系里有常数。"""
    data = np.column_stack([np.asarray(y, float), np.asarray(x, float)])
    mask = np.isfinite(data).all(axis=1)
    data = data[mask]
    if len(data) < 50:
        raise ValueError("Johansen 检验至少需要 50 个重叠交易日")
    res = coint_johansen(data, det_order=det_order, k_ar_diff=int(k_ar_diff))
    trace = float(res.lr1[0])
    crit95 = float(res.cvt[0, 1])
    max_eig = float(res.lr2[0])
    max_crit95 = float(res.cvm[0, 1])
    return {
        "k_ar_diff": int(k_ar_diff),
        "trace_r0": trace,
        "trace_crit_95": crit95,
        "trace_reject_r0_95": bool(trace > crit95),
        "max_eig_r0": max_eig,
        "max_eig_crit_95": max_crit95,
        "max_eig_reject_r0_95": bool(max_eig > max_crit95),
        "nobs": int(len(data)),
    }


def ou_half_life(spread) -> dict:
    """用 AR(1) 估计价差半衰期。

    s_t = c + phi s_{t-1} + e_t。
    0 < phi < 1 时，半衰期 = -ln(2) / ln(phi)，单位是交易日。
    kappa = -ln(phi) 是连续时间 OU 的均值回复速度。
    """
    s = pd.Series(spread, dtype=float).dropna()
    if len(s) < 30:
        raise ValueError("半衰期估计至少需要 30 个价差观测")
    lagged = s.shift(1)
    design = pd.DataFrame({"const": 1.0, "lag": lagged})
    sample = pd.concat([s.rename("s"), design], axis=1).dropna()
    fit = sm.OLS(sample["s"], sample[["const", "lag"]]).fit()
    phi = float(fit.params["lag"])
    phi_se = float(fit.bse["lag"])
    intercept = float(fit.params["const"])
    out = {
        "nobs": int(fit.nobs),
        "phi": phi,
        "phi_se": phi_se,
        "half_life": np.nan,
        "kappa": np.nan,
        "long_run_mean": np.nan,
        "innovation_sigma": float(fit.resid.std(ddof=2)),
        "mean_reverting": False,
    }
    if 0 < phi < 1:
        out["half_life"] = float(-np.log(2.0) / np.log(phi))
        out["kappa"] = float(-np.log(phi))
        out["long_run_mean"] = float(intercept / (1.0 - phi))
        out["mean_reverting"] = True
    return out


def rolling_eg_pvalue(y, x, window: int = 252, step: int = 5) -> pd.Series:
    """滚动 Engle-Granger p 值。窗口只用当时已经实现的数据。"""
    y_s = pd.Series(y, dtype=float)
    x_s = pd.Series(x, dtype=float)
    values = []
    indexes = []
    n = len(y_s)
    for end in range(window, n + 1, step):
        yy = y_s.iloc[end - window : end]
        xx = x_s.iloc[end - window : end]
        if yy.isna().any() or xx.isna().any():
            continue
        try:
            _stat, pvalue, _crit = coint(
                yy.to_numpy(),
                xx.to_numpy(),
                trend="c",
                maxlag=1,
                autolag=None,
            )
        except Exception:
            continue
        values.append(float(pvalue))
        indexes.append(y_s.index[end - 1])
    return pd.Series(values, index=indexes, name="eg_pvalue")
