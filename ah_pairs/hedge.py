"""因果动态对冲比率：滚动 OLS 与卡尔曼滤波。

t 日的 alpha、beta 只使用 t 之前的价格。当日价差再用当日价格计算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ols(y, x, window: int) -> tuple[np.ndarray, np.ndarray]:
    """beta[t] 来自 [t-window, t) 的 OLS。"""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    n = len(y_arr)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    if window < 20 or n <= window:
        return alpha, beta
    for t in range(window, n):
        ys = y_arr[t - window : t]
        xs = x_arr[t - window : t]
        if not (np.isfinite(ys).all() and np.isfinite(xs).all()):
            continue
        design = np.column_stack([np.ones(window), xs])
        coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
        alpha[t] = coef[0]
        beta[t] = coef[1]
    return alpha, beta


def kalman_hedge(
    y,
    x,
    delta: float = 1e-4,
    init_window: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """时变截距和斜率。

    状态是随机游走。观测噪声取初始 OLS 的残差方差，状态噪声是 delta 乘这一个方差。
    写入 alpha[t]、beta[t] 的是预测状态，更新发生在记录之后，因此不使用 y[t]。
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    n = len(y_arr)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    if init_window < 20 or n <= init_window + 1:
        return alpha, beta
    if not (
        np.isfinite(y_arr[:init_window]).all() and np.isfinite(x_arr[:init_window]).all()
    ):
        return alpha, beta

    design = np.column_stack([np.ones(init_window), x_arr[:init_window]])
    coef, *_ = np.linalg.lstsq(design, y_arr[:init_window], rcond=None)
    resid = y_arr[:init_window] - design @ coef
    obs_var = float(np.var(resid, ddof=2))
    if not np.isfinite(obs_var) or obs_var <= 1e-12:
        obs_var = 1e-6

    theta = coef.astype(float)
    cov = np.eye(2) * obs_var
    state_var = np.eye(2) * (float(delta) * obs_var)
    eye = np.eye(2)

    for t in range(init_window, n):
        if not (np.isfinite(y_arr[t]) and np.isfinite(x_arr[t])):
            continue
        cov_pred = cov + state_var
        alpha[t] = theta[0]
        beta[t] = theta[1]
        obs = np.array([1.0, x_arr[t]])
        innovation = float(y_arr[t] - obs @ theta)
        scale = float(obs @ cov_pred @ obs + obs_var)
        if not np.isfinite(scale) or scale <= 0:
            continue
        gain = (cov_pred @ obs) / scale
        theta = theta + gain * innovation
        ikh = eye - np.outer(gain, obs)
        cov = ikh @ cov_pred @ ikh.T + np.outer(gain, gain) * obs_var
        cov = 0.5 * (cov + cov.T)
    return alpha, beta


def rolling_zscore(spread, window: int) -> np.ndarray:
    """z[t] 用含当日在内的滚动均值和标准差。交易放到之后的收盘，不用未来价差。"""
    series = pd.Series(np.asarray(spread, dtype=float))
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=1)
    z = (series - mean) / std
    z = z.where(std > 1e-12)
    return z.to_numpy(dtype=float)
