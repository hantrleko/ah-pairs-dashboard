"""因果动态对冲比率：滚动 OLS 与卡尔曼滤波。

t 日的 alpha、beta 只使用 t 之前的价格。当日价差再用当日价格计算。
卡尔曼的 δ 可以用预测误差的高斯似然在一段前缀上选取，选取本身若用到交易样本就不是样本外。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ols(y, x, window: int) -> tuple[np.ndarray, np.ndarray]:
    """beta[t] 来自 [t-window, t) 的 OLS。用滚动矩的闭式，和带截距的最小二乘是同一条式子。"""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    n = len(y_arr)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    if window < 20 or n <= window:
        return alpha, beta
    y_arr = y_arr.copy()
    x_arr = x_arr.copy()
    y_arr[~np.isfinite(y_arr)] = np.nan
    x_arr[~np.isfinite(x_arr)] = np.nan
    y_past = pd.Series(y_arr).shift(1)
    x_past = pd.Series(x_arr).shift(1)
    mean_y = y_past.rolling(window, min_periods=window).mean()
    mean_x = x_past.rolling(window, min_periods=window).mean()
    mean_xx = (x_past * x_past).rolling(window, min_periods=window).mean()
    mean_xy = (y_past * x_past).rolling(window, min_periods=window).mean()
    var_x = mean_xx - mean_x * mean_x
    cov_xy = mean_xy - mean_x * mean_y
    valid = var_x > 1e-18
    beta_s = (cov_xy / var_x).where(valid)
    alpha_s = (mean_y - beta_s * mean_x).where(valid)
    return alpha_s.to_numpy(dtype=float), beta_s.to_numpy(dtype=float)


def _kalman_path(y_arr, x_arr, delta: float, init_window: int) -> tuple[np.ndarray, np.ndarray, float, int]:
    n = len(y_arr)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    if init_window < 20 or n <= init_window + 1:
        return alpha, beta, np.nan, 0
    if not (
        np.isfinite(y_arr[:init_window]).all() and np.isfinite(x_arr[:init_window]).all()
    ):
        return alpha, beta, np.nan, 0

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
    loglik = 0.0
    n_obs = 0

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
        loglik += -0.5 * (np.log(2.0 * np.pi) + np.log(scale) + innovation * innovation / scale)
        n_obs += 1
        gain = (cov_pred @ obs) / scale
        theta = theta + gain * innovation
        ikh = eye - np.outer(gain, obs)
        cov = ikh @ cov_pred @ ikh.T + np.outer(gain, gain) * obs_var
        cov = 0.5 * (cov + cov.T)
    return alpha, beta, float(loglik), int(n_obs)


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
    alpha, beta, _loglik, _n = _kalman_path(y_arr, x_arr, delta, init_window)
    return alpha, beta


def estimate_kalman_delta(
    y,
    x,
    init_window: int = 60,
    deltas: tuple[float, ...] | None = None,
) -> dict:
    """在给定前缀上，用一步预测误差的高斯似然挑选 δ。"""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    grid = tuple(float(item) for item in (deltas or tuple(np.logspace(-6, -2, 13))))
    best_delta = 1e-4
    best_ll = -np.inf
    n_obs = 0
    for delta in grid:
        _alpha, _beta, loglik, used = _kalman_path(y_arr, x_arr, delta, init_window)
        if used > n_obs:
            n_obs = used
        if np.isfinite(loglik) and loglik > best_ll:
            best_ll = float(loglik)
            best_delta = float(delta)
    ok = np.isfinite(best_ll) and n_obs >= 30
    return {
        "ok": bool(ok),
        "delta": float(best_delta if ok else 1e-4),
        "loglik": float(best_ll) if np.isfinite(best_ll) else np.nan,
        "n_obs": int(n_obs),
        "n_grid": int(len(grid)),
    }


def rolling_zscore(spread, window: int) -> np.ndarray:
    """z[t] 用含当日在内的滚动均值和标准差。交易放到之后的收盘，不用未来价差。"""
    series = pd.Series(np.asarray(spread, dtype=float))
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=1)
    z = (series - mean) / std
    z = z.where(std > 1e-12)
    return z.to_numpy(dtype=float)
