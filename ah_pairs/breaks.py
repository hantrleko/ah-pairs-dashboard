"""Gregory–Hansen 协整检验：允许一次未知断点。

临界值来自 Gregory and Hansen (1996, Journal of Econometrics) 表 1，
ADF*，回归元个数 m=1。模型 C 的 5% 临界值是 -4.61。
这些数不是 Engle–Granger 的 MacKinnon p 值，不能替换摘要页的 EG p 值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 模型, 1%, 5%, 10%。统计量要小于临界值才拒绝「没有协整」。
_CRIT = {
    "C": (-5.13, -4.61, -4.34),
    "CT": (-5.45, -4.99, -4.72),
    "CS": (-5.47, -4.95, -4.68),
}
_NAMES = {
    "C": "水平位移",
    "CT": "水平位移加趋势",
    "CS": "水平位移和斜率一起变",
}


def _adf_t(resid: np.ndarray, lag: int) -> float:
    """残差 ADF 的 t 统计量。回归不含常数，因为协整式里已经有常数。"""
    dy = np.diff(resid)
    lagged = resid[:-1]
    start = int(lag)
    if len(dy) - start < lag + 5:
        return np.nan
    columns = [lagged[start:]]
    for step in range(1, start + 1):
        columns.append(dy[start - step : len(dy) - step])
    design = np.column_stack(columns)
    outcome = dy[start:]
    coef, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    fitted = design @ coef
    error = outcome - fitted
    dof = len(outcome) - design.shape[1]
    if dof <= 0:
        return np.nan
    sigma2 = float(np.dot(error, error) / dof)
    try:
        variance = sigma2 * np.linalg.inv(design.T @ design)[0, 0]
    except np.linalg.LinAlgError:
        return np.nan
    if not np.isfinite(variance) or variance <= 0:
        return np.nan
    return float(coef[0] / np.sqrt(variance))


def _schwert_lag(nobs: int) -> int:
    return max(1, int(np.floor(4.0 * (nobs / 100.0) ** 0.25)))


def _design(x: np.ndarray, dummy: np.ndarray, model: str, trend: np.ndarray) -> np.ndarray:
    if model == "C":
        return np.column_stack([np.ones(len(x)), dummy, x])
    if model == "CT":
        return np.column_stack([np.ones(len(x)), dummy, trend, x])
    return np.column_stack([np.ones(len(x)), dummy, x, x * dummy])


def gregory_hansen(y, x, trim: float = 0.15) -> dict:
    """在 log A 和 log H 上搜索一次断点。返回三个模型。"""
    left = pd.Series(np.asarray(y, dtype=float), index=getattr(y, "index", None))
    right = pd.Series(np.asarray(x, dtype=float), index=getattr(x, "index", None))
    pair = pd.concat({"y": left, "x": right}, axis=1).dropna()
    nobs = len(pair)
    if nobs < 80:
        return {"ok": False, "message": f"结构断点至少需要 80 个同时开市日，现在是 {nobs}。"}
    values_y = pair["y"].to_numpy(dtype=float)
    values_x = pair["x"].to_numpy(dtype=float)
    trend = np.arange(nobs, dtype=float)
    lag = _schwert_lag(nobs)
    first = int(np.floor(nobs * trim))
    last = int(np.ceil(nobs * (1.0 - trim))) - 1
    if last <= first:
        return {"ok": False, "message": "样本去掉两端之后没有可搜索的断点。"}
    rows = []
    for model in ("C", "CT", "CS"):
        best_stat = np.inf
        best_at = first
        best_shift = np.nan
        for point in range(first, last + 1):
            dummy = np.zeros(nobs)
            dummy[point + 1 :] = 1.0
            design = _design(values_x, dummy, model, trend)
            coef, _, _, _ = np.linalg.lstsq(design, values_y, rcond=None)
            resid = values_y - design @ coef
            stat = _adf_t(resid, lag)
            if np.isfinite(stat) and stat < best_stat:
                best_stat = float(stat)
                best_at = point
                best_shift = float(coef[1])
        crit_1, crit_5, crit_10 = _CRIT[model]
        break_at = min(best_at + 1, nobs - 1)
        rows.append(
            {
                "model": model,
                "name": _NAMES[model],
                "stat": None if not np.isfinite(best_stat) else float(best_stat),
                "crit_1": crit_1,
                "crit_5": crit_5,
                "crit_10": crit_10,
                "reject_5": bool(np.isfinite(best_stat) and best_stat < crit_5),
                "break_date": pair.index[break_at],
                "break_fraction": float(break_at / nobs),
                "level_shift": best_shift,
            }
        )
    return {
        "ok": True,
        "nobs": int(nobs),
        "lag": int(lag),
        "trim": float(trim),
        "models": rows,
    }
