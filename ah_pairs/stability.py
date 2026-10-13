"""协整关系的扰动稳定性。

每个窗口里的 Engle-Granger 都用了该窗口的全部价格，用来看关系经不经得起换样本，
不是交易信号。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

DEFAULT_LENGTHS = (126, 252, 504, 756, 1008)


def eg_pvalue(y, x) -> float:
    """快速 Engle-Granger p 值。滞后固定为 1，便于大量重复计算。"""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(x_arr)
    y_arr = y_arr[mask]
    x_arr = x_arr[mask]
    if len(y_arr) < 80:
        return np.nan
    try:
        _stat, pvalue, _crit = coint(y_arr, x_arr, trend="c", maxlag=1, autolag=None)
    except Exception:
        return np.nan
    return float(pvalue)


def stability_grid(
    log_a,
    log_h,
    index,
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
    step: int | None = None,
) -> pd.DataFrame:
    """不同窗口长度、不同终点上的协整 p 值。"""
    y = np.asarray(log_a, dtype=float)
    x = np.asarray(log_h, dtype=float)
    dates = pd.DatetimeIndex(index)
    n = len(y)
    if step is None:
        step = max(21, n // 36)
    rows = []
    for length in lengths:
        if length >= n or length < 80:
            continue
        ends = list(range(length, n + 1, step))
        if not ends or ends[-1] != n:
            ends.append(n)
        for end in ends:
            start = end - length
            pvalue = eg_pvalue(y[start:end], x[start:end])
            rows.append(
                {
                    "start": dates[start],
                    "end": dates[end - 1],
                    "length": int(length),
                    "nobs": int(length),
                    "pvalue": pvalue,
                    "reject_5": bool(np.isfinite(pvalue) and pvalue < 0.05),
                }
            )
    return pd.DataFrame(rows)


def fragility_report(grid: pd.DataFrame, top: int = 5) -> dict:
    """成立比例、脆弱程度，以及 p 值最高的几个窗口。"""
    if grid.empty or grid["pvalue"].notna().sum() == 0:
        return {
            "n_tests": 0,
            "reject_rate": np.nan,
            "fragility": np.nan,
            "worst": grid,
        }
    valid = grid.dropna(subset=["pvalue"])
    rate = float(valid["reject_5"].mean())
    worst = valid.sort_values(["pvalue", "length"], ascending=[False, False]).head(top)
    return {
        "n_tests": int(len(valid)),
        "reject_rate": rate,
        "fragility": float(1.0 - rate),
        "worst": worst.reset_index(drop=True),
    }


def slice_checks(frame: pd.DataFrame) -> pd.DataFrame:
    """全区间、前后半段、去掉一段，以及剔除极端日之后的检验。"""
    specs = [("全区间", frame)]
    n = len(frame)
    if n >= 240:
        specs.append(("前半段", frame.iloc[: n // 2]))
        specs.append(("后半段", frame.iloc[n // 2 :]))
    if n >= 400:
        specs.append(("去掉最早约一年", frame.iloc[252:]))
        specs.append(("去掉最近约一年", frame.iloc[:-252]))
    specs.extend(_extreme_slices(frame))
    rows = []
    for name, sample in specs:
        pvalue = eg_pvalue(sample["log_a"], sample["log_h"])
        rows.append(
            {
                "情形": name,
                "天数": int(len(sample.dropna(subset=["log_a", "log_h"]))),
                "pvalue": pvalue,
                "reject_5": bool(np.isfinite(pvalue) and pvalue < 0.05),
            }
        )
    return pd.DataFrame(rows)


def _extreme_slices(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    out = []
    if "ret_a" not in frame.columns or "ret_h" not in frame.columns:
        return out
    ret_a = frame["ret_a"].abs()
    ret_h = frame["ret_h"].abs()
    out.append(("去掉单日涨跌超过 5% 的日子", frame.loc[(ret_a <= 0.05) & (ret_h <= 0.05)]))
    out.append(("去掉单日涨跌超过 3% 的日子", frame.loc[(ret_a <= 0.03) & (ret_h <= 0.03)]))
    if "spread" in frame.columns:
        change = frame["spread"].diff().abs()
        cutoff = change.quantile(0.99)
        if np.isfinite(cutoff):
            keep = change.isna() | (change <= cutoff)
            out.append(("去掉价差跳动最大的 1% 日子", frame.loc[keep]))
    return out
