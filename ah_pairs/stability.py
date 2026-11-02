"""协整关系的扰动稳定性。

每个窗口里的 Engle-Granger 都用了该窗口的全部价格，用来看关系经不经得起换样本，
不是交易信号。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

DEFAULT_LENGTHS = (126, 252, 504, 756, 1008)


def eg_pvalue(y, x, autolag: str | None = None) -> float:
    """Engle-Granger p 值。默认滞后固定为 1，便于大量重复计算。autolag='aic' 与摘要页同一口径。"""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(x_arr)
    y_arr = y_arr[mask]
    x_arr = x_arr[mask]
    if len(y_arr) < 80:
        return np.nan
    try:
        if autolag == "aic":
            _stat, pvalue, _crit = coint(y_arr, x_arr, trend="c", autolag="aic")
        else:
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


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> dict:
    """Benjamini–Hochberg。返回被拒绝的个数，不是把每个 p 值和 0.05 单独比较。"""
    p = np.sort(np.asarray(pvalues, dtype=float))
    p = p[np.isfinite(p)]
    m = int(len(p))
    if m == 0:
        return {"m": 0, "n_reject": 0, "rate": np.nan, "alpha": float(alpha)}
    k = 0
    for rank, pvalue in enumerate(p, start=1):
        if pvalue <= alpha * rank / m:
            k = rank
    return {"m": m, "n_reject": int(k), "rate": float(k / m), "alpha": float(alpha)}


def window_count_facts(n_days: int, lengths: tuple[int, ...] = DEFAULT_LENGTHS, step: int | None = None) -> dict:
    """窗口个数。相邻窗口重叠，次数不是独立样本数。"""
    if step is None:
        step = max(21, int(n_days) // 36)
    parts = []
    total = 0
    for length in lengths:
        if length >= n_days or length < 80:
            continue
        ends = list(range(length, n_days + 1, step))
        if not ends or ends[-1] != n_days:
            ends.append(n_days)
        parts.append((int(length), int(len(ends))))
        total += len(ends)
    return {"n_days": int(n_days), "step": int(step), "total": int(total), "parts": parts}


def window_count_note(n_days: int, lengths: tuple[int, ...] = DEFAULT_LENGTHS, step: int | None = None) -> str:
    """说明地图上的窗口是怎么数出来的。窗口重叠，次数不是独立样本数。"""
    facts = window_count_facts(n_days, lengths=lengths, step=step)
    parts = [f"{length} 日 {count} 个" for length, count in facts["parts"]]
    detail = "、".join(parts) if parts else "没有足够长的窗口"
    return (
        f"这段有 {facts['n_days']} 个价格点。步长 {facts['step']} 个交易日，每种长度都包含最后一个终点。"
        f"按这个设计大约 {facts['total']} 个窗口（{detail}）。相邻窗口共用大部分交易日，不能把检验次数当成独立样本。"
        "下面的 BH 比例把这些 p 值放进 Benjamini–Hochberg，只作参考：窗口重叠，校正既不是严格的独立试验，也不代替 AIC 那一次全样本检验。"
    )


def fragility_report(grid: pd.DataFrame, top: int = 5) -> dict:
    """成立比例、脆弱程度，以及 p 值最高的几个窗口。"""
    empty = {
        "n_tests": 0,
        "reject_rate": np.nan,
        "fragility": np.nan,
        "bh_n_reject": 0,
        "bh_rate": np.nan,
        "worst": grid,
    }
    if grid.empty or grid["pvalue"].notna().sum() == 0:
        return empty
    valid = grid.dropna(subset=["pvalue"])
    rate = float(valid["reject_5"].mean())
    worst = valid.sort_values(["pvalue", "length"], ascending=[False, False]).head(top)
    bh = benjamini_hochberg(valid["pvalue"])
    return {
        "n_tests": int(len(valid)),
        "reject_rate": rate,
        "fragility": float(1.0 - rate),
        "bh_n_reject": int(bh["n_reject"]),
        "bh_rate": bh["rate"],
        "worst": worst.reset_index(drop=True),
    }


def length_summary(grid: pd.DataFrame) -> pd.DataFrame:
    """每种窗口长度上的检验次数和 5% 成立比例。"""
    columns = ["length", "n_tests", "reject_rate"]
    if grid.empty or "pvalue" not in grid.columns or grid["pvalue"].notna().sum() == 0:
        return pd.DataFrame(columns=columns)
    valid = grid.dropna(subset=["pvalue"])
    out = (
        valid.groupby("length", as_index=False)
        .agg(n_tests=("reject_5", "size"), reject_rate=("reject_5", "mean"))
        .sort_values("length")
        .reset_index(drop=True)
    )
    return out


def slice_checks(frame: pd.DataFrame) -> pd.DataFrame:
    """全区间、前后半段、去掉一段，以及剔除极端日之后的检验。

    code 是稳定代号。情形列保留中文，界面按 code 翻译。
    """
    from ah_pairs.labels import SLICE_ZH

    specs = [("full", frame)]
    n = len(frame)
    if n >= 240:
        specs.append(("first_half", frame.iloc[: n // 2]))
        specs.append(("second_half", frame.iloc[n // 2 :]))
    if n >= 400:
        specs.append(("drop_first_year", frame.iloc[252:]))
        specs.append(("drop_last_year", frame.iloc[:-252]))
    specs.extend(_extreme_slices(frame))
    if {"log_a_raw", "log_h_raw"}.issubset(frame.columns):
        raw = frame.dropna(subset=["log_a_raw", "log_h_raw"]).copy()
        if len(raw) >= 80:
            raw["log_a"] = raw["log_a_raw"]
            raw["log_h"] = raw["log_h_raw"]
            specs.append(("unadjusted", raw))
    rows = []
    for code, sample in specs:
        p_fixed = eg_pvalue(sample["log_a"], sample["log_h"])
        p_aic = eg_pvalue(sample["log_a"], sample["log_h"], autolag="aic")
        rows.append(
            {
                "code": code,
                "情形": SLICE_ZH.get(code, code),
                "天数": int(len(sample.dropna(subset=["log_a", "log_h"]))),
                "pvalue": p_fixed,
                "reject_5": bool(np.isfinite(p_fixed) and p_fixed < 0.05),
                "pvalue_aic": p_aic,
                "reject_5_aic": bool(np.isfinite(p_aic) and p_aic < 0.05),
            }
        )
    return pd.DataFrame(rows)


def _extreme_slices(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    out = []
    if "ret_a" not in frame.columns or "ret_h" not in frame.columns:
        return out
    ret_a = frame["ret_a"].abs()
    ret_h = frame["ret_h"].abs()
    out.append(("drop_move_5", frame.loc[(ret_a <= 0.05) & (ret_h <= 0.05)]))
    out.append(("drop_move_3", frame.loc[(ret_a <= 0.03) & (ret_h <= 0.03)]))
    if "spread" in frame.columns:
        change = frame["spread"].diff().abs()
        cutoff = change.quantile(0.99)
        if np.isfinite(cutoff):
            keep = change.isna() | (change <= cutoff)
            out.append(("drop_spread_p99", frame.loc[keep]))
    return out
