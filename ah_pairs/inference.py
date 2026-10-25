"""夏普的区块自助法区间，以及 Bailey–López de Prado 的压缩夏普概率。

自助法区间针对年化夏普。概率用的是同一条日收益的非年化夏普：大于 0 与年化夏普大于 0 是同一件事。
试验次数按这次实际搜过的参数组合计。它不是独立样本的个数。
"""

from __future__ import annotations

import numpy as np

EULER = 0.5772156649015329


def _finite(returns) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    return values[np.isfinite(values)]


def annualized_sharpe(returns) -> float:
    values = _finite(returns)
    if len(values) < 3:
        return np.nan
    vol = float(np.std(values, ddof=1))
    if not np.isfinite(vol) or vol <= 0:
        return np.nan
    return float(np.mean(values) / vol * np.sqrt(252.0))


def block_bootstrap_sharpe(
    returns,
    block: int = 21,
    n_boot: int = 400,
    seed: int = 7,
) -> dict:
    """环形区块自助法。区间是年化夏普的 2.5% 和 97.5% 分位。"""
    values = _finite(returns)
    point = annualized_sharpe(values)
    n = len(values)
    if n < max(block * 2, 30) or not np.isfinite(point):
        return {
            "ok": False,
            "sharpe": point,
            "low": np.nan,
            "high": np.nan,
            "n": int(n),
            "block": int(block),
            "n_boot": 0,
        }
    rng = np.random.default_rng(seed)
    width = min(int(block), n)
    n_blocks = int(np.ceil(n / width))
    draws = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        starts = rng.integers(0, n, size=n_blocks)
        pieces = []
        for start in starts:
            if start + width <= n:
                pieces.append(values[start : start + width])
            else:
                tail = n - start
                pieces.append(np.concatenate([values[start:], values[: width - tail]]))
        sample = np.concatenate(pieces)[:n]
        draws[i] = annualized_sharpe(sample)
    draws = draws[np.isfinite(draws)]
    if len(draws) < 20:
        return {
            "ok": False,
            "sharpe": point,
            "low": np.nan,
            "high": np.nan,
            "n": int(n),
            "block": width,
            "n_boot": int(len(draws)),
        }
    return {
        "ok": True,
        "sharpe": point,
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "n": int(n),
        "block": width,
        "n_boot": int(len(draws)),
    }


def sharpe_probability(returns, n_trials: int = 1) -> dict:
    """非年化夏普大于 0 的概率。n_trials>1 时用压缩夏普的原假设，否则原假设是 0。"""
    from scipy.stats import kurtosis, norm, skew

    values = _finite(returns)
    n = len(values)
    if n < 30:
        return {"ok": False, "psr": np.nan, "dsr": np.nan, "sr": np.nan, "sr_benchmark": np.nan, "n_trials": int(n_trials)}
    vol = float(np.std(values, ddof=1))
    if not np.isfinite(vol) or vol <= 0:
        return {"ok": False, "psr": np.nan, "dsr": np.nan, "sr": np.nan, "sr_benchmark": 0.0, "n_trials": int(max(n_trials, 1))}
    sr = float(np.mean(values) / vol)
    gamma3 = float(skew(values, bias=False))
    gamma4 = float(kurtosis(values, fisher=False, bias=False))
    if not np.isfinite(gamma3):
        gamma3 = 0.0
    if not np.isfinite(gamma4):
        gamma4 = 3.0
    inner = 1.0 - gamma3 * sr + ((gamma4 - 1.0) / 4.0) * sr * sr
    if not np.isfinite(inner) or inner <= 1e-12:
        inner = 1e-12
    se = float(np.sqrt(inner / (n - 1)))
    trials = int(max(n_trials, 1))
    sr_benchmark = 0.0
    if trials > 1:
        z_term = (1.0 - EULER) * float(norm.ppf(1.0 - 1.0 / trials))
        z_term += EULER * float(norm.ppf(1.0 - 1.0 / (trials * np.e)))
        sr_benchmark = se * z_term
    psr = float(norm.cdf(sr / se))
    dsr = float(norm.cdf((sr - sr_benchmark) / se))
    return {
        "ok": True,
        "sr": sr,
        "psr": psr,
        "dsr": dsr,
        "sr_benchmark": float(sr_benchmark),
        "n_trials": trials,
        "n": int(n),
    }
