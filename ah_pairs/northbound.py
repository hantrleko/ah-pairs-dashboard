"""北向资金与价差变化的描述性对照。

北向成交在 2024 年 8 月之后不再按日披露，序列后段会是空的。
这是市场层面的净买入，不是这一只股票的北向持仓变化。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from ah_pairs.paths import CACHE_DIR


def _as_dates(values) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(values)
    if isinstance(parsed, pd.Series):
        parsed = pd.DatetimeIndex(parsed)
    if getattr(parsed, "tz", None) is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(parsed).normalize()


def _cache_path(cache_dir: Path) -> Path:
    return cache_dir / "northbound.parquet"


def _read_flow(path: Path) -> pd.Series:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        if "date" in frame.columns:
            frame = frame.set_index("date")
    else:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    series = frame.iloc[:, 0].astype(float)
    series.index = _as_dates(series.index)
    series.name = "northbound"
    return series.sort_index()


def load_northbound(cache_dir: Path = CACHE_DIR, refresh: bool = False) -> pd.Series:
    path = _cache_path(cache_dir)
    legacy = cache_dir / "northbound.csv"
    if not refresh and path.exists():
        return _read_flow(path)
    if not refresh and legacy.exists():
        return _read_flow(legacy)

    import akshare as ak

    raw = ak.stock_hsgt_hist_em(symbol="北向资金")
    # 第 0 列是日期，第 1 列是当日成交净买额，单位亿元。
    series = pd.Series(
        pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(),
        index=_as_dates(raw.iloc[:, 0]),
        name="northbound",
    )
    series = series[~series.index.duplicated(keep="last")].sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = series.to_frame("northbound")
    frame.index.name = "date"
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp)
    tmp.replace(path)
    return series


def describe_northbound(spread: pd.Series, northbound: pd.Series) -> dict:
    """Δ价差 对当日北向净买入。HAC 标准误，滞后 5 个交易日。"""
    change = spread.diff()
    sample = pd.concat(
        [change.rename("d_spread"), northbound.rename("northbound")],
        axis=1,
    ).dropna()
    if len(sample) < 60:
        return {
            "nobs": int(len(sample)),
            "ok": False,
            "message": "重叠样本少于 60 个交易日，没有做回归。",
        }
    y = sample["d_spread"]
    x = sm.add_constant(sample["northbound"])
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    beta = float(fit.params["northbound"])
    se = float(fit.bse["northbound"])
    return {
        "ok": True,
        "nobs": int(fit.nobs),
        "start": sample.index.min(),
        "end": sample.index.max(),
        "beta": beta,
        "se": se,
        "t_stat": float(fit.tvalues["northbound"]),
        "pvalue": float(fit.pvalues["northbound"]),
        "r2": float(fit.rsquared),
        "corr": float(sample["d_spread"].corr(sample["northbound"])),
        "message": "描述性回归，不是北向资金影响价差的因果估计。",
    }


def latest_valid(series: pd.Series):
    valid = series.dropna()
    if valid.empty:
        return None
    return valid.index.max()
