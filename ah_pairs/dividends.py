"""H 股现金红利的 20% 预扣，只作为成本对照的一行。

主回测仍用复权价，不在这里改。税只在前一日已经持有 H 股多头时计提。
空仓和做空 H 股的日子为 0。没有分红序列时不算这一行。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

H_DIVIDEND_TAX = 0.20


def h_dividend_tax_cny(dividend_hkd, fx, long_h) -> pd.Series:
    """每股预扣，单位是人民币。long_h 为假的日子返回 0。"""
    dividend = _series(dividend_hkd)
    rate = _series(fx).reindex(dividend.index)
    held = _series(long_h).reindex(dividend.index)
    held = held.fillna(0).astype(float) > 0
    tax = dividend.fillna(0.0) * H_DIVIDEND_TAX * rate
    tax = tax.where(held & np.isfinite(tax), 0.0)
    tax.name = "h_div_tax_cny"
    return tax


def h_dividend_return_drag(frame: pd.DataFrame, dividends) -> pd.Series:
    """把每股红利税换成组合收益率。用的是前一日的 H 股权重。"""
    dividend = _series(dividends)
    by_day = dividend.groupby(dividend.index).sum()
    aligned = by_day.reindex(_days(frame.index)).fillna(0.0)
    aligned.index = frame.index
    weight = frame["w_h"].shift(1).fillna(0.0).astype(float)
    price = frame["h_hkd"].astype(float)
    price = price.where(price > 0)
    drag = H_DIVIDEND_TAX * aligned.to_numpy(dtype=float) / price.to_numpy(dtype=float) * weight.to_numpy(dtype=float)
    drag = np.where(weight.to_numpy(dtype=float) > 0, drag, 0.0)
    drag = np.where(np.isfinite(drag), drag, 0.0)
    return pd.Series(drag, index=frame.index, name="h_div_drag")


def h_dividend_tax_row(frame: pd.DataFrame, dividends, trades: pd.DataFrame | None = None) -> pd.DataFrame:
    """成本对照用的一行。没有正的现金红利时，净收益和夏普留空。"""
    from ah_pairs.backtest import _performance_stats

    blank = {
        "code": "h_div_tax",
        "情形": "H 股红利税 20%",
        "净收益": np.nan,
        "夏普": np.nan,
        "交易次数": np.nan,
        "成本合计": np.nan,
        "融券费合计": np.nan,
        "status": "missing",
    }
    if dividends is None or not _has_cash_dividend(dividends):
        return pd.DataFrame([blank])
    if "w_h" not in frame.columns or "h_hkd" not in frame.columns or "ret_strat" not in frame.columns:
        return pd.DataFrame([blank])
    taxed = frame.copy()
    taxed["ret_strat"] = frame["ret_strat"].fillna(0.0) - h_dividend_return_drag(frame, dividends)
    taxed["equity"] = _equity(taxed["ret_strat"])
    peak = taxed["equity"].cummax().where(lambda column: column > 0)
    taxed["drawdown"] = taxed["equity"] / peak - 1.0
    if not isinstance(trades, pd.DataFrame):
        trades = pd.DataFrame(columns=["closed", "trade_return", "holding_days"])
    stats = _performance_stats(taxed, trades, 0, bool((taxed["equity"] <= 0).any()))
    blank.update(
        {
            "净收益": stats["total_return"],
            "夏普": stats["sharpe"],
            "交易次数": stats["n_trades"],
            "成本合计": stats["total_cost"],
            "融券费合计": stats["total_borrow"],
            "status": "ok",
        }
    )
    return pd.DataFrame([blank])


def _has_cash_dividend(dividends) -> bool:
    series = _series(dividends)
    return bool((series.fillna(0.0) > 0).any())


def _series(values) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values, dtype=float)
    series = series.copy()
    series.index = _days(series.index)
    return series.sort_index()


def _days(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()


def _equity(returns: pd.Series) -> pd.Series:
    equity = np.ones(len(returns))
    values = returns.fillna(0.0).to_numpy(dtype=float)
    ruined = False
    for i, step in enumerate(values):
        if ruined:
            equity[i] = 0.0
            continue
        growth = 1.0 + float(step)
        if growth <= 0:
            equity[i] = 0.0
            ruined = True
            continue
        base = 1.0 if i == 0 else equity[i - 1]
        equity[i] = base * growth
    return pd.Series(equity, index=returns.index)
