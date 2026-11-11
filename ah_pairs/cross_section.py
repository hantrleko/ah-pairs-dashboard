"""每月按溢价 z 给流动性子集排序。

做多 A 相对便宜的一组，同时做空这些名字的 H 股。不做空 A。
z 只用当时已经过去的窗口。换仓要等到下一个交易日的收盘，当天涨跌仍算旧组合。
只读本地行情。没有缓存的股票跳过，不在这里下载。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ah_pairs.backtest import sample_clock
from ah_pairs.costs import CostParams, rebalance_cost
from ah_pairs.data import assemble_pair, read_local_close
from ah_pairs.hedge import rolling_zscore
from ah_pairs.universe import a_to_yahoo, h_to_yahoo


def _fx_series(cache_dir=None) -> pd.Series | None:
    series = read_local_close("HKDCNY=X", cache_dir)
    if series is None or series.empty:
        return None
    median = float(series.median())
    if median > 2:
        series = 1.0 / series
    elif not (0.6 <= median <= 1.3):
        return None
    series.name = "fx"
    return series


def load_local_panels(universe: pd.DataFrame, start, end, cache_dir=None) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """返回 (代码 -> 行情表, 跳过的代码)。不访问网络。"""
    fx = _fx_series(cache_dir)
    if fx is None:
        return {}, ["FX"]
    panels: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for item in universe.itertuples(index=False):
        code = str(item.a_code).zfill(6)
        a_px = read_local_close(a_to_yahoo(code), cache_dir)
        h_px = read_local_close(h_to_yahoo(item.h_code), cache_dir)
        if a_px is None or h_px is None:
            skipped.append(code)
            continue
        try:
            prices, _notes = assemble_pair(a_px, h_px, fx, start, end)
        except Exception:
            skipped.append(code)
            continue
        panels[code] = prices
    return panels, skipped


def _summarize(index, returns: pd.Series) -> dict:
    clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = (1.0 + clean).cumprod()
    clock = sample_clock(index)
    years = clock["years"]
    ann = clock["bars_per_year"] if np.isfinite(clock["bars_per_year"]) and clock["bars_per_year"] > 0 else 252.0
    end_equity = float(equity.iloc[-1]) if len(equity) else np.nan
    vol = float(clean.std(ddof=1)) if len(clean) > 2 else np.nan
    sharpe = float(clean.mean() / vol * np.sqrt(ann)) if vol and vol > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": float(end_equity - 1.0) if np.isfinite(end_equity) else np.nan,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "n_days": int(len(clean)),
        "start": index.min(),
        "end": index.max(),
    }


def monthly_cheap_book(
    panels: dict[str, pd.DataFrame],
    z_window: int = 60,
    costs: CostParams | None = None,
    min_names: int = 8,
) -> dict:
    """A 溢价 z 最低的四分之一，等权做多 A、做空 H。"""
    costs = costs or CostParams()
    if len(panels) < min_names:
        return {"ok": False, "message": f"本地有行情的股票只有 {len(panels)} 只，不足 {min_names} 只。"}
    z_columns = {}
    ret_a = {}
    ret_h = {}
    for code, prices in panels.items():
        premium = prices["premium"]
        if "both_open" in prices.columns:
            premium = premium.where(prices["both_open"])
        # 单边开市日不放进滚动窗口，否则每隔几周的缺口会把 z 整段打成空值。
        usable = premium.dropna()
        z_columns[code] = pd.Series(rolling_zscore(usable, int(z_window)), index=usable.index)
        ret_a[code] = prices["ret_a"]
        ret_h[code] = prices["ret_h"]
    z_frame = pd.DataFrame(z_columns).sort_index()
    a_frame = pd.DataFrame(ret_a).reindex(z_frame.index)
    h_frame = pd.DataFrame(ret_h).reindex(z_frame.index)
    codes = list(z_frame.columns)
    picks: dict[pd.Timestamp, list[str]] = {}
    for _period, block in z_frame.groupby(z_frame.index.to_period("M")):
        chosen = None
        for day in reversed(block.index.unique()):
            row = block.loc[day]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            row = row.dropna()
            if len(row) >= min_names:
                chosen = row
                break
        if chosen is None:
            continue
        count = max(1, len(chosen) // 4)
        picks[pd.Timestamp(chosen.name).normalize()] = list(chosen.nsmallest(count).index)

    if not picks:
        return {"ok": False, "message": "没有任何一个月末同时有足够的溢价 z。"}

    n = len(z_frame)
    target = [[] for _ in range(n)]
    pending: list[str] = []
    held: list[str] = []
    signal_at = set(picks)
    for i, day in enumerate(z_frame.index):
        held = list(pending)
        target[i] = held
        key = pd.Timestamp(day).normalize()
        if key in signal_at:
            pending = list(picks[key])

    pnl = np.zeros(n)
    spent = 0.0
    rebalances = 0
    sizes = []
    for i, day in enumerate(z_frame.index):
        prev = [] if i == 0 else target[i - 1]
        current = target[i]
        names = set(prev) | set(current)
        if names and prev != current:
            rebalances += 1
        gross = 0.0
        cost = 0.0
        prev_n = max(len(prev), 1)
        cur_n = max(len(current), 1)
        for code in names:
            old_a = (0.5 / prev_n) if code in prev else 0.0
            old_h = (-0.5 / prev_n) if code in prev else 0.0
            new_a = (0.5 / cur_n) if code in current else 0.0
            new_h = (-0.5 / cur_n) if code in current else 0.0
            if i > 0:
                ra = a_frame.iloc[i][code]
                rh = h_frame.iloc[i][code]
                if np.isfinite(ra) and np.isfinite(rh):
                    gross += old_a * float(ra) + old_h * float(rh)
                if old_h < 0:
                    gross -= -old_h * costs.h_borrow_annual / costs.trading_days
            cost += rebalance_cost(old_a, old_h, new_a, new_h, day, costs)
        if current:
            sizes.append(len(current))
        spent += cost
        pnl[i] = gross - cost
    summary = _summarize(z_frame.index, pd.Series(pnl, index=z_frame.index))
    latest_day = max(picks)
    summary.update(
        {
            "ok": True,
            "n_names": int(len(codes)),
            "n_months": int(len(picks)),
            "mean_book": float(np.mean(sizes)) if sizes else np.nan,
            "rebalances": int(rebalances),
            "total_cost": float(spent),
            "latest_book": picks[latest_day],
            "latest_signal": latest_day,
            "z_window": int(z_window),
        }
    )
    return summary
