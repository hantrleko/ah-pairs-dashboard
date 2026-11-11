"""把行情、协整、对冲和回测串成一次分析。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ah_pairs.backtest import StrategyParams, run_backtest
from ah_pairs.cointegration import (
    engle_granger,
    johansen_test,
    ou_half_life,
    residual_adf,
    rolling_eg_pvalue,
    select_vecm_lags,
)
from ah_pairs.hedge import estimate_kalman_delta, kalman_hedge, rolling_ols, rolling_zscore


def warmup_start(start, params: StrategyParams) -> pd.Timestamp:
    """评估区间之前多取的自然日，用来准备 beta 和 z。"""
    days = int((params.hedge_window + params.z_window) * 2.2) + 20
    return pd.Timestamp(start).normalize() - pd.Timedelta(days=days)


def _common(prices: pd.DataFrame) -> pd.DataFrame:
    if "both_open" in prices.columns:
        return prices.loc[prices["both_open"].fillna(False)].copy()
    return prices


def _align(values, source_index, target_index, both_open=None) -> pd.Series:
    aligned = pd.Series(np.asarray(values, dtype=float), index=source_index).reindex(target_index)
    if both_open is None:
        return aligned
    open_day = pd.Series(both_open, index=target_index).fillna(False).astype(bool)
    return aligned.where(open_day, aligned.ffill())


def build_signals(prices: pd.DataFrame, params: StrategyParams, alt: bool = True) -> pd.DataFrame:
    """alt 为假时只算当前选中的对冲。另一条 β 留给对照开关，避免每次打开页面都跑卡尔曼。"""
    base = _common(prices)
    y = base["log_a"].to_numpy(dtype=float)
    x = base["log_h"].to_numpy(dtype=float)
    need_roll = alt or params.hedge == "rolling"
    need_kalman = alt or params.hedge == "kalman"
    if need_roll:
        alpha_roll, beta_roll = rolling_ols(y, x, params.hedge_window)
    else:
        alpha_roll = beta_roll = np.full(len(base), np.nan)
    if need_kalman:
        alpha_kalman, beta_kalman = kalman_hedge(
            y, x, delta=params.kalman_delta, init_window=params.hedge_window
        )
    else:
        alpha_kalman = beta_kalman = np.full(len(base), np.nan)
    if params.hedge == "kalman":
        alpha, beta = alpha_kalman, beta_kalman
    else:
        alpha, beta = alpha_roll, beta_roll
    spread = y - alpha - beta * x
    both = prices["both_open"] if "both_open" in prices.columns else None
    out = prices.copy()
    out["alpha"] = _align(alpha, base.index, out.index, both)
    out["beta"] = _align(beta, base.index, out.index, both)
    if need_roll:
        out["beta_rolling"] = _align(beta_roll, base.index, out.index, both)
    if need_kalman:
        out["beta_kalman"] = _align(beta_kalman, base.index, out.index, both)
    out["spread"] = _align(spread, base.index, out.index, both)
    out["z"] = _align(rolling_zscore(spread, params.z_window), base.index, out.index, both)
    out["spread_beta1"] = prices["log_a"] - prices["log_h"]
    return out


def _resolve_kalman(prices: pd.DataFrame, params: StrategyParams, eval_start) -> tuple[StrategyParams, dict]:
    manual = {
        "auto": False,
        "ok": True,
        "delta": float(params.kalman_delta),
        "in_sample": False,
        "n_obs": 0,
        "reason": "sidebar",
        "message": "δ 用侧栏里的数。",
    }
    if params.hedge != "kalman" or not params.kalman_delta_auto:
        return params, manual
    sample = _common(prices)
    pre = sample
    in_sample = eval_start is None
    if eval_start is not None:
        pre = sample.loc[sample.index < pd.Timestamp(eval_start).normalize()]
    if len(pre) < params.hedge_window + 40:
        return params, {
            "auto": True,
            "ok": False,
            "delta": float(params.kalman_delta),
            "in_sample": True,
            "n_obs": int(len(pre)),
            "reason": "warmup_short",
            "message": "评估区间之前的预热不够长，δ 仍用侧栏里的数。",
        }
    fit = estimate_kalman_delta(pre["log_a"], pre["log_h"], init_window=params.hedge_window)
    fit["auto"] = True
    fit["in_sample"] = bool(in_sample)
    fit["reason"] = "in_sample" if in_sample else "warmup"
    fit["message"] = (
        "没有单独的预热区间，δ 用了评估样本本身，这一项不是样本外。"
        if in_sample
        else "δ 用评估区间之前的预热样本，按预测误差似然在对数网格上选取。"
    )
    return replace(params, kalman_delta=float(fit["delta"])), fit


def analyze_frame(
    prices: pd.DataFrame,
    params: StrategyParams,
    eval_start=None,
    northbound: pd.Series | None = None,
    full_diagnostics: bool = True,
    alt_hedge: bool = True,
    with_backtest: bool = True,
) -> dict:
    """prices 可以包含评估区间之前的预热样本。

    协整检验和回测净值只使用 eval_start 及之后的日期。
    预热样本只用来生成评估区间第一天就已经可用的 beta 和 z。
    卡尔曼 δ 若自动估计，也只用这段预热。
    """
    params, kalman_fit = _resolve_kalman(prices, params, eval_start)
    signals = build_signals(prices, params, alt=alt_hedge)
    if eval_start is not None:
        start = pd.Timestamp(eval_start).normalize()
        view = signals.loc[signals.index >= start].copy()
    else:
        view = signals.copy()
    if view.empty:
        raise ValueError("评估区间里没有数据")
    diag = _common(view)
    usable = int(np.isfinite(diag["z"].to_numpy(dtype=float)).sum()) if len(diag) else 0
    if usable < 30:
        raise ValueError(
            f"有效 z-score 只有 {usable} 天。把起始日往前移，或缩短估计窗口。"
        )

    eg = engle_granger(diag["log_a"], diag["log_h"])
    try:
        adf = residual_adf(eg["resid"])
    except Exception as exc:
        adf = {"ok": False, "message": str(exc)}
    else:
        adf["ok"] = True
    try:
        lags = select_vecm_lags(diag["log_a"], diag["log_h"])
        johansen = johansen_test(diag["log_a"], diag["log_h"], k_ar_diff=lags)
    except Exception as exc:
        johansen = {"ok": False, "message": str(exc)}
    else:
        johansen["ok"] = True

    static_hl = ou_half_life(eg["resid"])
    traded = diag["spread"].replace([np.inf, -np.inf], np.nan).dropna()
    traded_hl = ou_half_life(traded) if len(traded) >= 30 else None
    view["static_resid"] = eg["resid"].reindex(view.index)

    if not with_backtest:
        if northbound is not None and not northbound.empty:
            view["northbound"] = northbound.reindex(view.index)
        rolling_p = pd.Series(dtype=float, name="eg_pvalue")
        if full_diagnostics:
            try:
                rolling_p = rolling_eg_pvalue(diag["log_a"], diag["log_h"], window=252, step=42)
            except Exception:
                rolling_p = pd.Series(dtype=float, name="eg_pvalue")
        return {
            "frame": view,
            "trades": pd.DataFrame(),
            "stats": {},
            "eg": {k: v for k, v in eg.items() if k != "resid"},
            "residual_adf": adf,
            "half_life_static": static_hl,
            "half_life_traded": traded_hl,
            "johansen": johansen,
            "northbound": None,
            "rolling_p": rolling_p,
            "sensitivity": pd.DataFrame(),
            "params": params,
            "kalman_fit": kalman_fit,
            "n_common": int(len(diag)),
            "n_one_sided": int((~view["both_open"].fillna(False)).sum()) if "both_open" in view.columns else 0,
        }

    result, trades, stats = run_backtest(view, params)
    north = None
    if northbound is not None and not northbound.empty:
        from ah_pairs.northbound import describe_northbound

        north = describe_northbound(diag["spread"], northbound)
        result["northbound"] = northbound.reindex(result.index)

    if full_diagnostics:
        try:
            rolling_p = rolling_eg_pvalue(diag["log_a"], diag["log_h"], window=252, step=42)
        except Exception:
            rolling_p = pd.Series(dtype=float, name="eg_pvalue")
        sensitivity = entry_sensitivity(result, params)
    else:
        rolling_p = pd.Series(dtype=float, name="eg_pvalue")
        sensitivity = pd.DataFrame()

    return {
        "frame": result,
        "trades": trades,
        "stats": stats,
        "eg": {k: v for k, v in eg.items() if k != "resid"},
        "residual_adf": adf,
        "half_life_static": static_hl,
        "half_life_traded": traded_hl,
        "johansen": johansen,
        "northbound": north,
        "rolling_p": rolling_p,
        "sensitivity": sensitivity,
        "params": params,
        "kalman_fit": kalman_fit,
        "n_common": int(len(diag)),
        "n_one_sided": int((~view["both_open"].fillna(False)).sum()) if "both_open" in view.columns else 0,
    }


def entry_sensitivity(
    frame: pd.DataFrame,
    params: StrategyParams,
    entries: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0),
    short_a_days: np.ndarray | None = None,
) -> pd.DataFrame:
    """同一条 z 和 beta 上，只改开仓阈值，比较夏普和回撤。"""
    rows = []
    for entry in entries:
        if entry <= params.exit_z:
            continue
        stop = params.stop_z
        if stop != 0 and stop <= entry:
            stop = round(entry + 0.5, 1)
        try:
            trial = replace(params, entry_z=float(entry), stop_z=float(stop))
            _path, _trades, stats = run_backtest(frame, trial, short_a_days=short_a_days)
        except Exception:
            continue
        rows.append(
            {
                "entry_z": float(entry),
                "sharpe": stats["sharpe"],
                "max_drawdown": stats["max_drawdown"],
                "total_return": stats["total_return"],
                "n_trades": stats["n_trades"],
                "current": abs(float(entry) - float(params.entry_z)) < 1e-9,
            }
        )
    return pd.DataFrame(rows)


def walk_forward(
    frame: pd.DataFrame,
    params: StrategyParams,
    entries: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0),
    exits: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict:
    """锚定式分折。样本跨过三个日历年时，按年扩展训练、下一年验证。

    每一折只在训练段里同时挑选开仓和平仓阈值，验证段锁定，不再调参。
    样本太短时退回一次对半切分，行名仍叫训练（选出参数）和验证（锁定参数）。
    """
    if len(frame) < 120 or "z" not in frame.columns:
        return {"ok": False, "message": "样本太短，无法分成训练区间和验证区间。"}
    folds, scheme = _walk_folds(frame)
    if not folds:
        return {"ok": False, "message": "样本太短，无法分成训练区间和验证区间。"}
    pieces = []
    valid_sharpes = []
    chosen_entries = []
    chosen_exits = []
    n_trials = 0
    last_locked = params
    oos_parts = []
    for train, valid, label in folds:
        locked, tried = _select_thresholds(train, params, entries, exits)
        n_trials += tried
        if locked is None:
            locked = params
        last_locked = locked
        chosen_entries.append(float(locked.entry_z))
        chosen_exits.append(float(locked.exit_z))
        train_name = "训练（选出参数）" if scheme == "half" else f"训练 {label}"
        valid_name = "验证（锁定参数）" if scheme == "half" else f"验证 {label}"
        train_role = "train_half" if scheme == "half" else "train"
        valid_role = "valid_half" if scheme == "half" else "valid"
        fold_id = "" if scheme == "half" else label
        for name, sample, trial, role in (
            (train_name, train, locked, train_role),
            (valid_name, valid, locked, valid_role),
        ):
            path, _trades, stats = run_backtest(sample, trial)
            pieces.append(_wf_row(name, trial, stats, role, fold_id))
            if role.startswith("valid"):
                valid_sharpes.append(stats["sharpe"])
                oos_parts.append(path["ret_strat"])
    if scheme == "half":
        for name, trial, role in (
            ("全样本（锁定参数）", last_locked, "full_locked"),
            ("全样本（侧栏当前参数）", params, "full_sidebar"),
        ):
            _path, _trades, stats = run_backtest(frame, trial)
            pieces.append(_wf_row(name, trial, stats, role, ""))
    else:
        _path, _trades, stats = run_backtest(frame, params)
        pieces.append(_wf_row("全样本（侧栏当前参数）", params, stats, "full_sidebar", ""))
    finite = [float(item) for item in valid_sharpes if np.isfinite(item)]
    train = folds[0][0]
    valid = folds[-1][1]
    if oos_parts:
        oos = pd.concat(oos_parts)
        oos = oos[~oos.index.duplicated(keep="first")].sort_index()
        oos_equity = (1.0 + oos.fillna(0.0)).cumprod()
        oos_equity.name = "oos_equity"
    else:
        oos = pd.Series(dtype=float, name="oos_return")
        oos_equity = pd.Series(dtype=float, name="oos_equity")
    return {
        "ok": True,
        "scheme": scheme,
        "chosen_entry": float(chosen_entries[0]),
        "chosen_exit": float(chosen_exits[0]),
        "chosen_entries": chosen_entries,
        "chosen_exits": chosen_exits,
        "n_folds": int(len(folds)),
        "n_trials": int(n_trials),
        "valid_sharpe_median": float(np.median(finite)) if finite else np.nan,
        "valid_sharpe_min": float(np.min(finite)) if finite else np.nan,
        "valid_sharpe_max": float(np.max(finite)) if finite else np.nan,
        "train_end": train.index.max(),
        "valid_start": valid.index.min(),
        "valid_end": valid.index.max(),
        "table": pd.DataFrame(pieces),
        "oos_returns": oos,
        "oos_equity": oos_equity,
    }


def _walk_folds(frame: pd.DataFrame) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame, str]], str]:
    years = sorted(set(int(year) for year in frame.index.year))
    folds = []
    if len(years) >= 3 and len(frame) >= 400:
        for year in years[1:]:
            train = frame.loc[frame.index.year < year]
            valid = frame.loc[frame.index.year == year]
            if len(train) >= 120 and len(valid) >= 40:
                folds.append((train, valid, str(year)))
    if folds:
        return folds, "anchored-year"
    cut = len(frame) // 2
    return [(frame.iloc[:cut], frame.iloc[cut:], "后半段")], "half"


def _select_thresholds(train, params, entries, exits):
    best = None
    best_sharpe = -np.inf
    tried = 0
    for entry in entries:
        for exit_ in exits:
            if not (0 <= float(exit_) < float(entry)):
                continue
            stop = float(params.stop_z)
            if stop != 0 and stop <= float(entry):
                stop = round(float(entry) + 0.5, 1)
            trial = replace(
                params,
                entry_z=float(entry),
                exit_z=float(exit_),
                stop_z=float(stop),
            )
            try:
                _path, _trades, stats = run_backtest(train, trial)
            except Exception:
                continue
            tried += 1
            sharpe = stats["sharpe"]
            if np.isfinite(sharpe) and sharpe > best_sharpe:
                best_sharpe = float(sharpe)
                best = trial
    return best, tried


def _wf_row(name: str, trial: StrategyParams, stats: dict, role: str = "", fold: str = "") -> dict:
    return {
        "区间": name,
        "role": role,
        "fold": fold,
        "开仓 |z|": float(trial.entry_z),
        "平仓 |z|": float(trial.exit_z),
        "净收益": stats["total_return"],
        "夏普": stats["sharpe"],
        "最大回撤": stats["max_drawdown"],
        "交易次数": stats["n_trades"],
        "成本合计": stats["total_cost"],
    }


def pick_lowest_p(table: pd.DataFrame) -> pd.Series | None:
    """在一组对照里挑 Engle-Granger p 值最小的一行。"""
    if table.empty or "eg_pvalue" not in table.columns:
        return None
    valid = table.dropna(subset=["eg_pvalue"])
    if valid.empty:
        return None
    return valid.loc[valid["eg_pvalue"].idxmin()]


def pick_best_sharpe(table: pd.DataFrame) -> pd.Series | None:
    """在一组对照里挑扣费后夏普最高的一行。"""
    if table.empty or "sharpe" not in table.columns:
        return None
    valid = table.dropna(subset=["sharpe"])
    if valid.empty:
        return None
    return valid.loc[valid["sharpe"].idxmax()]


def screen_one(prices: pd.DataFrame, params: StrategyParams, eval_start) -> dict:
    result = analyze_frame(prices, params, eval_start=eval_start, full_diagnostics=False)
    eg = result["eg"]
    hl = result["half_life_static"]
    stats = result["stats"]
    premium = result["frame"]["premium"].dropna()
    return {
        "nobs": eg["nobs"],
        "eg_pvalue": eg["pvalue"],
        "eg_beta": eg["beta"],
        "eg_t": eg["t_stat"],
        "half_life": hl["half_life"],
        "sharpe": stats["sharpe"],
        "cagr": stats["cagr"],
        "total_return": stats["total_return"],
        "max_drawdown": stats["max_drawdown"],
        "n_trades": stats["n_trades"],
        "total_cost": stats["total_cost"],
        "latest_premium": float(premium.iloc[-1]) if len(premium) else np.nan,
    }
