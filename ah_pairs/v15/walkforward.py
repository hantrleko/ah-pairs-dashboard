"""滚动样本外折，以及 White RC / Hansen SPA。

训练折只看到测试起点之前、再去掉 embargo 的共同开市日。
网格是 z_in × z_out × W_est。并列时取得成交更少的那组。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ah_pairs.backtest import sample_clock
from ah_pairs.cointegration import engle_granger
from ah_pairs.v15.adaptive import ar1_phi, execute, gate_open, half_life_from_phi
from ah_pairs.v15.net_premium import net_z

Z_IN_GRID = (1.5, 2.0, 2.5)
Z_OUT_GRID = (0.0, 0.5, 1.0)
W_EST_GRID = (120, 250)


def fold_bounds(dates, train: int = 756, test: int = 126, step: int = 126, embargo: int = 121) -> list[dict]:
    days = pd.DatetimeIndex(dates)
    folds = []
    cursor = int(train) + int(embargo)
    fold_id = 0
    while cursor + int(test) <= len(days):
        folds.append(
            {
                "fold_id": fold_id,
                "test_start": days[cursor],
                "test_end": days[cursor + int(test) - 1],
            }
        )
        fold_id += 1
        cursor += int(step)
    return folds


def _common_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if "both_open" not in frame.columns:
        return pd.DatetimeIndex(frame.index)
    mask = frame["both_open"].fillna(False).to_numpy(dtype=bool)
    return pd.DatetimeIndex(frame.index[mask])


def training_index(frame: pd.DataFrame, test_start, train: int = 756, embargo: int = 121) -> pd.DatetimeIndex:
    """测试起点之前的共同开市日，丢掉紧挨着测试折的 embargo。"""
    eligible = _common_index(frame)
    eligible = eligible[eligible < pd.Timestamp(test_start)]
    if int(embargo):
        if len(eligible) <= int(embargo):
            return eligible[:0]
        eligible = eligible[: -int(embargo)]
    if len(eligible) > int(train):
        eligible = eligible[-int(train) :]
    return eligible


def _beta_residual(log_a: np.ndarray, log_h: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    from ah_pairs.hedge import rolling_ols

    _alpha, beta = rolling_ols(log_a, log_h, int(window))
    beta = np.asarray(beta, dtype=np.float64)
    resid = np.asarray(log_a, dtype=np.float64) - beta * np.asarray(log_h, dtype=np.float64)
    return beta, resid


def prepare_frame(frame: pd.DataFrame, w_est: int, z_window: int = 60) -> tuple[pd.DataFrame, np.ndarray]:
    work = frame.copy()
    log_a = work["log_a"].to_numpy(dtype=np.float64) if "log_a" in work.columns else np.log(work["a_cny"].to_numpy(dtype=np.float64))
    log_h = work["log_h"].to_numpy(dtype=np.float64) if "log_h" in work.columns else np.log(work["h_cny"].to_numpy(dtype=np.float64))
    _beta, resid = _beta_residual(log_a, log_h, int(w_est))
    life = np.full(len(work), np.nan, dtype=np.float64)
    span = max(int(w_est), 40)
    for end in range(span, len(work)):
        life[end] = half_life_from_phi(ar1_phi(resid[end - span : end]))
    premium = log_a - log_h
    wedge = np.zeros(len(work), dtype=np.float64)
    z = net_z(premium, wedge, int(z_window)).to_numpy(dtype=np.float64)
    gate = np.array([gate_open(item, None)[0] if np.isfinite(item) else False for item in life], dtype=bool)
    work = work.copy()
    work["z_net"] = z
    work["half_life"] = life
    return work, gate


def _performance(net: np.ndarray, index, n_trades: int) -> dict:
    values = np.asarray(net, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    equity = np.cumprod(1.0 + values)
    total = float(equity[-1] - 1.0) if len(equity) else np.nan
    clock = sample_clock(index)
    ann = clock["bars_per_year"] if np.isfinite(clock["bars_per_year"]) and clock["bars_per_year"] > 0 else 252.0
    vol = float(np.std(values, ddof=1)) if len(values) > 2 else np.nan
    sharpe = float(np.mean(values) / vol * math.sqrt(ann)) if vol and vol > 0 else np.nan
    peak = np.maximum.accumulate(equity) if len(equity) else np.asarray([])
    drawdown = float(np.min(equity / peak - 1.0)) if len(equity) else np.nan
    return {"total_return": total, "sharpe": sharpe, "max_drawdown": drawdown, "n_trades": int(n_trades)}


def _trade_count(position: np.ndarray) -> int:
    if len(position) == 0:
        return 0
    changes = np.diff(position.astype(np.int64))
    entries = int(np.sum((position[1:] != 0) & (changes != 0)))
    if position[0] != 0:
        entries += 1
    return entries


def score_variant(frame: pd.DataFrame, z_in: float, z_out: float, w_est: int, allow_short_a: bool, z_window: int, prepared=None) -> dict:
    if prepared is None:
        prepared = prepare_frame(frame, w_est, z_window=z_window)
    work, gate = prepared
    result = execute(
        work,
        work["z_net"].to_numpy(dtype=np.float64),
        z_in,
        z_out,
        max_hold=60,
        allow_short_a=allow_short_a,
        gate=gate,
    )
    stats = _performance(result["net"], work.index, _trade_count(result["position"]))
    stats["n_blocked"] = result["n_blocked"]
    stats["blocked_share"] = result["blocked_share"]
    stats["net_path"] = result["net"]
    return stats


def select_params(
    frame: pd.DataFrame,
    *,
    inner_train: int = 500,
    inner_valid: int = 256,
    min_trades: int = 6,
    allow_short_a: bool = False,
    z_window: int = 40,
) -> dict:
    """只在传入的训练样本上选参数。调用方要先切掉测试折和 embargo。"""
    common = _common_index(frame)
    if len(common) < int(inner_train) + int(inner_valid):
        split = max(len(common) // 2, 1)
        valid_index = common[split:]
    else:
        valid_index = common[-int(inner_valid) :]
    best = None
    prepared_by_window = {}
    for w_est in W_EST_GRID:
        work, gate = prepare_frame(frame, w_est, z_window=z_window)
        positions = work.index.get_indexer(valid_index)
        positions = positions[positions >= 0]
        prepared_by_window[w_est] = (work.iloc[positions], gate[positions])
    for w_est in W_EST_GRID:
        for z_in in Z_IN_GRID:
            for z_out in Z_OUT_GRID:
                if z_out >= z_in:
                    continue
                stats = score_variant(
                    frame,
                    z_in,
                    z_out,
                    w_est,
                    allow_short_a,
                    z_window,
                    prepared=prepared_by_window[w_est],
                )
                qualified = stats["n_trades"] >= int(min_trades) and np.isfinite(stats["sharpe"])
                rank = (
                    0 if qualified else 1,
                    -stats["sharpe"] if np.isfinite(stats["sharpe"]) else 1e9,
                    stats["n_trades"],
                    z_in,
                    z_out,
                    w_est,
                )
                if best is None or rank < best[0]:
                    best = (rank, {"z_in": z_in, "z_out": z_out, "W_est": w_est, "qualified": qualified})
    if best is None:
        return {"z_in": 2.0, "z_out": 0.5, "W_est": 120, "qualified": False}
    return best[1]


def choose_for_test_start(
    frame: pd.DataFrame,
    test_start,
    *,
    train: int = 756,
    embargo: int = 121,
    allow_short_a: bool = False,
    inner_train: int = 500,
    inner_valid: int = 256,
    min_trades: int = 6,
    z_window: int = 40,
) -> dict:
    index = training_index(frame, test_start, train=train, embargo=embargo)
    if len(index) == 0:
        return {"z_in": 2.0, "z_out": 0.5, "W_est": 120, "qualified": False}
    return select_params(
        frame.loc[index],
        inner_train=inner_train,
        inner_valid=inner_valid,
        min_trades=min_trades,
        allow_short_a=allow_short_a,
        z_window=z_window,
    )


def sharpe_interval(returns, seed: int, n_boot: int = 200, block: int = 5) -> tuple[float, float]:
    values = np.asarray(returns, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    stats = np.empty(int(n_boot), dtype=np.float64)
    for draw in range(int(n_boot)):
        sample = _stationary_bootstrap(values.reshape(-1, 1), block, rng)[:, 0]
        vol = float(np.std(sample, ddof=1))
        stats[draw] = float(np.mean(sample) / vol * math.sqrt(252.0)) if vol > 0 else np.nan
    finite = stats[np.isfinite(stats)]
    if len(finite) < 10:
        return np.nan, np.nan
    return float(np.quantile(finite, 0.05)), float(np.quantile(finite, 0.95))


def politis_white_block(values) -> int:
    series = np.asarray(values, dtype=np.float64)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 10:
        return 5
    centered = series - float(np.mean(series))
    c0 = float(np.dot(centered, centered) / n)
    c1 = float(np.dot(centered[1:], centered[:-1]) / n)
    rho = 0.0 if c0 <= 0 else max(min(c1 / c0, 0.99), 0.0)
    if rho <= 0:
        return 5
    g = (6.0 * rho * rho) / ((1.0 - rho * rho) ** 2)
    length = (g ** (1.0 / 3.0)) * (n ** (1.0 / 3.0))
    return int(np.clip(round(length), 2, max(n // 5, 2)))


def _stationary_bootstrap(values: np.ndarray, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    n = len(data)
    probability = 1.0 / max(int(mean_block), 1)
    out = np.empty_like(data)
    cursor = int(rng.integers(0, n))
    for row in range(n):
        out[row] = data[cursor]
        if float(rng.random()) < probability:
            cursor = int(rng.integers(0, n))
        else:
            cursor = (cursor + 1) % n
    return out


def _max_finite(values) -> float:
    picked = np.asarray(values, dtype=np.float64)
    picked = picked[np.isfinite(picked)]
    if len(picked) == 0:
        return 0.0
    return float(np.max(picked))


def reality_check(returns: pd.DataFrame, seed: int, n_boot: int = 200, block: int | None = None) -> dict:
    """White RC，以及 SPA 的下界、一致、上界。同一种子两次结果相同。"""
    data = np.asarray(returns.to_numpy(dtype=np.float64), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    data = np.where(np.isfinite(data), data, 0.0)
    n_obs, _n_models = data.shape
    means = data.mean(axis=0)
    stat = float(np.max(means))
    length = int(block) if block else politis_white_block(data[:, int(np.argmax(means))])
    rng = np.random.default_rng(int(seed))
    centered = data - means
    hits = 0
    spa_hits = {"lower": 0, "consistent": 0, "upper": 0}
    std = data.std(axis=0, ddof=1)
    std = np.where(std > 1e-12, std, np.nan)
    student = np.sqrt(n_obs) * means / std
    observed = _max_finite(student)
    threshold = math.sqrt(2.0 * math.log(math.log(max(n_obs, 16))))
    recenter = {
        "lower": np.zeros_like(means),
        "upper": means,
        "consistent": np.where(means >= -threshold * (std / math.sqrt(n_obs)), means, 0.0),
    }
    for _draw in range(int(n_boot)):
        sample = _stationary_bootstrap(centered, length, rng)
        if float(np.max(sample.mean(axis=0))) >= stat - 1e-15:
            hits += 1
        boot_mean = sample.mean(axis=0)
        boot_std = sample.std(axis=0, ddof=1)
        boot_std = np.where(boot_std > 1e-12, boot_std, np.nan)
        for name, center in recenter.items():
            t_boot = np.sqrt(n_obs) * (boot_mean + center) / boot_std
            if _max_finite(t_boot) >= observed - 1e-12:
                spa_hits[name] += 1
    scale = int(n_boot) + 1
    return {
        "rc_p": float((hits + 1) / scale),
        "spa_p": float((spa_hits["consistent"] + 1) / scale),
        "spa_p_lower": float((spa_hits["lower"] + 1) / scale),
        "spa_p_upper": float((spa_hits["upper"] + 1) / scale),
        "block": length,
        "seed": int(seed),
        "n_boot": int(n_boot),
    }


def sensitivity_rc(returns: pd.DataFrame, seed: int, lengths=(5, 10, 20), n_boot: int = 100) -> dict[str, float]:
    out = {}
    for length in lengths:
        checked = reality_check(returns, seed=seed, n_boot=n_boot, block=int(length))
        out[f"rc_p_l{int(length)}"] = checked["rc_p"]
    return out


def flag_implausible(sharpe: float) -> bool:
    return bool(np.isfinite(sharpe) and float(sharpe) > 2.0)


def _eg_at(frame: pd.DataFrame) -> tuple[float, float]:
    common = frame.loc[frame["both_open"]] if "both_open" in frame.columns else frame
    if len(common) < 40 or "log_a" not in common.columns:
        return np.nan, np.nan
    try:
        eg = engle_granger(common["log_a"], common["log_h"])
    except Exception:
        return np.nan, np.nan
    life = half_life_from_phi(ar1_phi(eg["resid"].to_numpy(dtype=np.float64)))
    return float(eg["pvalue"]), float(life)


def walk_forward(
    frame: pd.DataFrame,
    *,
    train: int = 756,
    test: int = 126,
    step: int = 126,
    embargo: int = 121,
    allow_short_a: bool = False,
    seed: int = 7,
    inner_train: int = 500,
    inner_valid: int = 256,
    min_trades: int = 6,
    z_window: int = 40,
    n_boot: int = 100,
) -> dict:
    common = _common_index(frame)
    bounds = fold_bounds(common, train=train, test=test, step=step, embargo=embargo)
    rows = []
    variant_paths = []
    oos_indexes: list[pd.DatetimeIndex] = []
    blocked_n = 0
    signal_n = 0
    for fold in bounds:
        chosen = choose_for_test_start(
            frame,
            fold["test_start"],
            train=train,
            embargo=embargo,
            allow_short_a=allow_short_a,
            inner_train=inner_train,
            inner_valid=inner_valid,
            min_trades=min_trades,
            z_window=z_window,
        )
        test_index = common[(common >= fold["test_start"]) & (common <= fold["test_end"])]
        train_index = training_index(frame, fold["test_start"], train=train, embargo=embargo)
        history = train_index[-max(W_EST_GRID) :]
        span_index = history.append(test_index)
        span = frame.loc[span_index]
        test_frame = frame.loc[test_index]
        eg_p, hl = _eg_at(frame.loc[train_index]) if len(train_index) else (np.nan, np.nan)
        paths = {}
        chosen_stats = None
        for w_est in W_EST_GRID:
            prepared, gate = prepare_frame(span, w_est, z_window=z_window)
            positions = prepared.index.get_indexer(test_index)
            positions = positions[positions >= 0]
            test_work = prepared.iloc[positions]
            test_gate = gate[positions]
            if w_est == chosen["W_est"] and np.isfinite(eg_p) and eg_p > 0.10:
                test_gate = np.zeros(len(test_work), dtype=bool)
            for z_in in Z_IN_GRID:
                for z_out in Z_OUT_GRID:
                    if z_out >= z_in:
                        continue
                    result = execute(
                        test_work,
                        test_work["z_net"].to_numpy(dtype=np.float64),
                        z_in,
                        z_out,
                        max_hold=60,
                        allow_short_a=allow_short_a,
                        gate=test_gate,
                    )
                    key = (z_in, z_out, w_est)
                    paths[key] = result["net"]
                    if key == (chosen["z_in"], chosen["z_out"], chosen["W_est"]):
                        chosen_stats = (result, eg_p, hl)
        if chosen_stats is None:
            continue
        result, eg_p, hl = chosen_stats
        stats = _performance(result["net"], test_frame.index, _trade_count(result["position"]))
        lo, hi = sharpe_interval(result["net"], seed=int(seed) + int(fold["fold_id"]) * 17, n_boot=n_boot)
        matrix = pd.DataFrame({f"{a}|{b}|{c}": path for (a, b, c), path in paths.items()})
        checked = reality_check(matrix, seed=int(seed) + int(fold["fold_id"]), n_boot=n_boot)
        sensitive = sensitivity_rc(matrix, seed=int(seed) + int(fold["fold_id"]), n_boot=max(n_boot // 2, 20))
        implausible = flag_implausible(stats["sharpe"])
        rows.append(
            {
                "fold_id": int(fold["fold_id"]),
                "test_start": pd.Timestamp(fold["test_start"]),
                "test_end": pd.Timestamp(fold["test_end"]),
                "z_in": float(chosen["z_in"]),
                "z_out": float(chosen["z_out"]),
                "W_est": int(chosen["W_est"]),
                "oos_ret": stats["total_return"],
                "oos_sharpe": stats["sharpe"],
                "oos_dd": stats["max_drawdown"],
                "n_trades": stats["n_trades"],
                "n_blocked": int(result["n_blocked"]),
                "blocked_share": result["blocked_share"],
                "oos_sharpe_ci_lo": lo,
                "oos_sharpe_ci_hi": hi,
                "eg_p_at_fold_start": eg_p,
                "hl_at_fold_start": hl,
                "rc_p": checked["rc_p"],
                "spa_p": checked["spa_p"],
                "spa_p_lower": checked["spa_p_lower"],
                "spa_p_upper": checked["spa_p_upper"],
                "implausible": implausible,
                **sensitive,
            }
        )
        variant_paths.append(result["net"])
        oos_indexes.append(pd.DatetimeIndex(test_frame.index))
        blocked_n += int(result["n_blocked"])
        signal_n += int(result["n_short_a_signals"])
    table = pd.DataFrame(rows)
    stitched = np.concatenate(variant_paths) if variant_paths else np.asarray([], dtype=np.float64)
    if oos_indexes:
        stitched_index = oos_indexes[0]
        for extra in oos_indexes[1:]:
            stitched_index = stitched_index.append(extra)
    else:
        stitched_index = pd.DatetimeIndex([])
    summary = _performance(stitched, stitched_index, int(table["n_trades"].sum()) if len(table) else 0)
    blocked = float(blocked_n / signal_n) if signal_n else np.nan
    return {
        "folds": table,
        "n_folds": int(len(table)),
        "folds_insufficient": int(len(table)) < 4,
        "oos": summary,
        "blocked_share": blocked,
        "seed": int(seed),
        "engine": "v15",
    }


def write_folds(table: pd.DataFrame, path) -> None:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(target, index=False)
