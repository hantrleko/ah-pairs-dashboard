"""默认规格的 walk-forward 和稳定性地图。

只保存侧栏初次打开时的那一套。页面的结束日可以比样本截止日晚最多 10 天，并写明这一天。
更早的结束日、对冲、窗口、开平仓、成本或起始日不同时，不拿这份结果冒充。
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
import json
import math
from pathlib import Path

import pandas as pd

from ah_pairs.backtest import StrategyParams
from ah_pairs.eflab_calendar import END_SLACK_DAYS
from ah_pairs.paths import DESK_DIR
from ah_pairs.stability import fragility_report, length_summary

# 和侧栏四个快捷按钮同一组。
DESK_A_CODES = ("601318", "601398", "600036", "601857")
_SIGNAL_FIELDS = ("hedge", "hedge_window", "z_window", "kalman_delta", "kalman_delta_auto")


def default_desk_params() -> StrategyParams:
    """侧栏第一次打开、还没改过的规格。卡尔曼 δ 按预热样本估计，和页面默认一致。"""
    return StrategyParams(kalman_delta_auto=True)


def write_result(
    a_code: str,
    params: StrategyParams,
    start,
    end,
    sample_end,
    walk: dict,
    grid: pd.DataFrame,
    checks: pd.DataFrame,
    include_costs: bool = True,
    desk_dir: Path | None = None,
) -> Path:
    """先写表，最后写 meta。读的时候看到 meta，就说明表已经写完。"""
    if not walk.get("ok"):
        raise ValueError(walk.get("message") or "walk-forward 没有结果，不写入预计算")
    root = (desk_dir or DESK_DIR) / str(a_code).zfill(6)
    root.mkdir(parents=True, exist_ok=True)
    walk["table"].to_parquet(root / "walk_table.parquet")
    oos = pd.concat(
        [
            walk["oos_returns"].rename("oos_return"),
            walk["oos_equity"].rename("oos_equity"),
        ],
        axis=1,
    )
    oos.to_parquet(root / "oos.parquet")
    grid.to_parquet(root / "grid.parquet")
    checks.to_parquet(root / "checks.parquet")
    meta = {
        "a_code": str(a_code).zfill(6),
        "start": pd.Timestamp(start).date().isoformat(),
        "end": pd.Timestamp(end).date().isoformat(),
        "asof": pd.Timestamp.today().date().isoformat(),
        "sample_end": pd.Timestamp(sample_end).date().isoformat(),
        "include_costs": bool(include_costs),
        "params": json.loads(json.dumps(asdict(params))),
        "walk": _walk_meta(walk),
    }
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return root


def load_walk(page, desk_dir: Path | None = None) -> dict | None:
    """规格一致时返回和 walk_forward 相同的字典，并带上 asof、sample_end。"""
    meta = _meta_for(page, desk_dir)
    if meta is None or not _walk_matches(meta, page):
        return None
    root = _root(page, desk_dir)
    try:
        table = pd.read_parquet(root / "walk_table.parquet")
        oos = pd.read_parquet(root / "oos.parquet")
    except (OSError, ValueError):
        return None
    walk = dict(meta["walk"])
    walk["ok"] = True
    walk["table"] = table
    walk["oos_returns"] = _column(oos, "oos_return")
    walk["oos_equity"] = _column(oos, "oos_equity")
    walk["asof"] = meta["asof"]
    walk["sample_end"] = meta["sample_end"]
    walk["precomputed"] = True
    for key in ("train_end", "valid_start", "valid_end"):
        if walk.get(key):
            walk[key] = pd.Timestamp(walk[key])
    return walk


def load_stability(page, desk_dir: Path | None = None) -> dict | None:
    """稳定性地图只跟价格样本和对冲有关，不跟开平仓阈值走。"""
    meta = _meta_for(page, desk_dir)
    if meta is None or not _stability_matches(meta, page):
        return None
    root = _root(page, desk_dir)
    try:
        grid = pd.read_parquet(root / "grid.parquet")
        checks = pd.read_parquet(root / "checks.parquet")
    except (OSError, ValueError):
        return None
    report = fragility_report(grid)
    return {
        "grid": grid,
        "report": report,
        "by_length": length_summary(grid),
        "checks": checks,
        "asof": meta["asof"],
        "sample_end": meta["sample_end"],
        "precomputed": True,
    }


def _walk_meta(walk: dict) -> dict:
    return {
        "scheme": walk.get("scheme"),
        "chosen_entry": _num(walk.get("chosen_entry")),
        "chosen_exit": _num(walk.get("chosen_exit")),
        "chosen_entries": [float(item) for item in walk.get("chosen_entries", [])],
        "chosen_exits": [float(item) for item in walk.get("chosen_exits", [])],
        "n_folds": int(walk.get("n_folds") or 0),
        "n_trials": int(walk.get("n_trials") or 0),
        "valid_sharpe_median": _num(walk.get("valid_sharpe_median")),
        "valid_sharpe_min": _num(walk.get("valid_sharpe_min")),
        "valid_sharpe_max": _num(walk.get("valid_sharpe_max")),
        "train_end": _iso(walk.get("train_end")),
        "valid_start": _iso(walk.get("valid_start")),
        "valid_end": _iso(walk.get("valid_end")),
    }


def _meta_for(page, desk_dir: Path | None) -> dict | None:
    path = _root(page, desk_dir) / "meta.json"
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict) or "params" not in meta:
        return None
    return meta


def _root(page, desk_dir: Path | None) -> Path:
    code = str(getattr(page.picked, "a_code")).zfill(6)
    return (desk_dir or DESK_DIR) / code


def _walk_matches(meta: dict, page) -> bool:
    if not _dates_ok(meta, page) or not _signal_ok(meta, page):
        return False
    if bool(getattr(page, "include_costs", True)) != bool(meta.get("include_costs", True)):
        return False
    return _close(_live_params(page), meta.get("params"))


def _stability_matches(meta: dict, page) -> bool:
    return _dates_ok(meta, page) and _signal_ok(meta, page)


def _dates_ok(meta: dict, page) -> bool:
    try:
        start = pd.Timestamp(page.start_day).date()
        end = pd.Timestamp(page.end_day).date()
        stored_start = date.fromisoformat(meta["start"])
        stored_end = date.fromisoformat(meta["sample_end"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if start != stored_start:
        return False
    # 快照里的行情不能晚于用户的结束日。只允许页面结束日比样本截止日新最多 10 天。
    return 0 <= (end - stored_end).days <= END_SLACK_DAYS


def _signal_ok(meta: dict, page) -> bool:
    stored = meta.get("params") or {}
    live = _live_params(page)
    return all(_close(live.get(key), stored.get(key)) for key in _SIGNAL_FIELDS)


def _live_params(page) -> dict:
    return json.loads(json.dumps(asdict(page.params)))


def _close(left, right) -> bool:
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            return False
        return all(_close(left[key], right[key]) for key in left)
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            return False
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(dtype=float, name=name)
    series = frame[name].copy()
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).normalize()
    series.name = name
    return series


def _num(value):
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _iso(value) -> str | None:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    return stamp.date().isoformat()
