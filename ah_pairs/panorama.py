"""流动性子集的最新溢价、z 和持仓。

打开页面只读已经写好的表。重算只读本地行情，不下载，也不跑协整。
只有默认规格才写入快照。侧栏改过参数，或结束日早于样本截止日时，不拿这份表冒充。
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
import json
from pathlib import Path

import pandas as pd

from ah_pairs.analysis import build_signals, warmup_start
from ah_pairs.backtest import StrategyParams, current_state, run_backtest
from ah_pairs.data import assemble_pair, read_local_close
from ah_pairs.desk import END_SLACK_DAYS, _close, default_desk_params
from ah_pairs.paths import CACHE_DIR, PUBLISHED_DIR
from ah_pairs.store import prefer, publish_file, same_dir
from ah_pairs.universe import a_to_yahoo, h_to_yahoo

TABLE_NAME = "panorama.parquet"
META_NAME = "panorama.json"


def _fx_series(cache_dir: Path | None) -> pd.Series | None:
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


def _params_dict(params: StrategyParams) -> dict:
    return json.loads(json.dumps(asdict(params), sort_keys=True))


def is_default_spec(params: StrategyParams, start) -> bool:
    """定时任务那一套：2018-01-01 起，侧栏第一次打开的参数。"""
    try:
        start_day = pd.Timestamp(start).date()
    except (TypeError, ValueError):
        return False
    return start_day == date(2018, 1, 1) and _close(_params_dict(params), _params_dict(default_desk_params()))


def _one_row(item: dict, fx: pd.Series, start, end, params: StrategyParams, cache_dir: Path | None) -> dict | None:
    code = str(item["a_code"]).zfill(6)
    h_code = str(item["h_code"]).zfill(5)
    a_px = read_local_close(a_to_yahoo(code), cache_dir)
    h_px = read_local_close(h_to_yahoo(h_code), cache_dir)
    if a_px is None or h_px is None:
        return None
    warm = warmup_start(start, params)
    prices, _notes = assemble_pair(a_px, h_px, fx, warm, end)
    signals = build_signals(prices, params, alt=False)
    view = signals.loc[signals.index >= pd.Timestamp(start).normalize()]
    if view.empty:
        return None
    result, trades, _stats = run_backtest(view, params)
    state = current_state(result, trades, params.fill)
    common = result.loc[result["both_open"]] if "both_open" in result.columns else result
    premium = common["premium"].dropna()
    z_values = common["z"].dropna()
    if premium.empty or z_values.empty:
        return None
    asof = pd.Timestamp(z_values.index[-1]).date().isoformat()
    return {
        "a_code": code,
        "h_code": h_code,
        "name": str(item.get("name") or code),
        "asof": asof,
        "premium": float(premium.iloc[-1]),
        "z": float(z_values.iloc[-1]),
        "position": int(state.get("position") or 0),
        "signal": int(state.get("signal") or 0),
        "pending": bool(state.get("pending")),
    }


def build_panorama(
    universe: pd.DataFrame,
    start,
    end,
    params: StrategyParams,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """只读本地缓存。没有文件的股票跳过，不访问网络。"""
    fx = _fx_series(cache_dir)
    skipped: list[str] = []
    if fx is None:
        return _empty(), _meta(start, None, params, ["FX"])
    rows = []
    for item in universe.to_dict(orient="records"):
        code = str(item["a_code"]).zfill(6)
        try:
            row = _one_row(item, fx, start, end, params, cache_dir)
        except Exception:
            row = None
        if row is None:
            skipped.append(code)
        else:
            rows.append(row)
    table = pd.DataFrame(rows) if rows else _empty()
    if not table.empty:
        table = table.sort_values("z", ascending=True, na_position="last").reset_index(drop=True)
    sample_end = None if table.empty else str(table["asof"].max())
    return table, _meta(start, sample_end, params, skipped)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["a_code", "h_code", "name", "asof", "premium", "z", "position", "signal", "pending"]
    )


def _meta(start, sample_end: str | None, params: StrategyParams, skipped: list[str]) -> dict:
    try:
        start_text = pd.Timestamp(start).date().isoformat()
    except (TypeError, ValueError):
        start_text = str(start)
    return {
        "start": start_text,
        "sample_end": sample_end,
        "params": _params_dict(params),
        "skipped": list(skipped),
    }


def write_panorama(table: pd.DataFrame, meta: dict, cache_dir: Path | None = None) -> Path | None:
    """空表不覆盖已有快照。只有写默认缓存时才抄到已发布目录。"""
    if table is None or table.empty or not meta.get("sample_end"):
        return None
    directory = CACHE_DIR if cache_dir is None else cache_dir
    directory.mkdir(parents=True, exist_ok=True)
    table_path = directory / TABLE_NAME
    meta_path = directory / META_NAME
    tmp = table_path.with_suffix(".tmp.parquet")
    table.to_parquet(tmp, index=False)
    tmp.replace(table_path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if same_dir(directory, CACHE_DIR):
        publish_file(table_path, PUBLISHED_DIR / TABLE_NAME)
        published_meta = PUBLISHED_DIR / META_NAME
        published_meta.parent.mkdir(parents=True, exist_ok=True)
        published_meta.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
    return table_path


def load_panorama(page, cache_dir: Path | None = None) -> tuple[pd.DataFrame | None, dict | None, str]:
    """匹配当前起始日、结束日和全部参数才返回表。不访问网络。

    状态是 ok、missing、earlier_end、window、spec 之一。
    """
    directory = CACHE_DIR if cache_dir is None else cache_dir
    path = prefer(
        directory / TABLE_NAME,
        PUBLISHED_DIR / TABLE_NAME,
        directory,
        CACHE_DIR,
    )
    if path is None:
        return None, None, "missing"
    meta_path = path.parent / META_NAME
    if not meta_path.exists():
        return None, None, "missing"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        table = pd.read_parquet(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None, "missing"
    if not isinstance(meta, dict) or table.empty or not meta.get("sample_end"):
        return None, meta if isinstance(meta, dict) else None, "missing"
    try:
        start = pd.Timestamp(page.start_day).date()
        end = pd.Timestamp(page.end_day).date()
        stored_start = date.fromisoformat(meta["start"])
        stored_end = date.fromisoformat(meta["sample_end"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None, meta, "window"
    if end < stored_end:
        return None, meta, "earlier_end"
    if start != stored_start or (end - stored_end).days > END_SLACK_DAYS:
        return None, meta, "window"
    live = _params_dict(page.params)
    if not _close(live, meta.get("params")):
        return None, meta, "spec"
    return table, meta, "ok"
