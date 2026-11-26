"""前瞻模拟账本。只追加，记录行情截止和实际观察两个时间。

- 每条账本是一只股票 × 一个预设的独立模拟，不是组合资金分配。
- 上线日空仓启动，不继承历史回测的仓位和收益。
- 行按决策日期封存；observed_at 是系统实际生成这一行的时间。迟到不倒签：
  漏观察的区间标 backfilled 并写 gap，事后重放的行不进“真实前瞻”口径。
- 同一日期内容相同则跳过；不同则记冲突，旧行不动。上游修订不改旧账。
- 正式账本只由每日刷新写。页面打开、手动重算、历史回放都没有写权限。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ah_pairs.analysis import build_signals, warmup_start
from ah_pairs.backtest import positions_with_reasons, run_backtest, _tradable
from ah_pairs.data import assemble_pair, read_local_close
from ah_pairs.eflab_calendar import as_shanghai
from ah_pairs.paths import PUBLISHED_DIR
from ah_pairs.presets import get_preset, spec_fingerprint
from ah_pairs.universe import a_to_yahoo, h_to_yahoo
from ah_pairs.v15.live import run_live

FORWARD_DIR = PUBLISHED_DIR / "forward"
START = "2010-01-01"

_COMPARE_FIELDS = ("position", "signal", "net", "z")


def ledger_dir(root: Path, preset_id: str) -> Path:
    return Path(root) / "ledgers" / preset_id


def rows_path(root: Path, preset_id: str, a_code: str) -> Path:
    return ledger_dir(root, preset_id) / f"{a_code}.jsonl"


def state_path(root: Path, preset_id: str, a_code: str) -> Path:
    return ledger_dir(root, preset_id) / f"{a_code}.state.json"


def read_rows(path: Path) -> list[dict]:
    rows = []
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def read_state(path: Path) -> dict | None:
    try:
        if Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return None


def fingerprint_frame(frame: pd.DataFrame) -> str:
    """输入指纹：行数、首尾日期、最后 5 行两腿收盘。上游修订历史会改变它。"""
    tail = frame.tail(5)
    payload = {
        "n": int(len(frame)),
        "first": str(frame.index.min().date()) if len(frame) else "",
        "last": str(frame.index.max().date()) if len(frame) else "",
        "a": [round(float(x), 6) for x in tail["a_cny"]] if "a_cny" in tail else [],
        "h": [round(float(x), 6) for x in tail["h_cny"]] if "h_cny" in tail else [],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def pool_fingerprint(codes) -> str:
    payload = ",".join(sorted(str(code).zfill(6) for code in codes))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _same_row(old: dict, new: dict) -> bool:
    for field in _COMPARE_FIELDS:
        a, b = old.get(field), new.get(field)
        if a is None and b is None:
            continue
        if isinstance(a, float) or isinstance(b, float):
            if a is None or b is None or not np.isclose(float(a), float(b), atol=1e-12, equal_nan=True):
                return False
        elif a != b:
            return False
    return True


def append_rows(path: Path, new_rows: list[dict]) -> dict:
    """追加新日期。旧日期内容一致跳过，不一致记冲突且旧行不动。返回写入摘要。"""
    existing = read_rows(path)
    by_date = {row["date"]: row for row in existing if "date" in row}
    written = 0
    skipped = 0
    conflicts = []
    lines = []
    for row in new_rows:
        date = row["date"]
        if date in by_date:
            if _same_row(by_date[date], row):
                skipped += 1
            else:
                conflicts.append(date)
            continue
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
        written += 1
    if lines:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
    return {"written": written, "skipped": skipped, "conflicts": conflicts, "total": len(existing) + written}


def write_state(path: Path, state: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def v14_daily_rows(frame: pd.DataFrame, preset, inception=None) -> list[dict]:
    """从 v14 批量回测提取逐日行。因果信号下重放等于逐日推进。

    inception 之前的历史只用来预热 z 和 β；仓位从 inception 空仓起步。
    """
    params = preset.strategy_params()
    signals = build_signals(frame, params, alt=False)
    if inception is not None:
        signals = signals.loc[signals.index >= pd.Timestamp(inception).normalize()]
    if signals.empty:
        return []
    result, _trades, _stats = run_backtest(signals, params)
    z = result["z"].to_numpy(dtype=float)
    beta = result["beta"].to_numpy(dtype=float)
    tradable = _tradable(result)
    target, reasons = positions_with_reasons(z, beta, params, tradable=tradable)
    rows = []
    dates = result.index
    position = result["position"].to_numpy(dtype=int)
    prev_position = np.roll(position, 1)
    prev_position[0] = 0
    fills_today = (position != prev_position).astype(int)
    for t in range(len(result)):
        z_now = float(z[t])
        rows.append(
            {
                "date": str(pd.Timestamp(dates[t]).date()),
                "z": z_now if np.isfinite(z_now) else None,
                "half_life": None,
                "gate": None,
                "position": int(position[t]),
                "signal": int(target[t]),
                "pending": bool(int(target[t]) != int(position[t])),
                "reason": reasons[t],
                "gross": float(result["ret_gross"].iloc[t]),
                "cost": float(result["cost"].iloc[t]),
                "borrow": float(result["borrow"].iloc[t]),
                "net": float(result["ret_strat"].iloc[t]),
                "equity": float(result["equity"].iloc[t]),
                "beta": float(beta[t]) if np.isfinite(beta[t]) else None,
                "n_fills": int(fills_today[t]),
                "fills": [],
            }
        )
    return rows


def v15_daily_rows(frame: pd.DataFrame, preset, locks: dict | None, inception=None) -> tuple[list[dict], dict]:
    """连续 v15。已有锁直接复用；inception 空仓起步。返回逐日行和期末状态。"""
    state = {"locks": dict(locks or {})}
    out = run_live(frame, preset, state=state, inception=inception)
    return out["rows"], out["state"]


def build_pair_frame(a_code: str, h_code: str, end=None) -> tuple[pd.DataFrame | None, str]:
    """只读本地缓存拼行情。缺腿返回 (None, 原因)。"""
    a_px = read_local_close(a_to_yahoo(a_code))
    h_px = read_local_close(h_to_yahoo(h_code))
    fx = read_local_close("HKDCNY=X")
    missing = []
    if a_px is None or a_px.empty:
        missing.append("A")
    if h_px is None or h_px.empty:
        missing.append("H")
    if fx is None or fx.empty:
        missing.append("FX")
    if missing:
        return None, "缺少" + "/".join(missing) + "本地行情"
    end_day = pd.Timestamp(end).normalize() if end is not None else None
    if end_day is None:
        last = min(a_px.index.max(), h_px.index.max(), fx.index.max())
        end_day = last + pd.Timedelta(days=1)
    try:
        frame, _notes = assemble_pair(a_px, h_px, fx, pd.Timestamp(START), end_day)
    except Exception as exc:
        return None, str(exc)[:200]
    return frame, ""


def update_ledger(
    item: dict,
    preset_id: str,
    *,
    root: Path = FORWARD_DIR,
    observed_at: datetime | None = None,
    end=None,
    pool_fp: str = "",
) -> dict:
    """更新一只股票 × 一个预设的账本。只读本地行情，不下载。

    返回摘要：written/skipped/conflicts/gap/data_asof/backfilled。
    """
    preset = get_preset(preset_id)
    a_code = str(item["a_code"]).zfill(6)
    h_code = str(item["h_code"]).zfill(5)
    observed = as_shanghai(observed_at).isoformat()
    rpath = rows_path(root, preset_id, a_code)
    spath = state_path(root, preset_id, a_code)
    existing = read_rows(rpath)
    last_sealed = existing[-1]["date"] if existing else None
    prior_state = read_state(spath) or {}

    frame, problem = build_pair_frame(a_code, h_code, end=end)
    summary = {
        "a_code": a_code,
        "preset_id": preset_id,
        "written": 0,
        "skipped": 0,
        "conflicts": [],
        "gap": "",
        "data_asof": None,
        "backfilled": 0,
    }
    if frame is None or frame.empty:
        summary["gap"] = problem or "无行情"
        state = dict(prior_state)
        gaps = list(state.get("gaps") or [])
        gaps.append({"observed_at": observed, "reason": summary["gap"]})
        state["gaps"] = gaps[-50:]
        state["last_observed_at"] = observed
        write_state(spath, state)
        return summary

    data_asof = str(pd.Timestamp(frame.index.max()).date())
    summary["data_asof"] = data_asof
    input_fp = fingerprint_frame(frame)
    spec_fp = spec_fingerprint(preset)
    # 上线日空仓启动：首次写入时 inception 就是这批数据的截止日，之后不再变。
    inception = prior_state.get("inception") or data_asof

    if preset.engine == "v14":
        daily = v14_daily_rows(frame, preset, inception=inception)
        locks = None
    else:
        daily, live_state = v15_daily_rows(frame, preset, (prior_state.get("v15") or {}).get("locks"), inception=inception)
        locks = live_state["locks"]
    if not daily:
        summary["gap"] = "样本不足以产生第一期信号"
        state = dict(prior_state)
        state["last_observed_at"] = observed
        write_state(spath, state)
        return summary

    # 漏观察检测：上次封存日到这次数据截止之间隔了不止一个共同开市日，
    # 重放出来的行标 backfilled，不进“真实前瞻”口径。
    backfill_dates: set[str] = set()
    if last_sealed is not None:
        common = frame.index[frame["both_open"].fillna(False)] if "both_open" in frame.columns else frame.index
        later = common[common > pd.Timestamp(last_sealed)]
        if len(later) > 1:
            backfill_dates = {str(pd.Timestamp(day).date()) for day in later[:-1]}

    enriched = []
    for row in daily:
        row = dict(row)
        row.update(
            {
                "engine": preset.engine,
                "preset_id": preset.preset_id,
                "preset_version": preset.version,
                "a_code": a_code,
                "h_code": h_code,
                "observed_at": observed,
                "data_asof": data_asof,
                "spec_fp": spec_fp,
                "input_fp": input_fp,
                "pool_fp": pool_fp,
            }
        )
        if row["date"] in backfill_dates:
            row["backfilled"] = True
            summary["backfilled"] += 1
        enriched.append(row)
    result = append_rows(rpath, enriched)
    summary.update(written=result["written"], skipped=result["skipped"], conflicts=result["conflicts"])

    # 缺口事件：漏观察窗口里补回放的行数、以及其中按规则会发生的成交笔数。
    # 成交是规则对数据的确定函数，但系统当时并未实时观察，所以永久留痕，
    # 展示端的前瞻净值在缺口之后重新归一，不把这段接进“真实前瞻”。
    gaps = list(prior_state.get("gaps") or [])
    if backfill_dates and last_sealed is not None:
        gap_rows = [row for row in enriched if row["date"] in backfill_dates]
        fills = sum(int(row.get("n_fills") or 0) for row in gap_rows)
        event = {
            "observed_at": observed,
            "from": str(last_sealed),
            "to": data_asof,
            "rows": len(gap_rows),
            "fills": fills,
        }
        if not any(g.get("from") == event["from"] and g.get("to") == event["to"] for g in gaps):
            gaps.append(event)

    last_row = daily[-1]
    state = {
        "engine": preset.engine,
        "preset_id": preset.preset_id,
        "preset_version": preset.version,
        "a_code": a_code,
        "h_code": h_code,
        "inception": inception,
        "last_date": last_row["date"],
        "last_observed_at": observed,
        "position": last_row["position"],
        "signal": last_row["signal"],
        "pending": last_row["pending"],
        "equity": last_row["equity"],
        "data_asof": data_asof,
        "input_fp": input_fp,
        "spec_fp": spec_fp,
        "v15": {"locks": locks} if locks is not None else None,
        "gaps": gaps[-50:],
    }
    write_state(spath, state)
    return summary


def annotate_segments(rows: list[dict]) -> list[dict]:
    """给每行标段号和段内净值。补回放行不进任何一段（seg_id 为 None）。

    每一段是连续实时观察的区间，净值从 1 重新归一；
    段与段之间隔着补回放行，不能拼成一条曲线。
    """
    out = []
    seg_id = 0
    base = None
    for row in rows:
        row = dict(row)
        if row.get("backfilled"):
            row["seg_id"] = None
            row["seg_equity"] = None
            out.append(row)
            base = None
            continue
        if base is None:
            seg_id += 1
            base = None
        equity = row.get("equity")
        if base is None:
            base = equity / (1.0 + float(row.get("net") or 0.0)) if equity is not None else None
            if base is not None and base <= 0:
                base = None
        row["seg_id"] = seg_id
        row["seg_equity"] = (equity / base) if (equity is not None and base) else None
        out.append(row)
    return out


def update_batch(
    universe: pd.DataFrame,
    preset_ids,
    *,
    root: Path = FORWARD_DIR,
    observed_at: datetime | None = None,
    end=None,
) -> list[dict]:
    """每日刷新调用的批量入口。逐股逐预设独立，单股失败不影响其他。"""
    codes = [str(code).zfill(6) for code in universe["a_code"]] if "a_code" in universe.columns else []
    pool_fp = pool_fingerprint(codes)
    summaries = []
    for item in universe.to_dict(orient="records"):
        for preset_id in preset_ids:
            try:
                summaries.append(
                    update_ledger(item, preset_id, root=root, observed_at=observed_at, end=end, pool_fp=pool_fp)
                )
            except Exception as exc:
                summaries.append(
                    {
                        "a_code": str(item.get("a_code", "?")).zfill(6),
                        "preset_id": preset_id,
                        "written": 0,
                        "skipped": 0,
                        "conflicts": [],
                        "gap": str(exc)[:200],
                        "data_asof": None,
                        "backfilled": 0,
                    }
                )
    return summaries


def read_ledger(root: Path, preset_id: str, a_code: str) -> tuple[list[dict], dict]:
    """页面只读入口。返回 (逐日行, 状态)。"""
    return read_rows(rows_path(root, preset_id, a_code)), read_state(state_path(root, preset_id, a_code)) or {}
