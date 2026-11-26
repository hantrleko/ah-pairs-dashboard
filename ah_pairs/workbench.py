"""今日变化工作台的后端。只读前瞻账本，不重算、不下载。

每行是一只股票 × 一个预设。变化比较的是账本里最近两个决策日，
不是两次任务时间。连续越阈只在首次触发日标“新触发”。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ah_pairs.forward import FORWARD_DIR, read_rows, read_state, rows_path, state_path

ENTRY_REASONS = {"enter_long_spread", "enter_short_spread"}
EXIT_REASONS = {"exit_z_out", "exit_stop", "exit_max_hold", "short_a_forced_exit"}
BLOCKED_REASONS = {"short_a_blocked", "gate_blocked", "beta_blocked"}


def classify_change(prev: dict | None, row: dict) -> str:
    """最近两个决策日之间的变化。数据不足或理由缺失时不猜。"""
    if row.get("backfilled"):
        return "backfilled"
    reason = row.get("reason") or ""
    prev_position = int(prev["position"]) if prev and prev.get("position") is not None else 0
    position = int(row.get("position") or 0)
    if position != 0 and prev_position == 0:
        return "filled_entry"
    if position == 0 and prev_position != 0:
        return "filled_exit"
    if position != 0 and prev_position != 0 and position != prev_position:
        return "filled_flip"
    if reason in ENTRY_REASONS:
        return "new_trigger"
    if reason in EXIT_REASONS:
        return "exit_signal"
    if reason in BLOCKED_REASONS:
        return "blocked"
    if reason == "one_sided":
        return "one_sided"
    return "unchanged"


CHANGE_ZH = {
    "new_trigger": "新触发",
    "exit_signal": "退出信号",
    "filled_entry": "已开仓",
    "filled_exit": "已平仓",
    "filled_flip": "已翻仓",
    "blocked": "持续受限",
    "one_sided": "单边行情日",
    "unchanged": "无变化",
    "backfilled": "补回放",
    "unknown": "无法判断",
}

CHANGE_EN = {
    "new_trigger": "New trigger",
    "exit_signal": "Exit signal",
    "filled_entry": "Filled entry",
    "filled_exit": "Filled exit",
    "filled_flip": "Flipped",
    "blocked": "Blocked",
    "one_sided": "One-sided day",
    "unchanged": "Unchanged",
    "backfilled": "Backfilled",
    "unknown": "Unknown",
}


def ledger_row(item_name: str, rows: list[dict], state: dict) -> dict | None:
    if not rows:
        return None
    last = rows[-1]
    prev = rows[-2] if len(rows) > 1 else None
    change = classify_change(prev, last)
    return {
        "name": item_name,
        "a_code": last.get("a_code"),
        "h_code": last.get("h_code"),
        "engine": last.get("engine"),
        "preset_id": last.get("preset_id"),
        "preset_version": last.get("preset_version"),
        "date": last.get("date"),
        "data_asof": last.get("data_asof"),
        "observed_at": last.get("observed_at"),
        "z": last.get("z"),
        "position": last.get("position"),
        "signal": last.get("signal"),
        "pending": last.get("pending"),
        "reason": last.get("reason"),
        "change": change,
        "equity": last.get("equity"),
        "inception": state.get("inception"),
        "n_rows": len(rows),
        "backfilled": bool(last.get("backfilled")),
    }


def build_workbench(universe: pd.DataFrame, root: Path = FORWARD_DIR) -> tuple[pd.DataFrame, list[str]]:
    """读取全部账本最新行。还没有账本的预设或股票不猜，列进 missing。"""
    from ah_pairs.presets import PRESETS

    names = {}
    if "a_code" in universe.columns:
        for item in universe.to_dict(orient="records"):
            names[str(item["a_code"]).zfill(6)] = str(item.get("name") or item["a_code"])
    rows = []
    missing = []
    if not Path(root).exists():
        return pd.DataFrame(), ["全部账本缺失"]
    ledgers = Path(root) / "ledgers"
    if not ledgers.exists():
        return pd.DataFrame(), ["全部账本缺失"]
    present = set()
    for preset_dir in sorted(ledgers.iterdir()):
        if not preset_dir.is_dir():
            continue
        preset_id = preset_dir.name
        for path in sorted(preset_dir.glob("*.jsonl")):
            a_code = path.stem
            ledger_rows = read_rows(path)
            state = read_state(state_path(root, preset_id, a_code)) or {}
            row = ledger_row(names.get(a_code, a_code), ledger_rows, state)
            if row is None:
                missing.append(f"{a_code}/{preset_id}")
            else:
                rows.append(row)
                present.add((preset_id, a_code))
    # 部分启账时，股票池 × 预设里没有账本的组合不静默消失，标成“无法判断”。
    if rows and "a_code" in universe.columns:
        for item in universe.to_dict(orient="records"):
            a_code = str(item["a_code"]).zfill(6)
            for preset_id in PRESETS:
                if (preset_id, a_code) in present:
                    continue
                rows.append(
                    {
                        "name": names.get(a_code, a_code),
                        "a_code": a_code,
                        "h_code": str(item.get("h_code") or ""),
                        "engine": get_preset_engine(preset_id),
                        "preset_id": preset_id,
                        "preset_version": None,
                        "date": None,
                        "data_asof": None,
                        "observed_at": None,
                        "z": None,
                        "position": 0,
                        "signal": 0,
                        "pending": None,
                        "reason": "",
                        "change": "unknown",
                        "equity": None,
                        "inception": None,
                        "n_rows": 0,
                        "backfilled": False,
                    }
                )
                missing.append(f"{a_code}/{preset_id}")
    table = pd.DataFrame(rows) if rows else pd.DataFrame()
    return table, missing


def get_preset_engine(preset_id: str) -> str:
    from ah_pairs.presets import get_preset

    try:
        return get_preset(preset_id).engine
    except Exception:
        return preset_id.split("-")[0]


def compare_with_rebuild_note() -> str:
    return "前瞻账本只追加；页面重算用同一套规格但数据来源不同，两者不能拼成一条曲线。"
