"""V1.5 连续每日策略。前瞻账本和诊断共用，不是分折样本外回测的别名。

规则：
- 共同开市日估计 z 和半衰期闸门；并集日历记账，单边日不成交、不读 z。
- 756 个共同开市日训练加 121 日禁区之后，每 126 个共同开市日锁一次参数。
- 锁定期内不重选；参数一经锁定不回改，即使上游后来修订历史。
- 跨锁定期不清仓：已有仓位沿用开仓时锁定的退出阈值，平仓后才按新参数开仓。
- 上线日空仓启动。逐日状态可序列化；给定同样的锁，重放结果完全一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ah_pairs.v15.adaptive import exec_step, new_exec_state
from ah_pairs.v15.walkforward import _common_index, _eg_at, choose_for_test_start, prepare_frame, training_index, union_exec_frame


def initial_state() -> dict:
    return {
        "locks": {},
        "exec": new_exec_state(),
        "equity": 1.0,
    }


def live_periods(common, train: int, embargo: int, lock_len: int) -> list[dict]:
    """按共同开市日数 tile 的锁定期。最后一期可以不满 lock_len，锁仍然有效。"""
    days = pd.DatetimeIndex(common)
    out = []
    cursor = int(train) + int(embargo)
    while cursor < len(days):
        end_idx = min(cursor + int(lock_len) - 1, len(days) - 1)
        out.append({"start": days[cursor], "end": days[end_idx], "pos": cursor})
        cursor += int(lock_len)
    return out


def _select_lock(frame: pd.DataFrame, period_start, preset) -> dict:
    chosen = choose_for_test_start(
        frame,
        period_start,
        train=preset.train,
        embargo=preset.embargo,
        allow_short_a=preset.allow_short_a,
        inner_train=preset.inner_train,
        inner_valid=preset.inner_valid,
        min_trades=preset.min_trades,
        z_window=preset.z_window,
        z_in_grid=tuple(preset.z_in_grid),
        z_out_grid=tuple(preset.z_out_grid),
        w_est_grid=tuple(preset.w_est_grid),
    )
    train_idx = training_index(frame, period_start, train=preset.train, embargo=preset.embargo)
    eg_p, hl = _eg_at(frame.loc[train_idx]) if len(train_idx) else (np.nan, np.nan)
    return {
        "z_in": float(chosen["z_in"]),
        "z_out": float(chosen["z_out"]),
        "W_est": int(chosen["W_est"]),
        "qualified": bool(chosen.get("qualified", False)),
        "eg_p": float(eg_p) if np.isfinite(eg_p) else None,
        "half_life_at_start": float(hl) if np.isfinite(hl) else None,
    }


def run_live(frame: pd.DataFrame, preset, state: dict | None = None, inception=None) -> dict:
    """重放连续策略。state 里已有的锁直接复用；缺失的锁用当时之前的数据补选。

    inception 给出上线日：之前的行情只用于估计和锁参，仓位在这一天从空仓开始，
    不继承历史回测的持仓。返回 {"rows": 逐日行, "state": 期末状态, "periods": 锁定期表}。
    """
    state = {"locks": dict((state or initial_state())["locks"]), "exec": new_exec_state(), "equity": 1.0}
    common = _common_index(frame)
    if len(common) == 0:
        return {"rows": [], "state": state, "periods": []}
    estimation = frame.loc[common]
    prepared_by_w = {}
    for w_est in tuple(preset.w_est_grid):
        prepared_by_w[int(w_est)] = prepare_frame(estimation, int(w_est), z_window=preset.z_window)
    periods = live_periods(common, preset.train, preset.embargo, preset.lock_len)
    if not periods:
        return {"rows": [], "state": state, "periods": []}

    rows: list[dict] = []
    exec_state = state["exec"]
    equity = float(state["equity"])
    first_start = periods[0]["start"]
    eval_all = frame.index[frame.index >= first_start]
    if len(eval_all) == 0:
        return {"rows": [], "state": state, "periods": periods}

    # 每一天属于哪个锁定期：按共同开市日位置判断。
    common_arr = pd.DatetimeIndex(common)
    period_bounds = [(p["start"], p["end"]) for p in periods]

    def period_for(day) -> dict | None:
        for meta in periods:
            if meta["start"] <= day <= meta["end"]:
                return meta
        return None

    z_master = prepared_by_w[int(preset.w_est_grid[0])][0]["z_net"]
    z_aligned = z_master.reindex(eval_all).to_numpy(dtype=np.float64)
    both_arr = frame["both_open"].fillna(False).reindex(eval_all).to_numpy(dtype=bool) if "both_open" in frame.columns else np.ones(len(eval_all), dtype=bool)
    ret_a_arr = frame["ret_a"].reindex(eval_all).to_numpy(dtype=np.float64) if "ret_a" in frame.columns else np.zeros(len(eval_all))
    ret_h_arr = frame["ret_h"].reindex(eval_all).to_numpy(dtype=np.float64) if "ret_h" in frame.columns else np.zeros(len(eval_all))
    fx_arr = frame["fx"].reindex(eval_all).to_numpy(dtype=np.float64) if "fx" in frame.columns else np.full(len(eval_all), 0.9)
    gate_series_by_w = {
        w: pd.Series(np.asarray(prepared_by_w[w][1], dtype=bool), index=prepared_by_w[w][0].index)
        for w in prepared_by_w
    }
    hl_series_by_w = {w: prepared_by_w[w][0]["half_life"] for w in prepared_by_w}

    inception_ts = pd.Timestamp(inception).normalize() if inception is not None else None
    current = None
    lock = None
    eg_blocked = False
    for pos, day in enumerate(eval_all):
        both = bool(both_arr[pos])
        if both:
            meta = period_for(day)
            if meta is not None and (current is None or meta["start"] != current["start"]):
                current = meta
                key = str(pd.Timestamp(meta["start"]).date())
                # 上线之前已经结束的锁定期不影响任何持仓，跳过昂贵的参数选择。
                needed = inception_ts is None or pd.Timestamp(meta["end"]) >= inception_ts
                if needed and key not in state["locks"]:
                    state["locks"][key] = _select_lock(frame, meta["start"], preset)
                lock = state["locks"].get(key)
                if lock is not None:
                    eg_p = lock.get("eg_p")
                    eg_blocked = bool(eg_p is None or not np.isfinite(eg_p) or float(eg_p) > 0.10)
        if inception_ts is not None and day < inception_ts:
            continue
        if current is None or lock is None:
            continue
        w_est = int(lock["W_est"])
        z_val = float(z_aligned[pos]) if np.isfinite(z_aligned[pos]) else np.nan
        gate_val = bool(gate_series_by_w[w_est].get(day, False)) if not eg_blocked else False
        hl_series = hl_series_by_w[w_est]
        hl_val = float(hl_series[day]) if day in hl_series.index and np.isfinite(hl_series[day]) else None
        step = exec_step(
            exec_state,
            date=day,
            both_open=both,
            ret_a=float(ret_a_arr[pos]),
            ret_h=float(ret_h_arr[pos]),
            fx=float(fx_arr[pos]) if np.isfinite(fx_arr[pos]) else 0.9,
            z_now=z_val,
            allowed=gate_val,
            z_in=float(lock["z_in"]),
            z_out=float(lock["z_out"]),
            max_hold=int(preset.max_hold),
            allow_short_a=bool(preset.allow_short_a),
            stop_z=float(preset.stop_z),
        )
        equity_before = equity
        equity = equity * (1.0 + step["net"]) if step["net"] > -1.0 else 0.0
        rows.append(
            {
                "date": str(pd.Timestamp(day).date()),
                "z": z_val if np.isfinite(z_val) else None,
                "half_life": hl_val,
                "gate": bool(gate_val),
                "position": int(step["position"]),
                "signal": int(step["signal"]),
                "pending": bool(step["pending"]),
                "reason": step["reason"],
                "gross": float(step["gross"]),
                "cost": float(step["cost"]),
                "borrow": None,
                "net": float(step["net"]),
                "equity": float(equity),
                "equity_before": float(equity_before),
                "z_in": float(lock["z_in"]),
                "z_out": float(lock["z_out"]),
                "exit_z": exec_state["exit_z"],
                "W_est": int(lock["W_est"]),
                "eg_p": lock.get("eg_p"),
                "period": str(pd.Timestamp(current["start"]).date()),
                "n_fills": len(step["rows"]),
                "fills": [
                    {"leg": row["leg"], "side": row["side"], "total": float(row["total"])}
                    for row in step["rows"]
                ],
            }
        )
    state["exec"] = exec_state
    state["equity"] = float(equity)
    return {"rows": rows, "state": state, "periods": periods}
