"""逐笔收益归因。每日各项之和等于引擎日净收益；逐笔各项之和等于该笔净收益。

H 腿人民币收益已含汇率。在 H 腿实际计价的日子拆成：
港股本币贡献 = wH × rHKD；汇率贡献 = wH × (1 + rHKD) × rFX，交叉项计入汇率。
H 腿不计价的日子两项都是零，和回测记账完全一致。
费用子项只加一次；v15 没建模借券，标“未计入”，不是实际费用为零。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = ("a_price", "h_price", "fx", "cost", "borrow")
TOL = 1e-10


def leg_decomposition(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """按记账口径拆 H 腿：rHKD 与 rFX 在 H 腿不计价的日子都是零。"""
    h_hkd = frame["h_hkd"].astype("float64")
    fx = frame["fx"].astype("float64")
    r_hkd = h_hkd.pct_change().fillna(0.0)
    r_fx = fx.pct_change().fillna(0.0)
    if "h_priced" in frame.columns:
        mask = frame["h_priced"].fillna(False).astype(bool)
        r_hkd = r_hkd.where(mask, 0.0)
        r_fx = r_fx.where(mask, 0.0)
    return r_hkd.to_numpy(dtype=np.float64), r_fx.to_numpy(dtype=np.float64)


def attribute_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """逐日归因。frame 需要 w_a/w_h（当日盘后权重）、ret_a、h_hkd、fx、cost、borrow、equity。

    敞口用前一日的盘后权重，和回测记账一致。返回每项的收益贡献（不是金额）。
    """
    n = len(frame)
    w_a = frame["w_a"].astype("float64").shift(1).fillna(0.0).to_numpy()
    w_h = frame["w_h"].astype("float64").shift(1).fillna(0.0).to_numpy()
    ret_a = frame["ret_a"].astype("float64").fillna(0.0).to_numpy()
    r_hkd, r_fx = leg_decomposition(frame)
    cost = frame["cost"].astype("float64").fillna(0.0).to_numpy() if "cost" in frame.columns else np.zeros(n)
    borrow = frame["borrow"].astype("float64").fillna(0.0).to_numpy() if "borrow" in frame.columns else np.zeros(n)
    out = pd.DataFrame(index=frame.index)
    out["a_price"] = w_a * ret_a
    out["h_price"] = w_h * r_hkd
    out["fx"] = w_h * (1.0 + r_hkd) * r_fx
    out["cost"] = -cost
    out["borrow"] = -borrow
    out["net"] = out["a_price"] + out["h_price"] + out["fx"] + out["cost"] + out["borrow"]
    out["equity"] = frame["equity"].astype("float64").to_numpy() if "equity" in frame.columns else np.nan
    if "ret_strat" in frame.columns:
        out["engine_net"] = frame["ret_strat"].astype("float64").fillna(0.0).to_numpy()
        out["residual"] = out["engine_net"] - out["net"]
    return out


def reconcile_daily(attr: pd.DataFrame, tol: float = TOL) -> bool:
    """每日归因合计与引擎净收益一致（未舍入，绝对误差 tol 内）。"""
    if "residual" not in attr.columns:
        return True
    res = attr["residual"].to_numpy(dtype=np.float64)
    res = res[np.isfinite(res)]
    return bool(len(res) == 0 or np.max(np.abs(res)) <= tol)


def attribute_window(attr: pd.DataFrame, entry, exit_) -> dict:
    """把 entry 到 exit（含）的逐日贡献换成金额再除以交易起始本金。

    各项之和等于这段窗口的复合净收益。入口日前一天净值为本金；
    净值耗尽或缺数据时标注不可归因。
    """
    entry_ts = pd.Timestamp(entry)
    exit_ts = pd.Timestamp(exit_)
    window = attr.loc[(attr.index >= entry_ts) & (attr.index <= exit_ts)]
    if window.empty or "equity" not in window.columns:
        return {"ok": False, "reason": "没有可归因的日子"}
    position = attr.index.get_indexer(window.index)
    base_idx = position[0] - 1
    base = float(attr["equity"].iloc[base_idx]) if base_idx >= 0 else 1.0
    if not np.isfinite(base) or base <= 0:
        return {"ok": False, "reason": "本金净值缺失或已耗尽，不可归因"}
    out = {"ok": True, "base": base, "n_days": int(len(window))}
    total = 0.0
    for component in COMPONENTS:
        amount = 0.0
        for t, value in enumerate(window[component].to_numpy(dtype=np.float64)):
            eq_prev = float(attr["equity"].iloc[position[t] - 1]) if position[t] > 0 else 1.0
            if not np.isfinite(eq_prev) or eq_prev <= 0:
                return {"ok": False, "reason": "净值耗尽，不可归因"}
            amount += float(value) * eq_prev
        out[component] = amount / base
        total += amount / base
    out["total"] = total
    end = float(window["equity"].iloc[-1])
    out["trade_return"] = end / base - 1.0 if base > 0 else np.nan
    out["reconciled"] = bool(np.isfinite(out["trade_return"]) and abs(out["trade_return"] - total) <= TOL)
    return out

def ledger_episodes(rows: list[dict]) -> list[dict]:
    """从账本行提取交易片段。翻仓当天先记旧片段平仓、再开新片段。未平仓 closed=False。"""
    episodes = []
    position = 0
    entry = None
    direction = 0
    for row in rows:
        pos = int(row.get("position") or 0)
        if position == 0 and pos != 0:
            entry = row["date"]
            direction = pos
        elif position != 0 and pos != position:
            episodes.append({"entry_date": entry, "exit_date": row["date"], "direction": direction, "closed": pos == 0 or True})
            if pos != 0:
                entry = row["date"]
                direction = pos
            else:
                entry = None
        position = pos
    if entry is not None and position != 0:
        episodes.append({"entry_date": entry, "exit_date": None, "direction": direction, "closed": False})
    for episode in episodes:
        if episode["exit_date"] is None:
            episode["closed"] = False
    return episodes


def frame_from_v15_ledger(rows: list[dict], price_frame: pd.DataFrame) -> pd.DataFrame:
    """把 v15 账本行对齐到行情，构造归因需要的框。权重固定为 ±0.5 等金额两腿。

    借券费在 v15 没有建模：borrow 列恒为 0，展示端必须标“未建模”，
    不能把这块当成真实发生的零成本。
    """
    idx = pd.DatetimeIndex(pd.to_datetime([row["date"] for row in rows]))
    aligned = price_frame.reindex(idx)
    position = np.array([int(row.get("position") or 0) for row in rows], dtype=np.float64)
    out = pd.DataFrame(index=idx)
    out["w_a"] = 0.5 * position
    out["w_h"] = -0.5 * position
    out["ret_a"] = aligned["ret_a"].astype("float64").fillna(0.0).to_numpy()
    out["h_hkd"] = aligned["h_hkd"].astype("float64").ffill().to_numpy()
    out["fx"] = aligned["fx"].astype("float64").ffill().to_numpy()
    if "h_priced" in aligned.columns:
        out["h_priced"] = aligned["h_priced"].fillna(False).astype(bool).to_numpy()
    out["cost"] = np.array([float(row.get("cost") or 0.0) for row in rows], dtype=np.float64)
    out["borrow"] = 0.0
    out["equity"] = np.array([float(row.get("equity") or np.nan) for row in rows], dtype=np.float64)
    out["ret_strat"] = np.array([float(row.get("net") or 0.0) for row in rows], dtype=np.float64)
    return out


def v14_attribution_frame(price_frame: pd.DataFrame, preset, inception=None) -> pd.DataFrame:
    """v14 账本口径的归因框：按预设重算同一套规则，权重、成本与账本一致。

    因果信号下重放等于逐日推进，所以重算框的逐日权重与封存账本相同；
    若上游修订过行情，归因反映的是当前数据，页面需注明这一点。
    """
    from ah_pairs.analysis import build_signals
    from ah_pairs.backtest import run_backtest

    params = preset.strategy_params()
    signals = build_signals(price_frame, params, alt=False)
    if inception is not None:
        signals = signals.loc[signals.index >= pd.Timestamp(inception).normalize()]
    result, _trades, _stats = run_backtest(signals, params)
    return result

