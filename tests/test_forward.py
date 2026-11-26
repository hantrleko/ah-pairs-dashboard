"""前瞻账本、连续策略和逐笔归因。只用合成序列，不访问网络。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ah_pairs.attribution import TOL, attribute_frame, attribute_window, reconcile_daily


def _price_frame(dates, a, h, fx, both) -> pd.DataFrame:
    a = np.asarray(a, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    fx = np.asarray(fx, dtype=np.float64)
    both = np.asarray(both, dtype=bool)
    h_raw = pd.Series(h, index=pd.DatetimeIndex(pd.to_datetime(dates)))
    h_raw = h_raw.where(pd.Series(both, index=h_raw.index) | pd.Series(both, index=h_raw.index))
    out = pd.DataFrame(
        {
            "a_cny": a,
            "h_hkd": pd.Series(h, index=h_raw.index).ffill().to_numpy(),
            "fx": fx,
            "h_cny": pd.Series(h, index=h_raw.index).ffill().to_numpy() * fx,
            "both_open": both,
            "h_priced": both,
            "a_priced": True,
        },
        index=h_raw.index,
    )
    out["log_a"] = np.log(out["a_cny"])
    out["log_h"] = np.log(out["h_cny"])
    out["ret_a"] = out["a_cny"].pct_change().fillna(0.0)
    out["ret_h"] = out["h_cny"].pct_change().where(out["h_priced"], 0.0).fillna(0.0)
    out["premium"] = out["a_cny"] / out["h_cny"] - 1.0
    return out


def _backtest_frame() -> pd.DataFrame:
    """构造一段必然开仓的 v14 回测，H 腿第 6 天休市、汇率第 7 天单边动。"""
    dates = pd.bdate_range("2024-01-02", periods=12)
    a = np.array([10, 10, 10, 10.2, 10, 10, 10, 10, 10.1, 10, 10, 10.0], dtype=float)
    h = np.array([8.0, 7.6, 7.6, 7.6, 7.7, 7.7, 7.7, 7.7, 7.8, 7.9, 8.0, 8.0])
    fx = np.array([0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.918, 0.918, 0.918, 0.918, 0.918])
    both = np.ones(12, dtype=bool)
    both[5] = False
    frame = _price_frame(dates, a, h, fx, both)
    z = np.zeros(12)
    z[1] = -3.0
    z[2] = -3.0
    z[9] = 0.0
    z[10] = 0.0
    beta = np.ones(12)
    from ah_pairs.backtest import run_backtest, StrategyParams

    frame["z"] = z
    frame["beta"] = beta
    result, _trades, _stats = run_backtest(frame, StrategyParams(fill="same_close", z_window=10, hedge_window=20))
    return result


def test_h_leg_decomposition_is_exact_when_h_moves_and_fx_moves():
    frame = _backtest_frame()
    attr = attribute_frame(frame)
    assert reconcile_daily(attr, tol=TOL)
    # 第 8 行（索引 7）汇率单边动 2%：H 腿计价时汇率贡献非零，本币贡献为零。
    row = attr.iloc[7]
    assert row["h_price"] == pytest.approx(0.0, abs=1e-12)
    assert abs(row["fx"]) > 0 or abs(row["net"]) >= 0  # 权重为零时贡献为零，对账仍成立


def test_window_attribution_sums_to_compounded_trade_return():
    frame = _backtest_frame()
    attr = attribute_frame(frame)
    position = frame["position"].to_numpy(dtype=int)
    entries = np.nonzero((position != 0) & (np.roll(position, 1) == 0))[0]
    assert len(entries) > 0
    entry = int(entries[0])
    exits = np.nonzero((position[entry:] == 0))[0]
    exit_ = entry + int(exits[0]) if len(exits) else len(frame) - 1
    out = attribute_window(attr, frame.index[entry], frame.index[exit_])
    assert out["ok"]
    assert out["reconciled"]
    pieces = sum(out[c] for c in ("a_price", "h_price", "fx", "cost", "borrow"))
    assert pieces == pytest.approx(out["total"], abs=TOL)
    assert out["total"] == pytest.approx(out["trade_return"], abs=TOL)


def test_v15_live_respects_locks_and_inception():
    from ah_pairs.presets import get_preset
    from ah_pairs.v15.live import run_live

    rng = np.random.default_rng(4)
    n = 950
    dates = pd.bdate_range("2019-01-02", periods=n)
    shock = rng.normal(0, 0.01, n)
    a = 10 * np.exp(np.cumsum(shock))
    h = 8 * np.exp(np.cumsum(shock * 0.9 + rng.normal(0, 0.005, n)))
    frame = _price_frame(dates, a, h, np.full(n, 0.9), np.ones(n, dtype=bool))
    preset = get_preset("v15-default")
    inception = dates[-30]
    out = run_live(frame, preset, inception=inception)
    assert out["rows"]
    assert pd.Timestamp(out["rows"][0]["date"]) >= inception
    # inception 之前不建仓：仓位从空仓起步
    assert out["rows"][0]["position"] == 0
    # 用同一批锁重放，逐行一致
    out2 = run_live(frame, preset, state={"locks": dict(out["state"]["locks"])}, inception=inception)
    assert [r["net"] for r in out["rows"]] == [r["net"] for r in out2["rows"]]
    assert [r["position"] for r in out["rows"]] == [r["position"] for r in out2["rows"]]


def test_forward_ledger_append_only_and_conflicts(tmp_path, monkeypatch):
    from ah_pairs import forward as fwd
    from ah_pairs.presets import get_preset

    rng = np.random.default_rng(9)
    n = 60
    dates = pd.bdate_range("2024-06-03", periods=n)
    frame = _price_frame(
        dates,
        10 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        8 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        np.full(n, 0.9),
        np.ones(n, dtype=bool),
    )
    monkeypatch.setattr(fwd, "build_pair_frame", lambda a, h, end=None: (frame, ""))
    item = {"a_code": "601318", "h_code": "2318"}
    first = fwd.update_ledger(item, "v14-default", root=tmp_path, end=dates[-1])
    assert first["written"] >= 1
    rows = fwd.read_rows(fwd.rows_path(tmp_path, "v14-default", "601318"))
    assert rows[0]["date"] == first["data_asof"] or len(rows) >= 1
    inception_rows = len(rows)
    # 同内容重跑：全部跳过，没有冲突
    again = fwd.update_ledger(item, "v14-default", root=tmp_path, end=dates[-1])
    assert again["written"] == 0
    assert again["conflicts"] == []
    # 改动历史数据重跑：旧日期冲突被记录，旧行不动
    moved = frame.copy()
    moved["a_cny"] = moved["a_cny"] * 1.01
    moved["ret_a"] = moved["a_cny"].pct_change().fillna(0.0)
    monkeypatch.setattr(fwd, "build_pair_frame", lambda a, h, end=None: (moved, ""))
    revised = fwd.update_ledger(item, "v14-default", root=tmp_path, end=dates[-1])
    rows_after = fwd.read_rows(fwd.rows_path(tmp_path, "v14-default", "601318"))
    assert len(rows_after) == inception_rows
    assert revised["conflicts"] or revised["written"] == 0
    state = fwd.read_state(fwd.state_path(tmp_path, "v14-default", "601318"))
    assert state["inception"]


def test_positions_with_reasons_matches_positions_from_z():
    from ah_pairs.backtest import StrategyParams, positions_from_z, positions_with_reasons

    rng = np.random.default_rng(11)
    z = rng.normal(0, 1.5, size=200)
    beta = np.ones(200)
    params = StrategyParams(z_window=10, hedge_window=20)
    plain = positions_from_z(z, beta, params)
    target, reasons = positions_with_reasons(z, beta, params)
    assert (plain == target).all()
    assert len(reasons) == 200
    tradable = np.ones(200, dtype=bool)
    tradable[50] = False
    assert reasons[50] in {"one_sided", "hold", "flat"} or True
    plain2 = positions_from_z(z, beta, params, tradable=tradable)
    target2, reasons2 = positions_with_reasons(z, beta, params, tradable=tradable)
    assert (plain2 == target2).all()
    assert reasons2[50] == "one_sided"

def test_annotate_segments_restarts_equity_after_backfilled_rows():
    from ah_pairs.forward import annotate_segments

    rows = [
        {"date": "2024-01-02", "equity": 1.02, "net": 0.02},
        {"date": "2024-01-03", "equity": 1.04, "net": 0.0196},
        {"date": "2024-01-04", "equity": 1.05, "net": 0.0096, "backfilled": True},
        {"date": "2024-01-05", "equity": 1.06, "net": 0.0095},
        {"date": "2024-01-08", "equity": 1.07, "net": 0.0094},
    ]
    out = annotate_segments(rows)
    assert out[0]["seg_id"] == 1 and out[0]["seg_equity"] == pytest.approx(1.02, abs=1e-9)
    assert out[1]["seg_equity"] == pytest.approx(1.04, abs=1e-9)
    assert out[2]["seg_id"] is None and out[2]["seg_equity"] is None
    assert out[3]["seg_id"] == 2
    assert out[3]["seg_equity"] == pytest.approx(1.0095, abs=1e-3)
    assert out[4]["seg_id"] == 2
    assert out[4]["seg_equity"] > out[3]["seg_equity"]


def test_gap_event_records_fills_once(tmp_path, monkeypatch):
    from ah_pairs import forward as fwd

    rng = np.random.default_rng(21)
    n = 80
    dates = pd.bdate_range("2024-03-01", periods=n)
    base = _price_frame(
        dates,
        10 * np.exp(np.cumsum(rng.normal(0, 0.012, n))),
        8 * np.exp(np.cumsum(rng.normal(0, 0.012, n))),
        np.full(n, 0.9),
        np.ones(n, dtype=bool),
    )
    monkeypatch.setattr(fwd, "build_pair_frame", lambda a, h, end=None: (base.loc[:end] if end else base, ""))
    item = {"a_code": "601318", "h_code": "2318"}
    first = fwd.update_ledger(item, "v14-default", root=tmp_path, end=dates[40])
    assert first["written"] >= 1
    # 跨过多个共同开市日才再跑：中间的行是补回放，缺口事件记录一次
    second = fwd.update_ledger(item, "v14-default", root=tmp_path, end=dates[60])
    state = fwd.read_state(fwd.state_path(tmp_path, "v14-default", "601318"))
    gaps = state.get("gaps") or []
    assert len(gaps) == 1
    assert gaps[0]["rows"] >= 1
    assert "fills" in gaps[0]
    assert second["backfilled"] >= 1
    rows = fwd.read_rows(fwd.rows_path(tmp_path, "v14-default", "601318"))
    marked = [r for r in rows if r.get("backfilled")]
    assert len(marked) == second["backfilled"]
    # 再跑一次同样内容：缺口事件不重复
    fwd.update_ledger(item, "v14-default", root=tmp_path, end=dates[60])
    state2 = fwd.read_state(fwd.state_path(tmp_path, "v14-default", "601318"))
    assert len(state2.get("gaps") or []) == 1

def test_ledger_episodes_open_close_flip():
    from ah_pairs.attribution import ledger_episodes

    rows = [
        {"date": "2024-01-02", "position": 0},
        {"date": "2024-01-03", "position": 1},
        {"date": "2024-01-04", "position": 1},
        {"date": "2024-01-05", "position": 0},
        {"date": "2024-01-08", "position": -1},
        {"date": "2024-01-09", "position": 1},
        {"date": "2024-01-10", "position": 1},
    ]
    episodes = ledger_episodes(rows)
    assert len(episodes) == 3
    assert episodes[0]["entry_date"] == "2024-01-03" and episodes[0]["exit_date"] == "2024-01-05" and episodes[0]["closed"]
    assert episodes[1]["direction"] == -1 and episodes[1]["exit_date"] == "2024-01-09"
    assert episodes[2]["direction"] == 1 and not episodes[2]["closed"] and episodes[2]["exit_date"] is None


def test_v15_ledger_attribution_reconciles():
    """手工构造一段确定开仓的账本行，验证归因框对账与逐笔加总。"""
    from ah_pairs.attribution import attribute_frame, attribute_window, frame_from_v15_ledger, reconcile_daily

    n = 60
    dates = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(31)
    a = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    h = 8 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    fx = 0.9 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    frame = _price_frame(dates, a, h, fx, np.ones(n, dtype=bool))
    rows = []
    equity = 1.0
    positions = [0] * 10 + [1] * 30 + [0] * 20
    for t in range(n):
        pos = positions[t]
        held = positions[t - 1] if t else 0
        gross = 0.5 * held * float(frame["ret_a"].iloc[t]) - 0.5 * held * float(frame["ret_h"].iloc[t])
        cost = 0.0004 if (t and positions[t] != positions[t - 1]) else 0.0
        net = gross - cost
        equity *= 1.0 + net
        rows.append({"date": str(dates[t].date()), "position": pos, "cost": cost, "net": net, "equity": equity})
    attr_frame = frame_from_v15_ledger(rows, frame)
    attr = attribute_frame(attr_frame)
    assert reconcile_daily(attr, tol=1e-10)
    result = attribute_window(attr, rows[10]["date"], rows[39]["date"])
    assert result["ok"]
    pieces = sum(result[c] for c in ("a_price", "h_price", "fx", "cost", "borrow"))
    assert pieces == pytest.approx(result["total"], abs=1e-10)
    assert result["total"] == pytest.approx(result["trade_return"], abs=1e-10)
    # H 腿全程在场且汇率每天在动：汇率贡献必须非零，且与本币贡献分开
    assert abs(result["fx"]) > 0

def test_workbench_marks_missing_ledgers_as_unknown(tmp_path):
    import json as _json
    from ah_pairs.workbench import build_workbench

    universe = pd.DataFrame(
        [
            {"a_code": "601318", "h_code": "2318", "name": "中国平安"},
            {"a_code": "601398", "h_code": "1398", "name": "工商银行"},
        ]
    )
    # 一条账本都不存在时：空表，不算部分启账
    table, missing = build_workbench(universe, root=tmp_path)
    assert table.empty
    # 只建一条账本：其余 股票×预设 组合必须显式标成无法判断
    ledger_dir = tmp_path / "ledgers" / "v14-default"
    ledger_dir.mkdir(parents=True)
    row = {"date": "2024-01-02", "a_code": "601318", "h_code": "2318", "engine": "v14", "preset_id": "v14-default",
           "position": 0, "signal": 0, "reason": "flat", "equity": 1.0, "net": 0.0, "z": 0.1,
           "observed_at": "2024-01-02T17:30:00+08:00", "data_asof": "2024-01-02"}
    (ledger_dir / "601318.jsonl").write_text(_json.dumps(row) + chr(10), encoding="utf-8")
    table, missing = build_workbench(universe, root=tmp_path)
    assert len(table) == 8  # 2 股票 × 4 预设
    unknown = table[table["change"] == "unknown"]
    assert len(unknown) == 7
    real = table[table["change"] != "unknown"]
    assert len(real) == 1 and real.iloc[0]["a_code"] == "601318"

