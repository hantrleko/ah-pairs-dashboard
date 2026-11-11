"""V1.5 验收。只用合成序列和 tests/fixtures，不访问网络。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ah_pairs.v15.adaptive import (
    execute,
    half_life_from_phi,
    ar1_phi,
    rolling_no_trade_share,
    synthetic_ar1,
)
from ah_pairs.v15.costs import a_stamp_rate, cost_parts_ok, explicit_round_trip_bp, fill_cost, h_stamp_hkd, h_stamp_rate
from ah_pairs.v15.net_premium import (
    clean_dividend_calendar,
    dividend_gap_note,
    log_premium,
    net_z,
    tau_a_for_holding_days,
    wedge_annual,
    wedge_is_implausible,
)
from ah_pairs.v15.walkforward import choose_for_test_start, flag_implausible, reality_check

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dividend_calendar.csv"


def _frame(dates, a, h, fx, both, ret_a=None, ret_h=None) -> pd.DataFrame:
    a = np.asarray(a, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    fx = np.asarray(fx, dtype=np.float64)
    out = pd.DataFrame(
        {
            "a_cny": a,
            "h_hkd": h,
            "fx": fx,
            "h_cny": h * fx,
            "both_open": np.asarray(both, dtype=bool),
        },
        index=pd.DatetimeIndex(pd.to_datetime(dates)),
    )
    out["log_a"] = np.log(out["a_cny"])
    out["log_h"] = np.log(out["h_cny"])
    out["ret_a"] = out["a_cny"].pct_change().fillna(0.0) if ret_a is None else np.asarray(ret_a, dtype=np.float64)
    out["ret_h"] = out["h_cny"].pct_change().fillna(0.0) if ret_h is None else np.asarray(ret_h, dtype=np.float64)
    return out


def test_half_life_converges_on_synthetic_ar1():
    target = 20.0
    n = 244 * 8
    hits = 0
    for seed in range(200):
        series = synthetic_ar1(target, n, seed)
        estimate = half_life_from_phi(ar1_phi(series))
        if abs(estimate / target - 1.0) < 0.35:
            hits += 1
    assert hits / 200 >= 0.90


def test_random_walk_spread_is_mostly_no_trade():
    rng = np.random.default_rng(1)
    shocks = rng.normal(0.0, 1.0, size=1400)
    spread = np.cumsum(shocks)
    share = rolling_no_trade_share(spread, window=1200)
    assert share >= 0.95


def test_cost_identity_and_round_trip_band():
    buy = fill_cost("2024-06-03", "A", "buy", 100_000.0, 0.9, half_spread=0.0005, adv_cny=1_000_000.0, charge_fx=False)
    sell = fill_cost("2024-06-04", "A", "sell", 110_000.0, 0.9, half_spread=0.0005, adv_cny=1_000_000.0, charge_fx=False)
    assert cost_parts_ok(buy) and cost_parts_ok(sell)
    gross = 10_000.0
    net = gross - buy["total"] - sell["total"]
    gap = abs((gross - net) - (buy["total"] + sell["total"]))
    scale = max(abs(buy["total"] + sell["total"]), 1.0)
    assert gap / scale < 1e-8
    base = explicit_round_trip_bp("2024-06-03")
    assert 25.0 <= base <= 60.0
    assert 15.0 <= base <= 100.0


def test_stamp_rates_follow_the_effective_dates():
    assert a_stamp_rate("2023-08-27") == pytest.approx(2.0 * a_stamp_rate("2023-08-28"))
    before = h_stamp_hkd(1_000_000.0, "2023-11-16")
    after = h_stamp_hkd(1_000_000.0, "2023-11-17")
    assert before / after == pytest.approx(1.30)
    assert h_stamp_rate("2023-11-16") / h_stamp_rate("2023-11-17") == pytest.approx(1.30)
    assert h_stamp_hkd(10_001.0, "2024-01-02") == pytest.approx(11.0)


def test_adjusted_premium_does_not_jump_on_an_ex_date():
    n = 80
    clock = np.arange(n, dtype=np.float64)
    a_adj = 100.0 * np.exp(0.002 * np.sin(clock / 8.0))
    h = np.full(n, 50.0)
    fx = np.full(n, 0.9)
    a_raw = a_adj.copy()
    a_raw[60:] = a_adj[60:] * 0.95
    z_raw = net_z(log_premium(a_raw, h, fx), np.zeros(n), 40)
    z_adj = net_z(log_premium(a_adj, h, fx), np.zeros(n), 40)
    assert abs(float(z_raw.iloc[60] - z_raw.iloc[59])) > 2.0
    assert abs(float(z_adj.iloc[60] - z_adj.iloc[59])) < 0.5


def test_wedge_scale_and_centered_z():
    annual = wedge_annual(0.035, 0.020, 0.10, 0.20)
    assert 0.005 <= annual <= 0.02
    assert wedge_is_implausible(0.06)
    assert not wedge_is_implausible(annual)
    rng = np.random.default_rng(5)
    premium = pd.Series(rng.normal(0.0, 0.002, size=160), dtype="float64")
    wedge = np.full(len(premium), 0.01)
    z = net_z(premium, wedge, 40).dropna()
    assert abs(float(z.mean())) < 0.15
    naive = (premium - premium.rolling(40, min_periods=40).mean() - 0.01) / premium.rolling(40, min_periods=40).std(ddof=1)
    assert abs(float(naive.dropna().mean())) > 0.15


def test_parameter_choice_ignores_the_test_fold():
    rng = np.random.default_rng(6)
    n = 180
    dates = pd.bdate_range("2020-01-01", periods=n)
    shock = rng.normal(0.0, 0.01, size=n)
    a = 10.0 * np.exp(np.cumsum(shock))
    h = 8.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))
    fx = np.full(n, 0.9)
    frame = _frame(dates, a, h, fx, np.ones(n, dtype=bool))
    start = dates[120]
    kwargs = dict(train=40, embargo=5, inner_train=15, inner_valid=10, min_trades=0, z_window=20)
    original = choose_for_test_start(frame, start, **kwargs)
    truncated = choose_for_test_start(frame.iloc[:120], start, **kwargs)
    leaked = frame.copy()
    leaked.loc[leaked.index >= start, "a_cny"] *= 1.4
    leaked["log_a"] = np.log(leaked["a_cny"].to_numpy(dtype=np.float64))
    future = choose_for_test_start(leaked, start, **kwargs)
    assert original == truncated == future


def test_reality_check_repeats_for_the_same_seed():
    rng = np.random.default_rng(8)
    data = pd.DataFrame(rng.normal(0.0002, 0.01, size=(80, 6)))
    first = reality_check(data, seed=11, n_boot=40, block=5)
    second = reality_check(data, seed=11, n_boot=40, block=5)
    assert first["rc_p"] == second["rc_p"]
    assert first["spa_p"] == second["spa_p"]
    assert first["spa_p_lower"] == second["spa_p_lower"]
    assert first["spa_p_upper"] == second["spa_p_upper"]
    assert first["seed"] == 11


def test_one_sided_day_marks_the_open_leg_and_does_not_trade():
    dates = pd.bdate_range("2024-06-03", periods=4)
    frame = _frame(
        dates,
        np.full(4, 10.0),
        np.full(4, 8.0),
        np.full(4, 0.9),
        [True, True, False, True],
        ret_a=[0.0, 0.0, 0.01, 0.0],
        ret_h=[0.0, 0.0, 0.0, 0.0],
    )
    z = np.array([-3.0, -3.0, -3.0, -3.0])
    result = execute(frame, z, z_in=2.0, z_out=0.5, max_hold=30, gate=np.ones(4, dtype=bool))
    assert result["position"][2] == 1
    assert result["cost"][2] == 0.0
    assert result["gross"][2] == pytest.approx(0.5 * 0.01)
    assert all(row["date"] != dates[2] for row in result["rows"])


def test_fill_waits_for_the_next_common_session():
    dates = pd.bdate_range("2024-06-03", periods=3)
    frame = _frame(
        dates,
        np.full(3, 10.0),
        np.full(3, 8.0),
        np.full(3, 0.9),
        [True, False, True],
        ret_a=[0.0, 0.02, 0.0],
        ret_h=[0.0, 0.03, 0.0],
    )
    result = execute(frame, np.array([-3.0, -3.0, -3.0]), 2.0, 0.5, 30, gate=np.ones(3, dtype=bool))
    assert list(result["position"]) == [0, 0, 1]
    assert result["cost"][1] == 0.0
    assert result["gross"][1] == 0.0
    assert {row["date"] for row in result["rows"]} == {dates[2]}
    assert {row["leg"] for row in result["rows"]} == {"A", "H"}


def test_implausible_oos_sharpe_is_flagged():
    assert flag_implausible(2.01)
    assert not flag_implausible(2.0)
    assert not flag_implausible(0.4)


def test_dividend_calendar_fixture_drops_bad_rows_and_records_a_gap():
    raw = pd.read_csv(FIXTURE)
    sessions = pd.bdate_range("2024-06-03", periods=5)
    cleaned, log = clean_dividend_calendar(raw, sessions, sessions)
    assert set(cleaned["leg"]) == {"A", "H"}
    assert (cleaned["dps"] > 0).all()
    assert any("不是正数" in line for line in log)
    assert any("不在该腿开市日" in line for line in log)
    assert dividend_gap_note(None, sessions[0], sessions[-1])
    assert tau_a_for_holding_days(30) == 0.20
    assert tau_a_for_holding_days(31) == 0.10
    assert tau_a_for_holding_days(366) == 0.0


def test_v14_engine_is_still_the_rollback_path():
    from ah_pairs.engine_v14 import ENGINE, run
    from ah_pairs.v15.config import DEFAULT_ENGINE

    assert ENGINE == "v14"
    assert DEFAULT_ENGINE == "v14"
    assert callable(run)
