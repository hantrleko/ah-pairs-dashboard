"""合成价格上的协整、前视和回测检验。不访问网络。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ah_pairs.analysis import entry_sensitivity
from ah_pairs.backtest import StrategyParams, run_backtest
from ah_pairs.cointegration import engle_granger, ou_half_life, residual_adf
from ah_pairs.costs import CostParams, a_share_stamp, h_share_stamp, leg_turnover_cost
from ah_pairs.hedge import kalman_hedge, rolling_ols
from ah_pairs.universe import a_to_yahoo, h_to_yahoo, load_universe


def _zero_costs() -> CostParams:
    return CostParams(
        a_commission=0.0,
        h_commission=0.0,
        a_borrow_annual=0.0,
        h_borrow_annual=0.0,
        use_statutory_stamp=False,
        a_stamp_sell=0.0,
        h_stamp=0.0,
        slippage=0.0,
    )


def _params(**kwargs) -> StrategyParams:
    base = dict(
        hedge_window=40,
        z_window=20,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=0.0,
        allow_short_a=True,
        fill="next_close",
        beta_band=0.10,
        costs=_zero_costs(),
    )
    base.update(kwargs)
    return StrategyParams(**base)


def _book(z, beta, ret_a=None, ret_h=None) -> pd.DataFrame:
    n = len(z)
    index = pd.bdate_range("2023-01-02", periods=n)
    if ret_a is None:
        ret_a = np.zeros(n)
    if ret_h is None:
        ret_h = np.zeros(n)
    return pd.DataFrame(
        {"z": z, "beta": beta, "ret_a": ret_a, "ret_h": ret_h},
        index=index,
    )


def _cointegrated_pair(n=1200, phi=0.95, beta=1.1, seed=7):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.01, n))
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = phi * eps[t - 1] + rng.normal(0, 0.005)
    y = 0.2 + beta * x + eps
    return y, x


def test_yahoo_codes_and_universe():
    assert a_to_yahoo("601318") == "601318.SS"
    assert a_to_yahoo("688009") == "688009.SS"
    assert a_to_yahoo("002594") == "002594.SZ"
    assert a_to_yahoo("000333") == "000333.SZ"
    assert h_to_yahoo("02318") == "2318.HK"
    assert h_to_yahoo("00857") == "0857.HK"
    assert h_to_yahoo("00300") == "0300.HK"
    universe = load_universe()
    ping_an = universe.loc[universe["a_code"] == "601318"].iloc[0]
    assert ping_an["h_code"] == "02318"
    assert ping_an["h_yahoo"] == "2318.HK"


def test_stamp_schedule():
    params = CostParams()
    assert a_share_stamp("2023-08-27", params) == (0.0, 0.001)
    assert a_share_stamp("2023-08-28", params) == (0.0, 0.0005)
    assert a_share_stamp("2008-09-18", params) == (0.001, 0.001)
    assert h_share_stamp("2021-07-30", params) == pytest.approx(0.001)
    assert h_share_stamp("2021-08-01", params) == pytest.approx(0.0013)
    assert h_share_stamp("2023-11-17", params) == pytest.approx(0.001)


def test_leg_cost_splits_long_and_short():
    # 平掉 0.5 的多头再开 0.2 的空头：两笔都是卖出。
    cost = leg_turnover_cost(0.5, -0.2, commission=0.01, stamp_buy=0.0, stamp_sell=0.02)
    assert cost == pytest.approx(0.5 * 0.03 + 0.2 * 0.03)


def test_half_life_and_engle_granger_recover_the_process():
    y, x = _cointegrated_pair()
    hl = ou_half_life(pd.Series(y - 0.2 - 1.1 * x))
    assert hl["mean_reverting"]
    assert 8 < hl["half_life"] < 25
    eg = engle_granger(y, x)
    assert eg["pvalue"] < 0.05
    assert eg["beta"] == pytest.approx(1.1, abs=0.05)
    assert eg["reject_5"]


def test_residual_adf_is_separate_from_the_cointegration_pvalue():
    y, x = _cointegrated_pair()
    eg = engle_granger(y, x)
    adf = residual_adf(eg["resid"])
    assert adf["adf_pvalue"] < 0.05
    assert "不能代替" in adf["note"]


def test_pick_lowest_p_uses_the_smallest_finite_value():
    from ah_pairs.analysis import pick_lowest_p

    table = pd.DataFrame(
        {
            "公司": ["平安", "工行", "缺数"],
            "eg_pvalue": [0.27, 0.04, np.nan],
        }
    )
    picked = pick_lowest_p(table)
    assert picked["公司"] == "工行"
    assert pick_lowest_p(pd.DataFrame()) is None


def test_entry_grid_marks_the_threshold_in_use():
    z = np.concatenate([np.zeros(2), np.full(6, -2.4), np.zeros(4)])
    frame = _book(z, np.ones(len(z)))
    params = _params(fill="same_close", entry_z=2.0, exit_z=0.5, stop_z=0.0)
    table = entry_sensitivity(frame, params, entries=(1.5, 2.0, 2.5))
    assert list(table["entry_z"]) == [1.5, 2.0, 2.5]
    assert bool(table.loc[table["entry_z"] == 2.0, "current"].iloc[0])
    assert not bool(table.loc[table["entry_z"] == 1.5, "current"].iloc[0])


def test_cointegrated_pair_survives_more_windows_than_two_random_walks():
    from ah_pairs.stability import fragility_report, stability_grid

    y, x = _cointegrated_pair(n=800, phi=0.6, seed=1)
    index = pd.bdate_range("2018-01-02", periods=len(y))
    stable = fragility_report(stability_grid(y, x, index, lengths=(200, 400), step=200))
    rng = np.random.default_rng(9)
    y2 = np.cumsum(rng.normal(0, 0.01, len(y)))
    x2 = np.cumsum(rng.normal(0, 0.01, len(y)))
    fragile = fragility_report(stability_grid(y2, x2, index, lengths=(200, 400), step=200))
    assert stable["reject_rate"] > fragile["reject_rate"]
    assert stable["reject_rate"] >= 0.8
    assert stable["n_tests"] > 0


def test_fragility_report_ranks_highest_p_and_handles_empty_grid():
    from ah_pairs.stability import fragility_report

    grid = pd.DataFrame(
        {
            "start": pd.to_datetime(["2020-01-01", "2020-06-01"]),
            "end": pd.to_datetime(["2021-01-01", "2021-06-01"]),
            "length": [200, 400],
            "pvalue": [0.01, 0.80],
            "reject_5": [True, False],
        }
    )
    report = fragility_report(grid)
    assert report["n_tests"] == 2
    assert report["reject_rate"] == 0.5
    assert report["fragility"] == 0.5
    assert report["worst"].iloc[0]["pvalue"] == 0.8
    empty = fragility_report(pd.DataFrame(columns=["pvalue", "length", "reject_5"]))
    assert empty["n_tests"] == 0
    assert np.isnan(empty["fragility"])


def test_length_summary_reports_reject_rate_by_window():
    from ah_pairs.stability import length_summary

    grid = pd.DataFrame(
        {
            "length": [126, 126, 252],
            "pvalue": [0.01, 0.20, 0.04],
            "reject_5": [True, False, True],
        }
    )
    summary = length_summary(grid)
    assert list(summary["length"]) == [126, 252]
    assert summary.loc[summary["length"] == 126, "n_tests"].iloc[0] == 2
    assert summary.loc[summary["length"] == 126, "reject_rate"].iloc[0] == 0.5
    assert length_summary(pd.DataFrame(columns=["pvalue", "length", "reject_5"])).empty


def test_price_is_stale_only_when_the_last_bar_lags_the_horizon():
    from ah_pairs.freshness import northbound_stops_before, price_is_stale

    assert not price_is_stale("2026-10-13", "2026-10-13", "2026-10-13")
    assert price_is_stale("2026-09-01", "2026-10-13", "2026-10-13")
    assert not price_is_stale("2020-12-31", "2020-12-31", "2026-10-13")
    assert northbound_stops_before("2026-10-13", "2024-08-16")
    assert not northbound_stops_before("2024-08-16", "2024-08-16")


def test_later_window_and_zero_cost_do_not_reuse_a_new_parameter_search():
    from ah_pairs.backtest import contrast_costs, later_window_performance

    idx = pd.bdate_range("2020-01-02", periods=80)
    frame = pd.DataFrame(
        {
            "z": np.concatenate([np.zeros(10), np.full(20, 3.0), np.zeros(50)]),
            "beta": np.ones(80),
            "ret_a": np.full(80, 0.001),
            "ret_h": np.zeros(80),
        },
        index=idx,
    )
    params = _params(
        costs=CostParams(
            a_commission=0.01,
            h_commission=0.0,
            a_borrow_annual=0.0,
            h_borrow_annual=0.0,
            use_statutory_stamp=False,
            a_stamp_sell=0.0,
            h_stamp=0.0,
            slippage=0.0,
        )
    )
    table = contrast_costs(frame, params)
    assert list(table["情形"]) == ["当前成本", "不计成本", "滑点再加 10 bps"]
    assert table.loc[table["情形"] == "不计成本", "成本合计"].iloc[0] == 0
    assert table.loc[table["情形"] == "当前成本", "成本合计"].iloc[0] > 0
    worked, trades, _stats = run_backtest(frame, params)
    later = later_window_performance(worked, trades)
    assert later["ok"]
    assert later["n_days"] == 40
    assert pd.Timestamp(later["start"]) > idx[0]


def test_hedge_ratio_ignores_the_future():
    y, x = _cointegrated_pair(n=600, seed=3)
    y2 = y.copy()
    x2 = x.copy()
    y2[400:] += 3
    x2[400:] -= 2
    _a1, b1 = rolling_ols(y, x, 80)
    _a2, b2 = rolling_ols(y2, x2, 80)
    np.testing.assert_allclose(b1[:401], b2[:401], equal_nan=True)
    k1, kb1 = kalman_hedge(y, x, delta=1e-4, init_window=80)
    k2, kb2 = kalman_hedge(y2, x2, delta=1e-4, init_window=80)
    np.testing.assert_allclose(kb1[:401], kb2[:401], equal_nan=True)
    assert np.isfinite(b1[100])
    assert np.isnan(b1[79])


def test_next_close_flags_a_pending_order_when_the_signal_changes_at_the_last_bar():
    from ah_pairs.backtest import current_state

    idx = pd.bdate_range("2020-01-02", periods=30)
    z = np.zeros(30)
    z[-1] = 3.0
    frame = pd.DataFrame(
        {"z": z, "beta": np.ones(30), "ret_a": np.zeros(30), "ret_h": np.zeros(30)},
        index=idx,
    )
    work, trades, _stats = run_backtest(frame, _params(fill="next_close"))
    state = current_state(work, trades, "next_close")
    assert state["position"] == 0
    assert state["signal"] == -1
    assert state["pending"]
    same, same_trades, _stats = run_backtest(frame, _params(fill="same_close"))
    flat = current_state(same, same_trades, "same_close")
    assert flat["pending"] is False
    assert flat["position"] == -1


def test_walk_forward_locks_the_entry_chosen_on_the_first_half():
    from ah_pairs.analysis import walk_forward

    idx = pd.bdate_range("2018-01-02", periods=160)
    z = np.zeros(160)
    z[20:40] = 3.0
    z[100:120] = 3.0
    frame = pd.DataFrame(
        {"z": z, "beta": np.ones(160), "ret_a": np.full(160, 0.0005), "ret_h": np.zeros(160)},
        index=idx,
    )
    result = walk_forward(frame, _params(fill="same_close"))
    assert result["ok"]
    table = result["table"]
    chosen = result["chosen_entry"]
    assert table.loc[table["区间"] == "验证（锁定参数）", "开仓 |z|"].iloc[0] == chosen
    assert table.loc[table["区间"] == "训练（选出参数）", "开仓 |z|"].iloc[0] == chosen


def test_pick_best_sharpe_is_not_the_lowest_p():
    from ah_pairs.analysis import pick_best_sharpe, pick_lowest_p

    table = pd.DataFrame(
        {
            "公司": ["石油", "平安"],
            "eg_pvalue": [0.007, 0.269],
            "sharpe": [-0.38, 0.10],
            "total_return": [-0.05, 0.02],
        }
    )
    assert pick_lowest_p(table)["公司"] == "石油"
    assert pick_best_sharpe(table)["公司"] == "平安"


def test_next_close_does_not_earn_the_signal_day_and_respects_t_plus_one():
    z = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    beta = np.ones(len(z))
    ret_a = np.zeros(len(z))
    ret_h = np.zeros(len(z))
    ret_a[3] = 0.02  # 信号次日，仓位还没成交，不该赚到
    ret_a[4] = 0.01  # 成交后的第一天
    frame, trades, _stats = run_backtest(_book(z, beta, ret_a, ret_h), _params())
    assert frame["position"].tolist() == [0, 0, 0, -1, 0, 0]
    # 空价差、beta=1：w_a=-0.5。只在持仓日赚到 A 股收益。
    assert frame["ret_strat"].iloc[3] == pytest.approx(0.0)
    assert frame["ret_strat"].iloc[4] == pytest.approx(-0.5 * 0.01)
    assert len(trades) == 1
    assert trades.iloc[0]["holding_days"] == 1
    assert trades.iloc[0]["entry_z"] == pytest.approx(3.0)


def test_disallowing_a_share_short_drops_the_rich_a_share_signal():
    z = np.array([0.0, 3.0, 3.0, -3.0, -3.0, -3.0])
    beta = np.ones(len(z))
    frame, _trades, _stats = run_backtest(
        _book(z, beta), _params(allow_short_a=False, fill="same_close")
    )
    assert (frame["w_a"] < -1e-12).sum() == 0
    assert (frame["position"] == 1).any()
    assert (frame["position"] == -1).sum() == 0


def test_costs_reduce_equity_and_beta_band_does_not_churn():
    z = np.full(8, -3.0)
    beta = np.array([1.0, 1.0, 1.01, 1.01, 1.01, 1.50, 1.50, 1.50])
    free, _t, free_stats = run_backtest(
        _book(z, beta), _params(fill="same_close", beta_band=0.10)
    )
    costly_params = _params(
        fill="same_close",
        beta_band=0.10,
        costs=CostParams(
            a_commission=0.01,
            h_commission=0.01,
            a_borrow_annual=0.0,
            h_borrow_annual=0.0,
            use_statutory_stamp=False,
            a_stamp_sell=0.01,
            h_stamp=0.01,
            slippage=0.0,
        ),
    )
    costly, _t2, costly_stats = run_backtest(_book(z, beta), costly_params)
    assert costly["equity"].iloc[-1] < free["equity"].iloc[-1]
    assert costly_stats["total_cost"] > 0
    # 前五天 beta 只从 1 漂到 1.01，不应该改权重。第 6 天 1.50 才再平衡。
    assert free["w_a"].iloc[0] == pytest.approx(0.5)
    assert free["w_a"].iloc[4] == pytest.approx(0.5)
    assert free["w_a"].iloc[5] == pytest.approx(1 / 2.5)
    assert free_stats["n_trades"] == 0  # 到样本结束还没平仓
    assert free_stats["n_open"] == 1


def test_default_strategy_does_not_short_a_shares():
    assert StrategyParams().allow_short_a is False


def test_rolling_ols_matches_the_normal_equation_and_stays_causal():
    y, x = _cointegrated_pair(n=300, seed=2)
    alpha, beta = rolling_ols(y, x, 80)
    for t in (100, 200):
        design = np.column_stack([np.ones(80), x[t - 80 : t]])
        coef = np.linalg.lstsq(design, y[t - 80 : t], rcond=None)[0]
        assert alpha[t] == pytest.approx(coef[0], rel=1e-6, abs=1e-6)
        assert beta[t] == pytest.approx(coef[1], rel=1e-6, abs=1e-6)


def test_kalman_delta_estimated_on_a_prefix_ignores_the_future():
    from ah_pairs.analysis import analyze_frame

    y, x = _cointegrated_pair(n=400, phi=0.8, seed=5)
    index = pd.bdate_range("2016-01-04", periods=len(y))

    def prices_for(series):
        return pd.DataFrame(
            {
                "log_a": series,
                "log_h": x,
                "a_cny": np.exp(series),
                "h_hkd": np.exp(x),
                "fx": 1.0,
                "h_cny": np.exp(x),
                "premium": 0.0,
                "ret_a": 0.0,
                "ret_h": 0.0,
                "both_open": True,
            },
            index=index,
        )

    params = _params(hedge="kalman", kalman_delta_auto=True, hedge_window=60, z_window=20)
    shifted = y.copy()
    shifted[250:] += 4.0
    original = analyze_frame(prices_for(y), params, eval_start=index[250], full_diagnostics=False)
    moved = analyze_frame(prices_for(shifted), params, eval_start=index[250], full_diagnostics=False)
    assert original["kalman_fit"]["reason"] == "warmup"
    assert original["kalman_fit"]["delta"] == moved["kalman_fit"]["delta"]
    assert original["kalman_fit"]["delta"] > 0


def test_one_sided_day_keeps_the_open_position_and_does_not_fill_early():
    index = pd.bdate_range("2024-01-02", periods=4)
    held = pd.DataFrame(
        {
            "z": [-3.0, -3.0, -3.0, 0.0],
            "beta": np.ones(4),
            "ret_a": [0.0, 0.04, 0.0, 0.0],
            "ret_h": [0.0, 0.0, 0.0, 0.0],
            "both_open": [True, False, True, True],
        },
        index=index,
    )
    work, _trades, stats = run_backtest(held, _params(fill="same_close", stop_z=0.0))
    assert work["position"].tolist() == [1, 1, 1, 0]
    assert work["ret_strat"].iloc[1] == pytest.approx(0.5 * 0.04)
    assert stats["n_one_sided"] == 1
    pending = pd.DataFrame(
        {
            "z": [0.0, -3.0, -3.0, 0.0],
            "beta": np.ones(4),
            "ret_a": np.zeros(4),
            "ret_h": np.zeros(4),
            "both_open": [True, True, False, True],
        },
        index=index,
    )
    delayed, _trades, _stats = run_backtest(pending, _params(fill="next_close", stop_z=0.0))
    assert delayed["position"].tolist()[2] == 0
    assert delayed["position"].tolist()[3] == 1


def test_assemble_pair_counts_one_sided_sessions_and_zeros_the_closed_leg():
    from ah_pairs.data import assemble_pair

    index = pd.bdate_range("2024-01-02", periods=4)
    a = pd.Series([10.0, 11.0, 11.0, 12.0], index=index)
    h = pd.Series([10.0, np.nan, 10.0, 10.0], index=index)
    fx = pd.Series(1.0, index=index)
    frame, notes = assemble_pair(a, h, fx, index[0], index[-1])
    assert frame["both_open"].tolist() == [True, False, True, True]
    assert frame["ret_a"].iloc[1] == pytest.approx(0.1)
    assert frame["ret_h"].iloc[1] == pytest.approx(0.0)
    assert any("只有一边开市 1 天" in note for note in notes)


def test_cointegration_sample_drops_one_sided_days():
    from ah_pairs.analysis import analyze_frame

    y, x = _cointegrated_pair(n=220, phi=0.6, seed=4)
    index = pd.bdate_range("2018-01-02", periods=len(y))
    prices = pd.DataFrame(
        {
            "log_a": y,
            "log_h": x,
            "a_cny": np.exp(y),
            "h_hkd": np.exp(x),
            "fx": 1.0,
            "h_cny": np.exp(x),
            "premium": 0.0,
            "ret_a": 0.0,
            "ret_h": 0.0,
            "both_open": True,
        },
        index=index,
    )
    prices.loc[prices.index[10:25], "both_open"] = False
    result = analyze_frame(prices, _params(hedge_window=40, z_window=20), full_diagnostics=False)
    assert result["eg"]["nobs"] == int(prices["both_open"].sum())
    assert result["n_one_sided"] == 15


def test_benjamini_hochberg_uses_the_largest_ranked_threshold():
    from ah_pairs.stability import benjamini_hochberg

    modest = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert modest["m"] == 4
    assert modest["n_reject"] == 1
    strong = benjamini_hochberg([0.001, 0.001, 0.001])
    assert strong["n_reject"] == 3
    assert benjamini_hochberg([])["m"] == 0


def test_sharpe_interval_and_deflated_probability_are_bounded():
    from ah_pairs.inference import block_bootstrap_sharpe, sharpe_probability

    rng = np.random.default_rng(3)
    returns = rng.normal(0.0002, 0.01, 400)
    interval = block_bootstrap_sharpe(returns, block=10, n_boot=80, seed=1)
    assert interval["ok"]
    assert np.isfinite(interval["low"]) and np.isfinite(interval["high"])
    assert interval["low"] <= interval["high"]
    prob = sharpe_probability(returns, n_trials=8)
    assert prob["ok"]
    assert 0 <= prob["psr"] <= 1
    assert 0 <= prob["dsr"] <= 1
    assert prob["dsr"] <= prob["psr"] + 1e-8


def test_walk_forward_yearly_folds_lock_both_thresholds():
    from ah_pairs.analysis import walk_forward

    index = pd.bdate_range("2016-01-04", periods=800)
    z = np.zeros(800)
    z[20:40] = 3.0
    z[300:330] = -3.0
    z[600:630] = 3.0
    frame = pd.DataFrame(
        {"z": z, "beta": np.ones(800), "ret_a": np.full(800, 0.0004), "ret_h": np.zeros(800)},
        index=index,
    )
    result = walk_forward(frame, _params(fill="same_close"))
    assert result["ok"]
    assert result["scheme"] == "anchored-year"
    assert result["n_folds"] >= 2
    table = result["table"]
    checked = 0
    for label in table["区间"]:
        if not str(label).startswith("验证 "):
            continue
        year = str(label).split()[-1]
        train = table.loc[table["区间"] == f"训练 {year}"].iloc[0]
        valid = table.loc[table["区间"] == label].iloc[0]
        assert train["开仓 |z|"] == valid["开仓 |z|"]
        assert train["平仓 |z|"] == valid["平仓 |z|"]
        assert valid["平仓 |z|"] < valid["开仓 |z|"]
        checked += 1
    assert checked >= 2


def test_inverse_vol_weights_are_lagged_and_sum_to_one():
    from ah_pairs.portfolio import inverse_vol_weights

    index = pd.bdate_range("2020-01-02", periods=80)
    rng = np.random.default_rng(1)
    left = pd.Series(rng.normal(0.0, 0.01, 80), index=index)
    right = pd.Series(rng.normal(0.0, 0.02, 80), index=index)
    shocked = pd.DataFrame({"a": left, "b": right})
    shocked.iloc[70, 1] = 1.0
    calm = shocked.copy()
    calm.iloc[70, 1] = 0.0
    weights = inverse_vol_weights(shocked, lookback=20)
    other = inverse_vol_weights(calm, lookback=20)
    pd.testing.assert_series_equal(weights.iloc[70], other.iloc[70])
    active = weights.sum(axis=1) > 0
    assert np.allclose(weights.loc[active].sum(axis=1), 1.0)


def test_research_bundle_is_a_zip_and_png_export_has_a_header():
    from ah_pairs.bundle import research_zip
    from ah_pairs.export import chosen_cjk_font, spread_png

    frame, trades, stats = run_backtest(_book(np.array([0.0, -3.0, -3.0, 0.0]), np.ones(4)), _params(fill="same_close"))
    frame = frame.assign(spread=np.linspace(0.0, 1.0, len(frame)))
    result = {
        "frame": frame,
        "trades": trades,
        "stats": stats,
        "eg": {"pvalue": 0.2, "beta": 1.0, "nobs": 4, "t_stat": -2.0},
        "half_life_static": {"phi": 0.9, "half_life": 6.0},
        "residual_adf": {},
        "params": _params(),
        "kalman_fit": {"delta": 1e-4, "message": ""},
        "n_common": 4,
        "n_one_sided": 0,
    }
    blob = research_zip(result, "平安", "601318", "02318", 20)
    assert blob[:2] == b"PK"
    picture = spread_png(frame.assign(spread=np.linspace(0, 1, len(frame))), 2)
    assert picture is not None and picture.startswith(b"\x89PNG")
    assert chosen_cjk_font()


def test_price_cache_round_trips_through_parquet(tmp_path):
    from ah_pairs.data import _read_series, _write_series

    index = pd.bdate_range("2024-01-02", periods=3)
    series = pd.Series([1.0, 2.0, 3.0], index=index, name="close")
    path = tmp_path / "601318.SS.parquet"
    _write_series(path, series)
    loaded = _read_series(path)
    assert loaded.tolist() == [1.0, 2.0, 3.0]


def test_alert_payload_is_absent_until_z_crosses_the_entry():
    from ah_pairs.alerts import alert_payload

    quiet = {"z": 0.4, "position": 0, "signal": 0, "pending": False, "asof": pd.Timestamp("2026-10-13")}
    assert alert_payload("平安", "601318", "02318", quiet, 2.0) is None
    loud = dict(quiet, z=2.4, signal=-1, pending=True)
    payload = alert_payload("平安", "601318", "02318", loud, 2.0)
    assert payload["pending"] is True
    assert "不是下单" in payload["note"]


def test_language_toggle_uses_english_and_traditional():
    from ah_pairs.i18n import set_lang, tx

    set_lang("en")
    assert tx("协整", "cointegration") == "cointegration"
    set_lang("zh-Hant")
    traditional = tx("协整", "cointegration")
    set_lang("zh")
    assert traditional != "cointegration"
    try:
        import zhconv
    except Exception:
        return
    assert traditional == zhconv.convert("协整", "zh-hant")


def test_signal_token_ignores_thresholds_and_trade_token_does_not():
    from dataclasses import replace

    from ah_pairs.backtest import signal_token, trade_token

    params = _params()
    moved = replace(params, entry_z=2.5, exit_z=0.25)
    assert signal_token(params) == signal_token(moved)
    assert trade_token(params) != trade_token(moved)


def test_sharpe_annualization_follows_calendar_bars():
    from ah_pairs.backtest import sample_clock

    index = pd.bdate_range("2020-01-02", periods=400)
    rng = np.random.default_rng(0)
    z = np.zeros(400)
    z[10:40] = -3.0
    frame = pd.DataFrame(
        {
            "z": z,
            "beta": np.ones(400),
            "ret_a": rng.normal(0.0, 0.01, 400),
            "ret_h": rng.normal(0.0, 0.01, 400),
        },
        index=index,
    )
    work, _trades, stats = run_backtest(frame, _params(fill="same_close"))
    clock = sample_clock(work.index)
    assert stats["years"] == pytest.approx(clock["years"])
    assert stats["bars_per_year"] == pytest.approx(clock["bars_per_year"])
    rets = work["ret_strat"].to_numpy(dtype=float)
    rets = rets[np.isfinite(rets)]
    vol = float(np.std(rets, ddof=1))
    expected = float(np.mean(rets) / vol * np.sqrt(stats["bars_per_year"]))
    assert stats["sharpe"] == pytest.approx(expected)
    assert abs(stats["years"] - len(work) / 252.0) > 0.01


def test_cost_parts_sum_to_the_total_and_rebalance_is_separate():
    z = np.full(8, -3.0)
    beta = np.array([1.0, 1.0, 1.01, 1.01, 1.01, 1.50, 1.50, 1.50])
    params = _params(
        fill="same_close",
        beta_band=0.10,
        costs=CostParams(
            a_commission=0.01,
            h_commission=0.01,
            a_borrow_annual=0.0,
            h_borrow_annual=0.0,
            use_statutory_stamp=False,
            a_stamp_sell=0.01,
            h_stamp=0.01,
            slippage=0.001,
        ),
    )
    _work, _trades, stats = run_backtest(_book(z, beta), params)
    parts = (
        stats["cost_a_commission"]
        + stats["cost_a_stamp"]
        + stats["cost_h_commission"]
        + stats["cost_h_stamp"]
        + stats["cost_slippage"]
    )
    assert parts == pytest.approx(stats["total_cost"])
    assert stats["cost_turnover"] > 0
    assert stats["cost_rebalance"] > 0
    assert stats["cost_turnover"] + stats["cost_rebalance"] == pytest.approx(stats["total_cost"])


def test_walk_forward_stitches_validation_equity():
    from ah_pairs.analysis import walk_forward

    index = pd.bdate_range("2018-01-02", periods=160)
    z = np.zeros(160)
    z[20:40] = 3.0
    frame = pd.DataFrame(
        {"z": z, "beta": np.ones(160), "ret_a": np.full(160, 0.0005), "ret_h": np.zeros(160)},
        index=index,
    )
    result = walk_forward(frame, _params(fill="same_close"))
    assert result["scheme"] == "half"
    assert result["oos_equity"].index.min() == result["valid_start"]
    assert len(result["oos_equity"]) == len(result["oos_returns"])
    assert result["table"]["role"].tolist().count("valid_half") == 1


def test_selected_hedge_skips_the_other_beta():
    from ah_pairs.analysis import build_signals

    y, x = _cointegrated_pair(n=80, seed=1)
    prices = pd.DataFrame(
        {"log_a": y, "log_h": x, "both_open": True},
        index=pd.bdate_range("2020-01-02", periods=len(y)),
    )
    rolling = build_signals(prices, _params(hedge="rolling", hedge_window=20, z_window=10), alt=False)
    assert "beta_rolling" in rolling.columns
    assert "beta_kalman" not in rolling.columns
    kalman = build_signals(prices, _params(hedge="kalman", hedge_window=20, z_window=10), alt=False)
    assert "beta_kalman" in kalman.columns
    assert "beta_rolling" not in kalman.columns


def test_slice_codes_and_english_labels_have_no_chinese():
    import re

    from ah_pairs.i18n import set_lang
    from ah_pairs.labels import SLICE_EN, slice_label
    from ah_pairs.stability import slice_checks

    y, x = _cointegrated_pair(n=120, phi=0.8, seed=8)
    frame = pd.DataFrame(
        {
            "log_a": y,
            "log_h": x,
            "log_a_raw": y + 0.2,
            "log_h_raw": x,
            "ret_a": np.zeros(len(y)),
            "ret_h": np.zeros(len(y)),
        }
    )
    checks = slice_checks(frame)
    assert "unadjusted" in set(checks["code"])
    assert "full" in set(checks["code"])
    set_lang("en")
    try:
        for code in SLICE_EN:
            assert re.search(r"[\u4e00-\u9fff]", slice_label(code)) is None
    finally:
        set_lang("zh")


def test_download_notes_keep_codes_the_page_can_translate():
    from ah_pairs.notes import parse_notes

    parsed = parse_notes(
        [
            "meta:src:a:yahoo",
            "meta:src:h:eastmoney",
            "meta:sessions:10:2",
            "meta:fx_lag:3",
            "meta:adjust_mismatch:1",
            "A股来源：cache",
            "含预热的下载区间：同时开市 10 天，只有一边开市 2 天。",
        ]
    )
    assert parsed["sources"]["a"] == "yahoo"
    assert parsed["sources"]["h"] == "eastmoney"
    assert parsed["meta"]["download_common"] == 10
    assert parsed["meta"]["download_one_sided"] == 2
    assert parsed["meta"]["fx_lag"] == 3
    assert parsed["meta"]["adjust_mismatch"] is True
    assert parsed["prose"]


def test_market_refresh_keeps_a_stale_cache_and_does_not_fetch_on_read(tmp_path):
    from ah_pairs.market import read_ah_premium_index, read_southbound, refresh_or_keep

    assert read_southbound(tmp_path) is None
    assert read_ah_premium_index(tmp_path) is None
    cached = pd.Series([120.0], index=pd.to_datetime(["2024-01-02"]), name="hsahp")

    def boom():
        raise RuntimeError("upstream down")

    kept, note = refresh_or_keep(cached, boom, lambda series: series)
    assert kept.iloc[0] == 120.0
    assert note.startswith("沿用")
    fresh = pd.Series([130.0], index=cached.index, name="hsahp")
    stored, source = refresh_or_keep(cached, lambda: (fresh, "yahoo:HSAHP.HK"), lambda series: series)
    assert source == "yahoo:HSAHP.HK"
    assert stored.iloc[0] == 130.0


def test_read_scan_prefers_the_newer_folder(tmp_path, monkeypatch):
    import os

    from ah_pairs.scan import read_scan, write_scan
    from ah_pairs import scan as scan_mod

    older = tmp_path / "old"
    newer = tmp_path / "new"
    write_scan(pd.DataFrame({"A": ["1"]}), pd.DataFrame({"x": [1.0]}), older)
    write_scan(pd.DataFrame({"A": ["2"]}), pd.DataFrame({"x": [2.0]}), newer)
    os.utime(older / "summary.parquet", (1_000, 1_000))
    os.utime(newer / "summary.parquet", (2_000_000_000, 2_000_000_000))
    monkeypatch.setattr(scan_mod, "_scan_folders", lambda folder=None: [older, newer])
    summary, _returns = read_scan()
    assert summary.iloc[0]["A"] == "2"


def test_spec_link_roundtrip_defaults_short_off_and_rejects_bad_hedge():
    from datetime import date

    from ah_pairs.links import encode_spec, parse_spec_query

    today = date(2026, 11, 5)
    raw = {
        "pair": "601318",
        "lang": "zh",
        "start": "2018-01-01",
        "end": "2026-11-02",
        "hedge": "kalman",
        "hw": "80",
        "zw": "40",
        "zin": "2.5",
        "zout": "0.5",
        "fill": "same_close",
        "short": "1",
        "cost": "zero",
    }
    spec = parse_spec_query(raw, today=today)
    assert spec.rejected == ()
    assert spec.hedge == "kalman"
    assert spec.hedge_window == 80
    assert spec.z_window == 40
    assert spec.entry_z == 2.5
    assert spec.exit_z == 0.5
    assert spec.fill == "same_close"
    assert spec.allow_short_a is True
    assert spec.include_costs is False
    encoded = encode_spec(spec, pair="601318", lang="zh")
    assert set(encoded) == {
        "pair",
        "lang",
        "start",
        "end",
        "hedge",
        "hw",
        "zw",
        "zin",
        "zout",
        "fill",
        "short",
        "cost",
    }
    assert encoded["short"] == "1"
    assert encoded["cost"] == "zero"
    assert "beta" not in encoded
    again = parse_spec_query(encoded, today=today)
    assert again.rejected == ()
    assert again == spec

    missing = parse_spec_query({}, today=today)
    assert missing.allow_short_a is False
    assert missing.include_costs is True
    assert missing.hedge == "rolling"
    assert missing.entry_z == 2.0
    assert missing.exit_z == 0.5
    assert missing.fill == "next_close"
    assert missing.start == date(2018, 1, 1)
    assert missing.end == today
    assert missing.rejected == ()

    bad = parse_spec_query({"hedge": "foo", "short": "yes", "hw": "10", "zin": "2", "zout": "0.5"}, today=today)
    assert bad.hedge == "rolling"
    assert bad.allow_short_a is False
    assert bad.hedge_window == 120
    assert bad.entry_z == 2.0
    assert bad.exit_z == 0.5
    assert bad.rejected == ("hedge", "hw", "short")

    flipped = parse_spec_query({"start": "2026-12-01", "end": "2018-01-01"}, today=today)
    assert flipped.start == date(2018, 1, 1)
    assert flipped.end == today
    assert flipped.rejected == ("start", "end")


def test_h_dividend_tax_is_zero_when_flat_and_does_not_change_the_main_backtest():
    import re

    from ah_pairs.backtest import run_backtest
    from ah_pairs.dividends import h_dividend_return_drag, h_dividend_tax_cny, h_dividend_tax_row
    from ah_pairs.i18n import set_lang
    from ah_pairs.labels import cost_case_label

    idx = pd.to_datetime(["2024-06-03", "2024-06-04"])
    dividend = pd.Series([1.5, 2.0], index=idx)
    fx = pd.Series([0.90, 0.80], index=idx)
    long_h = pd.Series([True, False], index=idx)
    tax = h_dividend_tax_cny(dividend, fx, long_h)
    assert tax.iloc[0] == pytest.approx(1.5 * 0.2 * 0.90)
    assert tax.iloc[1] == 0.0

    held = pd.DataFrame(
        {
            "w_h": [0.0, 0.4, 0.4],
            "w_a": [0.0, -0.6, -0.6],
            "h_hkd": [10.0, 10.0, 10.0],
            "ret_strat": [0.0, 0.0, 0.0],
            "cost": [0.0, 0.0, 0.0],
            "borrow": [0.0, 0.0, 0.0],
        },
        index=pd.to_datetime(["2024-06-03", "2024-06-04", "2024-06-05"]),
    )
    cash = pd.Series([2.0], index=pd.to_datetime(["2024-06-05"]))
    drag = h_dividend_return_drag(held, cash)
    assert drag.iloc[0] == 0.0
    assert drag.iloc[1] == 0.0
    assert drag.iloc[2] == pytest.approx(0.2 * 2.0 / 10.0 * 0.4)

    sample = pd.bdate_range("2020-01-02", periods=20)
    prices = pd.DataFrame(
        {
            "z": np.concatenate([np.zeros(5), np.full(10, -3.0), np.zeros(5)]),
            "beta": np.ones(20),
            "ret_a": np.full(20, 0.001),
            "ret_h": np.zeros(20),
            "h_hkd": np.full(20, 10.0),
        },
        index=sample,
    )
    worked, trades, stats = run_backtest(prices, _params(fill="same_close"))
    before = float(stats["total_return"])
    original = worked["ret_strat"].copy()
    missing = h_dividend_tax_row(worked, None, trades)
    assert missing["status"].iloc[0] == "missing"
    assert missing["code"].iloc[0] == "h_div_tax"
    assert worked["ret_strat"].equals(original)
    assert float(run_backtest(prices, _params(fill="same_close"))[2]["total_return"]) == before
    assert (worked["w_h"] <= 0).all()
    taxed = h_dividend_tax_row(worked, pd.Series([1.0], index=sample[[8]]), trades)
    assert taxed["status"].iloc[0] == "ok"
    assert taxed["净收益"].iloc[0] == pytest.approx(before)
    set_lang("en")
    try:
        label = cost_case_label("h_div_tax")
        assert label == "H dividend tax 20%"
        assert re.search(r"[\u4e00-\u9fff]", label) is None
    finally:
        set_lang("zh")


def test_read_h_dividends_stays_on_disk(tmp_path):
    from ah_pairs.data import read_h_dividends

    assert read_h_dividends("02318", tmp_path) is None


def test_scan_snapshot_records_asof_and_spec(tmp_path):
    from ah_pairs.scan import read_scan, scan_is_older, with_scan_meta, write_scan

    summary = with_scan_meta(
        pd.DataFrame({"A": ["601318"], "公司": ["平安"]}),
        "2018-01-01",
        "2026-11-02",
        asof="2026-11-01",
    )
    write_scan(summary, pd.DataFrame({"x": [1.0]}), tmp_path)
    got, _returns = read_scan(tmp_path)
    assert got["asof"].iloc[0] == "2026-11-01"
    assert got["spec"].iloc[0] == "short=0;cost=statutory;fill=next_close"
    assert got["start"].iloc[0] == "2018-01-01"
    assert scan_is_older("2026-11-01", "2026-11-05 22:55")
    assert not scan_is_older("2026-11-05", "2026-11-02 22:55")
    assert not scan_is_older("2026-11-05", "无本地缓存")

