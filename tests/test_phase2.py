"""二阶段：相对溢价、结构断点、纯多头、月度截面、融券名单。不访问网络。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ah_pairs.backtest import StrategyParams, run_backtest
from ah_pairs.breaks import gregory_hansen
from ah_pairs.borrow import _as_ratio, borrow_token_for, mask_from_borrow_token, read_borrow
from ah_pairs.costs import CostParams
from ah_pairs.cross_section import load_local_panels, monthly_cheap_book
from ah_pairs.data import _write_series, assemble_pair
from ah_pairs.holding import _end_weights, holding_comparison
from ah_pairs.relative import index_to_premium, relative_premium
from ah_pairs.universe import a_to_yahoo, h_to_yahoo


def _dates(n: int, start: str = "2018-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def test_index_level_becomes_a_premium_and_a_bad_median_is_refused():
    level = pd.Series([120.0, 110.0], index=_dates(2))
    premium = index_to_premium(level)
    assert premium.iloc[0] == pytest.approx(0.20)
    already = index_to_premium(pd.Series([0.20, 0.15], index=_dates(2)))
    assert already.iloc[0] == 0.20
    missing = relative_premium(premium, None)
    assert missing["ok"] is False
    bad = relative_premium(premium, pd.Series([5.0, 5.0], index=_dates(2)))
    assert bad["ok"] is False
    assert "中位数" in bad["message"]


def test_relative_premium_is_the_difference_and_a_stationary_gap_can_reject():
    idx = _dates(500)
    rng = np.random.default_rng(0)
    shock = rng.normal(0, 1, len(idx))
    gap = np.zeros(len(idx))
    for t in range(1, len(idx)):
        gap[t] = 0.2 * gap[t - 1] + shock[t]
    stock = pd.Series(gap + 0.20, index=idx)
    market = pd.Series(120.0, index=idx)
    result = relative_premium(stock, market)
    assert result["ok"] is True
    assert abs(result["latest"] - gap[-1]) < 1e-8
    assert result["adf_pvalue"] < 0.05
    assert result["reject_5"] is True


def test_gregory_hansen_uses_the_1996_critical_value_and_finds_a_level_shift():
    short = gregory_hansen(np.arange(40, dtype=float), np.arange(40, dtype=float))
    assert short["ok"] is False
    rng = np.random.default_rng(1)
    n = 220
    x = np.cumsum(rng.normal(0, 1, n))
    shock = rng.normal(0, 0.05, n)
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.3 * noise[t - 1] + shock[t]
    true_break = 140
    dummy = np.zeros(n)
    dummy[true_break:] = 1.0
    y = 0.5 + 1.1 * x + 3.0 * dummy + noise
    idx = _dates(n, "2016-01-04")
    result = gregory_hansen(pd.Series(y, index=idx), pd.Series(x, index=idx))
    assert result["ok"] is True
    model_c = next(row for row in result["models"] if row["model"] == "C")
    assert model_c["crit_5"] == -4.61
    found = int(idx.get_loc(pd.Timestamp(model_c["break_date"])))
    assert abs(found - true_break) <= 15


def _holding_frame(z: float) -> pd.DataFrame:
    idx = _dates(30, "2024-01-02")
    return pd.DataFrame(
        {
            "ret_a": np.full(len(idx), 0.001),
            "ret_h": np.full(len(idx), -0.002),
            "z": np.full(len(idx), z),
            "both_open": True,
        },
        index=idx,
    )


def test_long_only_switch_matches_the_cheap_leg_and_never_shorts():
    costs = CostParams()
    cheap = holding_comparison(_holding_frame(-1.0), costs).set_index("code")
    rich = holding_comparison(_holding_frame(1.0), costs).set_index("code")
    assert cheap.loc["switch", "total_return"] == cheap.loc["always_a", "total_return"]
    assert rich.loc["switch", "total_return"] == rich.loc["always_h", "total_return"]
    assert cheap.loc["always_a", "total_cost"] > 0
    want = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    end_a, end_h = _end_weights(want, np.ones(len(want), dtype=bool))
    assert (end_a >= -1e-12).all()
    assert (end_h >= -1e-12).all()
    assert np.allclose(end_a, [0, 1, 0, 1, 0])
    assert np.allclose(end_h, [0, 0, 1, 0, 1])


def _panel(code: str, premium: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    fx = pd.Series(0.9, index=idx)
    h = pd.Series(10.0, index=idx)
    a = pd.Series(9.0 * (1.0 + premium), index=idx)
    prices, _notes = assemble_pair(a, h, fx, idx.min(), idx.max())
    prices["premium"] = premium
    return prices


def test_monthly_book_longs_the_cheap_quartile_from_disk_only(tmp_path):
    idx = _dates(90, "2020-01-02")
    base = np.full(len(idx), 0.20)
    panels = {}
    for i, code in enumerate([f"{600000 + i:06d}" for i in range(8)]):
        premium = base.copy()
        premium[-8:] = 0.02 if i == 0 else 0.40
        panels[code] = _panel(code, premium, idx)
    result = monthly_cheap_book(panels, z_window=20, costs=CostParams(), min_names=4)
    assert result["ok"] is True
    assert "600000" in result["latest_book"]
    assert len(result["latest_book"]) == len(panels) // 4
    assert set(result["latest_book"]).issubset(set(panels))
    empty, skipped = load_local_panels(
        pd.DataFrame({"a_code": ["601398"], "h_code": ["01398"]}),
        "2020-01-01",
        "2020-06-30",
        cache_dir=tmp_path,
    )
    assert empty == {}
    assert skipped == ["FX"]


def test_one_sided_days_do_not_wipe_the_premium_z():
    idx = _dates(80, "2020-01-02")
    premium = pd.Series(np.linspace(0.1, 0.4, len(idx)), index=idx)
    premium.iloc[::10] = np.nan
    panels = {
        f"{600000 + i:06d}": pd.DataFrame(
            {
                "premium": premium + i * 0.01,
                "ret_a": 0.0,
                "ret_h": 0.0,
                "both_open": premium.notna(),
            },
            index=idx,
        )
        for i in range(8)
    }
    panels["600007"].loc[idx, "premium"] = np.linspace(0.4, -0.2, len(idx))
    result = monthly_cheap_book(panels, z_window=20, costs=CostParams(), min_names=4)
    assert result["ok"] is True
    assert result["n_months"] >= 2
    assert "600007" in result["latest_book"]


def test_month_end_uses_the_last_day_with_enough_names():
    idx = _dates(45, "2020-01-02")
    extra = idx.append(pd.DatetimeIndex([idx[-1] + pd.offsets.BDay(1)]))
    panels = {}
    for i in range(8):
        code = f"{600000 + i:06d}"
        days = extra if i == 0 else idx
        premium = np.linspace(0.30, 0.05 if i == 7 else 0.40, len(days))
        panels[code] = pd.DataFrame(
            {
                "premium": premium,
                "ret_a": np.zeros(len(days)),
                "ret_h": np.zeros(len(days)),
                "both_open": True,
            },
            index=days,
        )
    result = monthly_cheap_book(panels, z_window=10, costs=CostParams(), min_names=4)
    assert result["ok"] is True
    assert pd.Timestamp(result["latest_signal"]) == idx[-1]
    assert "600007" in result["latest_book"]


def test_local_panels_skip_names_without_files(tmp_path):
    idx = _dates(40, "2020-01-02")
    fx = pd.Series(0.9, index=idx)
    _write_series(tmp_path / "HKDCNY_X.parquet", fx)
    code = "601398"
    _write_series(tmp_path / f"{a_to_yahoo(code).replace('=', '_')}.parquet", pd.Series(10.0, index=idx))
    universe = pd.DataFrame({"a_code": [code, "600036"], "h_code": ["01398", "03968"]})
    panels, skipped = load_local_panels(universe, idx.min(), idx.max(), cache_dir=tmp_path)
    assert panels == {}
    assert code in skipped
    assert "600036" in skipped
    assert "FX" not in skipped
    _write_series(tmp_path / f"{h_to_yahoo('01398').replace('=', '_')}.parquet", pd.Series(20.0, index=idx))
    panels, skipped = load_local_panels(universe, idx.min(), idx.max(), cache_dir=tmp_path)
    assert code in panels
    assert "600036" in skipped


def _trade_frame(z: float) -> pd.DataFrame:
    idx = _dates(40, "2024-01-02")
    return pd.DataFrame(
        {
            "z": np.full(len(idx), z),
            "beta": np.ones(len(idx)),
            "ret_a": np.zeros(len(idx)),
            "ret_h": np.zeros(len(idx)),
            "both_open": True,
        },
        index=idx,
    )


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


def test_borrow_calendar_blocks_only_covered_days_and_default_backtest_is_unchanged(tmp_path, monkeypatch):
    from ah_pairs import borrow

    assert _as_ratio(80) == 0.8
    assert _as_ratio(0.8) == 0.8
    calendar = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")],
            "a_code": ["601398"],
            "eligible": [True],
            "margin_ratio": [0.8],
        }
    )
    calendar.to_parquet(tmp_path / "borrow.parquet", index=False)
    published = tmp_path / "published"
    published.mkdir()
    calendar.assign(a_code="000001").to_parquet(published / "borrow.parquet", index=False)
    monkeypatch.setattr(borrow, "PUBLISHED_DIR", published)
    monkeypatch.setattr(borrow, "CACHE_DIR", tmp_path / "default_cache")
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    assert read_borrow(tmp_path / "empty") is None
    assert borrow_token_for(idx, "601398", tmp_path) == "2024-01-02:1"
    blocked = borrow_token_for(idx, "600036", tmp_path)
    assert blocked == "2024-01-02:0"
    assert "2024-01-03" not in blocked
    mask = mask_from_borrow_token(idx, blocked)
    assert mask.tolist() == [False, True]
    assert borrow_token_for(idx, "601398", tmp_path / "empty") == ""

    frame = _trade_frame(3.0)
    params = StrategyParams(allow_short_a=True, costs=_zero_costs())
    short_book, _trades, short_stats = run_backtest(frame, params)
    same_book, _trades, same_stats = run_backtest(frame, params, short_a_days=None)
    blocked_book, _trades, _stats = run_backtest(frame, params, short_a_days=np.zeros(len(frame), dtype=bool))
    default_book, _trades, default_stats = run_backtest(frame, StrategyParams(costs=_zero_costs()))
    untouched, _trades, untouched_stats = run_backtest(frame, StrategyParams(costs=_zero_costs()), short_a_days=None)
    assert (short_book["w_a"] < 0).any()
    assert short_stats["total_return"] == same_stats["total_return"]
    assert np.allclose(blocked_book["w_a"], 0.0)
    assert np.allclose(blocked_book["w_h"], 0.0)
    assert default_stats["total_return"] == untouched_stats["total_return"]
    assert np.allclose(default_book["w_a"], 0.0)
    assert np.allclose(default_book["w_a"], untouched["w_a"].to_numpy(dtype=float))

