"""已发布副本和默认规格预计算。不访问网络。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from ah_pairs.desk import default_desk_params, load_stability, load_walk, write_result
from ah_pairs.paths import PUBLISHED_DIR
from ah_pairs.price_quality import MAX_ABS_DAILY_MOVE, scan_large_moves
from ah_pairs.universe import a_to_yahoo, h_to_yahoo, liquid_subset, load_universe
from ah_pairs.data import read_local_close


def test_default_desk_matches_the_sidebar_first_open():
    params = default_desk_params()
    assert params.hedge == "rolling"
    assert params.hedge_window == 120
    assert params.z_window == 60
    assert params.entry_z == 2.0
    assert params.exit_z == 0.5
    assert params.stop_z == 4.0
    assert params.fill == "next_close"
    assert params.allow_short_a is False
    assert params.kalman_delta_auto is True
    assert params.beta_band == 0.10
    assert params.require_positive_beta is True
    assert params.costs.use_statutory_stamp is True
    assert params.costs.a_commission == 0.00025


def test_published_dividend_is_read_only_for_the_default_cache(tmp_path, monkeypatch):
    from ah_pairs import data

    cache = tmp_path / "cache"
    published = tmp_path / "published"
    monkeypatch.setattr(data, "CACHE_DIR", cache)
    monkeypatch.setattr(data, "PUBLISHED_DIR", published)
    series = pd.Series([1.5], index=pd.to_datetime(["2024-06-03"]))
    data._write_series(data.dividend_path(published / "dividends", "2318.HK"), series)

    got = data.read_h_dividends("02318", cache)
    assert got is not None
    assert float(got.iloc[0]) == 1.5
    assert data.read_h_dividends("02318", tmp_path / "other") is None


def test_published_price_is_used_without_a_download(tmp_path, monkeypatch):
    from ah_pairs import data

    cache = tmp_path / "cache"
    published = tmp_path / "published"
    monkeypatch.setattr(data, "CACHE_DIR", cache)
    monkeypatch.setattr(data, "PUBLISHED_DIR", published)
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=3)
    data._write_series(
        data._cache_file(published / "prices", "601318.SS"),
        pd.Series([10.0, 10.1, 10.2], index=index),
    )

    def boom(ticker):
        raise AssertionError(ticker)

    monkeypatch.setattr(data, "_yahoo_close", boom)
    got, source = data.get_close("601318.SS", cache)
    assert source == "published"
    assert len(got) == 3

    def offline(ticker):
        raise RuntimeError(ticker)

    monkeypatch.setattr(data, "_yahoo_close", offline)
    with pytest.raises(RuntimeError):
        data.get_close("601318.SS", tmp_path / "other")


def test_newer_published_southbound_wins(tmp_path, monkeypatch):
    import os

    from ah_pairs import market

    cache = tmp_path / "cache"
    published = tmp_path / "published"
    cache.mkdir()
    published.mkdir()
    monkeypatch.setattr(market, "CACHE_DIR", cache)
    monkeypatch.setattr(market, "PUBLISHED_DIR", published)
    market._store(
        cache / "southbound.parquet",
        pd.Series([1.0], index=pd.to_datetime(["2020-01-02"])),
        "southbound",
    )
    market._store(
        published / "southbound.parquet",
        pd.Series([2.0], index=pd.to_datetime(["2024-01-02"])),
        "southbound",
    )
    os.utime(cache / "southbound.parquet", (1, 1))
    got = market.read_southbound()
    assert float(got.iloc[-1]) == 2.0
    assert market.read_southbound(tmp_path / "elsewhere") is None


def test_desk_snapshot_follows_the_default_spec(tmp_path):
    end = date(2026, 11, 1)
    grid = pd.DataFrame(
        {
            "start": [pd.Timestamp("2018-01-02")],
            "end": [pd.Timestamp("2019-01-02")],
            "length": [252],
            "nobs": [252],
            "pvalue": [0.2],
            "reject_5": [False],
        }
    )
    checks = pd.DataFrame({"code": ["full"], "pvalue": [0.2]})
    index = pd.to_datetime(["2021-01-04"])
    walk = {
        "ok": True,
        "scheme": "anchored-year",
        "chosen_entry": 2.0,
        "chosen_exit": 0.5,
        "chosen_entries": [2.0],
        "chosen_exits": [0.5],
        "n_folds": 1,
        "n_trials": 4,
        "valid_sharpe_median": 0.1,
        "valid_sharpe_min": 0.1,
        "valid_sharpe_max": 0.1,
        "train_end": pd.Timestamp("2020-12-31"),
        "valid_start": pd.Timestamp("2021-01-04"),
        "valid_end": pd.Timestamp("2021-12-31"),
        "table": pd.DataFrame({"role": ["valid"]}),
        "oos_returns": pd.Series([0.01], index=index),
        "oos_equity": pd.Series([1.01], index=index),
    }
    write_result(
        "601318",
        default_desk_params(),
        date(2018, 1, 1),
        end,
        pd.Timestamp("2026-11-01"),
        walk,
        grid,
        checks,
        desk_dir=tmp_path,
    )
    near = _page(date(2026, 11, 10))
    loaded = load_walk(near, desk_dir=tmp_path)
    assert loaded is not None and loaded["precomputed"] is True
    assert abs(float(loaded["oos_equity"].iloc[-1]) - 1.01) < 1e-9
    assert load_walk(_page(date(2026, 11, 12)), desk_dir=tmp_path) is None
    assert load_walk(_page(date(2026, 10, 20)), desk_dir=tmp_path) is None
    changed = _page(date(2026, 11, 10), entry_z=2.5)
    assert load_walk(changed, desk_dir=tmp_path) is None
    assert load_stability(changed, desk_dir=tmp_path) is not None
    assert load_stability(_page(date(2026, 11, 10), hedge="kalman"), desk_dir=tmp_path) is None
    write_result(
        "601398",
        default_desk_params(),
        date(2018, 1, 1),
        date(2026, 11, 10),
        pd.Timestamp("2026-11-01"),
        walk,
        grid,
        checks,
        desk_dir=tmp_path,
    )
    assert load_walk(_page(date(2026, 11, 6), a_code="601398"), desk_dir=tmp_path) is not None
    assert load_walk(_page(date(2026, 11, 10), a_code="601398"), desk_dir=tmp_path) is not None
    assert load_walk(_page(date(2026, 11, 12), a_code="601398"), desk_dir=tmp_path) is None
    assert load_walk(_page(date(2026, 10, 31), a_code="601398"), desk_dir=tmp_path) is None


@pytest.mark.skipif(not (PUBLISHED_DIR / "prices").is_dir(), reason="no published price directory")
def test_published_prices_have_no_absurd_single_day_moves():
    hits: list[tuple[str, object, float]] = []
    for row in liquid_subset(load_universe()).itertuples(index=False):
        for ticker in (a_to_yahoo(row.a_code), h_to_yahoo(row.h_code)):
            series = read_local_close(ticker)
            if series is None or len(series) < 2:
                continue
            for day, move in scan_large_moves(series, MAX_ABS_DAILY_MOVE):
                hits.append((ticker, day, move))
    assert hits == []


def test_refresh_with_everything_skipped_does_not_download():
    from ah_pairs.refresh import main

    assert main(["--skip-prices", "--skip-dividends", "--skip-market", "--skip-scan", "--skip-desk", "--skip-borrow", "--skip-panorama"]) == 0


def test_forward_write_is_opt_in_and_off_by_default(monkeypatch, tmp_path):
    """默认运行不产生正式账本记录：未传 --enable-forward-write 时连 update_batch 都不调用。"""
    from ah_pairs import refresh as refresh_mod

    calls = []

    def fake_forward(end, errors, items):
        calls.append(end)
        return False

    monkeypatch.setattr(refresh_mod, "_forward", fake_forward)
    rc = refresh_mod.main(["--skip-prices", "--skip-dividends", "--skip-market", "--skip-scan", "--skip-desk", "--skip-borrow", "--skip-panorama"])
    assert rc == 0
    assert calls == []
    rc = refresh_mod.main(["--skip-prices", "--skip-dividends", "--skip-market", "--skip-scan", "--skip-desk", "--skip-borrow", "--skip-panorama", "--enable-forward-write"])
    assert calls != []


def _page(end: date, a_code: str = "601318", **changes):
    params = default_desk_params()
    if changes:
        params = replace(params, **changes)
    return SimpleNamespace(
        picked=SimpleNamespace(a_code=a_code),
        start_day=date(2018, 1, 1),
        end_day=end,
        params=params,
        include_costs=True,
    )
