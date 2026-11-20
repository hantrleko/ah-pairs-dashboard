from __future__ import annotations

from datetime import date

import pandas as pd

from ah_pairs.price_quality import MAX_ABS_DAILY_MOVE, sanitize_close_series, scan_large_moves


def test_sanitize_trims_before_h_listing_and_removes_jump():
    idx = pd.bdate_range("2024-07-01", periods=6)
    series = pd.Series([2.23, 2.24, 2.25, 45.0, 45.5, 46.0], index=idx)
    clean, notes = sanitize_close_series(series, list_date=date(2024, 9, 17))
    assert clean.empty
    assert any("上市日" in item for item in notes)

    idx2 = pd.bdate_range("2024-09-17", periods=6)
    series2 = pd.Series([45.0, 45.5, 200.0, 46.0, 46.5, 47.0], index=idx2)
    clean2, notes2 = sanitize_close_series(series2, list_date=date(2024, 9, 17))
    assert len(clean2) < len(series2)
    assert not scan_large_moves(clean2, MAX_ABS_DAILY_MOVE)
    assert any("单日涨跌" in item for item in notes2)