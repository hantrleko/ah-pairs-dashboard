"""全池看板状态分类。"""

from datetime import date

import pandas as pd

from ah_pairs.board import board_state, classify

REF = date(2026, 11, 26)


def test_unavailable_when_row_lags_snapshot():
    assert board_state("2026-11-24", 0, False, "", 0.5, 2.0, REF) == "unavailable"


def test_pending_beats_holding():
    assert board_state("2026-11-26", 1, True, "", 2.3, 2.0, REF) == "pending"


def test_holding_and_blocked():
    assert board_state("2026-11-26", 1, False, "hold", 0.3, 2.0, REF) == "holding"
    assert board_state("2026-11-26", 1, False, "beta_blocked", 0.3, 2.0, REF) == "blocked"


def test_near_entry_only_when_flat_and_close():
    assert board_state("2026-11-26", 0, False, "", -1.7, 2.0, REF) == "near_entry"
    assert board_state("2026-11-26", 0, False, "", 1.4, 2.0, REF) == "calm"
    assert board_state("2026-11-26", 0, False, "", 0.2, 2.0, REF) == "calm"


def test_priority_unavailable_first():
    assert board_state("2026-11-20", 1, True, "beta_blocked", 2.5, 2.0, REF) == "unavailable"


def test_classify_aligns_with_table():
    table = pd.DataFrame(
        {
            "asof": ["2026-11-26", "2026-11-24", "2026-11-26"],
            "position": [0, 1, 0],
            "pending": [True, False, False],
            "reason": ["enter_long_spread", "hold", ""],
            "z": [2.3, 0.5, -1.8],
        }
    )
    states = classify(table, REF, 2.0)
    assert list(states) == ["pending", "unavailable", "near_entry"]


def test_classify_without_reason_column():
    table = pd.DataFrame(
        {
            "asof": ["2026-11-26"],
            "position": [1],
            "pending": [False],
            "z": [0.5],
        }
    )
    states = classify(table, REF, 2.0)
    assert list(states) == ["holding"]
