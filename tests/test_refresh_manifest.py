"""刷新清单、行情缓存一致性和共享汇率。只用合成序列，不访问网络。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ah_pairs.manifest import AUX, CORE, FAILED, STALE, UPDATED, core_failed, make_item, overall_status, write_manifest


def test_manifest_roundtrip_and_partial_status(tmp_path):
    items = [
        make_item("HKDCNY=X", kind=CORE, status=UPDATED, data_end=pd.Timestamp("2024-01-05"), rows=100, source="yahoo"),
        make_item("hsahp.parquet", kind=AUX, status=FAILED, data_end=pd.Timestamp("2024-01-02"), detail="upstream down"),
    ]
    payload = write_manifest(items, tmp_path / "refresh_manifest.json")
    assert payload["status"] == "partial"
    loaded = json.loads((tmp_path / "refresh_manifest.json").read_text(encoding="utf-8"))
    assert loaded["items"][1]["data_end"] == "2024-01-02"
    assert loaded["items"][1]["detail"] == "upstream down"
    assert "generated_at_utc" in loaded
    assert "generated_at_beijing" in loaded
    assert not core_failed(items)


def test_core_failure_marks_the_run_failed():
    items = [make_item("prices 601318", kind=CORE, status=FAILED, detail="no data")]
    assert overall_status(items) == "failed"
    assert core_failed(items)
    ok = [make_item("prices 601318", kind=CORE, status=STALE, detail="沿用缓存")]
    assert overall_status(ok) == "partial"
    assert not core_failed(ok)


def test_refresh_prefers_the_consistent_cache_over_eastmoney(tmp_path, monkeypatch):
    from ah_pairs import data as data_mod

    index = pd.bdate_range("2024-01-02", periods=30)
    cached = pd.Series(np.linspace(10.0, 11.0, len(index)), index=index, name="close")
    data_mod._write_series(data_mod._cache_file(tmp_path, "601318.SS"), cached)

    def yahoo_down(ticker):
        raise RuntimeError("yahoo down")

    calls = {"eastmoney": 0}

    def eastmoney(kind, code):
        calls["eastmoney"] += 1
        return pd.Series(99.0, index=index, name="close")

    monkeypatch.setattr(data_mod, "_yahoo_close", yahoo_down)
    monkeypatch.setattr(data_mod, "_eastmoney_close", eastmoney)
    out, source = data_mod.get_close("601318.SS", tmp_path, kind="a", code="601318", refresh=True)
    # 有缓存时沿用同一条口径，不用东财回退把已发布结果在两种复权之间来回搬。
    assert calls["eastmoney"] == 0
    assert "沿用" in source
    assert float(out.iloc[-1]) == pytest.approx(float(cached.iloc[-1]))


def test_eastmoney_is_only_the_cold_cache_fallback(tmp_path, monkeypatch):
    from ah_pairs import data as data_mod

    index = pd.bdate_range("2024-01-02", periods=30)

    def yahoo_down(ticker):
        raise RuntimeError("yahoo down")

    def eastmoney(kind, code):
        return pd.Series(np.linspace(20.0, 21.0, len(index)), index=index, name="close")

    monkeypatch.setattr(data_mod, "_yahoo_close", yahoo_down)
    monkeypatch.setattr(data_mod, "_eastmoney_close", eastmoney)
    out, source = data_mod.get_close("601318.SS", tmp_path, kind="a", code="601318", refresh=True)
    assert source == "eastmoney"
    assert float(out.iloc[-1]) == pytest.approx(21.0)


def test_load_pair_uses_a_shared_fx_series(tmp_path, monkeypatch):
    from ah_pairs import data as data_mod

    index = pd.bdate_range("2024-01-02", periods=40)

    def fake_close(ticker, cache_dir, kind="", code=None, refresh=False, list_date=None):
        return pd.Series(10.0, index=index, name="close"), "yahoo"

    def fx_must_not_run(cache_dir=data_mod.CACHE_DIR, refresh=False):
        raise AssertionError("传入了共享汇率，get_fx 不该再下载")

    monkeypatch.setattr(data_mod, "get_close", fake_close)
    monkeypatch.setattr(data_mod, "get_fx", fx_must_not_run)
    fx = pd.Series(0.9, index=index, name="close")
    frame, notes = data_mod.load_pair("601318", "2318", index[0], index[-1], cache_dir=tmp_path, fx=fx)
    assert not frame.empty
    assert "汇率来源：shared" in notes
