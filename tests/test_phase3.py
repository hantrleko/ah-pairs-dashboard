"""三阶段：今日全景、提醒订阅、行情日期。不访问网络。"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ah_pairs.backtest import StrategyParams
from ah_pairs.data import _write_series
from ah_pairs.desk import default_desk_params
from ah_pairs.eflab_calendar import last_completed_session, yahoo_exclusive_end
from ah_pairs.panorama import build_panorama, is_default_spec, load_panorama, write_panorama
from ah_pairs.subscriptions import (
    add_subscription,
    deliver,
    load_subscriptions,
    mask_target,
    remove_subscription,
    subscriptions_path,
)
from ah_pairs.universe import a_to_yahoo, h_to_yahoo


def test_last_completed_session_waits_for_the_hong_kong_close():
    assert last_completed_session(datetime(2026, 11, 13, 16, 9)) == date(2026, 11, 12)
    assert last_completed_session(datetime(2026, 11, 13, 16, 10)) == date(2026, 11, 13)
    assert last_completed_session(datetime(2026, 11, 14, 12, 0)) == date(2026, 11, 13)
    assert last_completed_session(datetime(2026, 11, 16, 9, 0)) == date(2026, 11, 13)
    assert last_completed_session(datetime(2026, 11, 13, 8, 10, tzinfo=timezone.utc)) == date(2026, 11, 13)
    assert last_completed_session(datetime(2026, 11, 13, 8, 9, tzinfo=timezone.utc)) == date(2026, 11, 12)
    assert yahoo_exclusive_end(datetime(2026, 11, 13, 16, 10)) == date(2026, 11, 14)
    with pytest.raises(TypeError):
        last_completed_session("2026-11-13")


def test_sibling_calendars_match_when_those_repos_are_present():
    here = Path(__file__).resolve().parents[1] / "ah_pairs" / "eflab_calendar.py"
    github = here.parents[2]
    for relative in (
        Path("omnifinance") / "core" / "eflab_calendar.py",
        Path("financial-analysis-agent") / "src" / "eflab_calendar.py",
    ):
        other = github / relative
        if other.exists():
            assert other.read_bytes() == here.read_bytes()


def test_panorama_uses_only_the_given_directory_and_does_not_download(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("download")

    monkeypatch.setattr("ah_pairs.data.get_close", boom)
    dates = pd.bdate_range("2020-01-01", periods=160)
    step = np.arange(160)
    h = 100 + 8 * np.sin(step / 2.0)
    a = h.copy()
    a[-3:] = h[-3:] * 0.5
    _write_series(tmp_path / "HKDCNY_X.parquet", pd.Series(np.full(160, 0.9), index=dates))
    _write_series(tmp_path / f"{a_to_yahoo('601318')}.parquet", pd.Series(a, index=dates))
    _write_series(tmp_path / f"{h_to_yahoo('02318')}.parquet", pd.Series(h, index=dates))
    universe = pd.DataFrame(
        [
            {"name": "平安", "a_code": "601318", "h_code": "02318"},
            {"name": "缺文件", "a_code": "601398", "h_code": "01398"},
        ]
    )
    params = StrategyParams(hedge_window=20, z_window=10, entry_z=0.8, exit_z=0.2, stop_z=0.0)
    table, meta = build_panorama(universe, dates[40], dates[-1], params, cache_dir=tmp_path)
    assert meta["skipped"] == ["601398"]
    assert list(table["a_code"]) == ["601318"]
    assert table.iloc[0]["premium"] < -0.2
    assert table.iloc[0]["z"] < -0.8
    assert int(table.iloc[0]["position"]) == 1


def test_snapshot_hides_an_earlier_end_and_a_different_spec(tmp_path, monkeypatch):
    published = tmp_path / "published"
    default_cache = tmp_path / "default"
    monkeypatch.setattr("ah_pairs.panorama.PUBLISHED_DIR", published)
    monkeypatch.setattr("ah_pairs.panorama.CACHE_DIR", default_cache)
    table = pd.DataFrame(
        [
            {
                "a_code": "601318",
                "h_code": "02318",
                "name": "中国平安",
                "asof": "2026-11-05",
                "premium": 0.2,
                "z": -0.4,
                "position": 0,
                "signal": 0,
                "pending": False,
            }
        ]
    )
    meta = {
        "start": "2018-01-01",
        "sample_end": "2026-11-05",
        "params": json.loads(json.dumps(asdict(default_desk_params()), sort_keys=True)),
        "skipped": [],
    }
    assert write_panorama(table, meta) == default_cache / "panorama.parquet"
    assert (published / "panorama.parquet").exists()
    assert load_panorama(_page(date(2026, 11, 10)))[2] == "ok"
    assert load_panorama(_page(date(2026, 11, 4)))[2] == "earlier_end"
    assert load_panorama(_page(date(2026, 11, 20)))[2] == "window"
    assert load_panorama(_page(date(2026, 11, 10), entry_z=2.5))[2] == "spec"
    assert load_panorama(_page(date(2026, 11, 10)), cache_dir=tmp_path / "other")[2] == "missing"
    assert is_default_spec(default_desk_params(), date(2018, 1, 1))
    assert not is_default_spec(replace(default_desk_params(), entry_z=2.5), date(2018, 1, 1))


def test_email_is_saved_but_not_sent_without_smtp(tmp_path, monkeypatch):
    monkeypatch.delenv("ALERT_SMTP_HOST", raising=False)
    monkeypatch.delenv("ALERT_SMTP_FROM", raising=False)
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("AH_SUBSCRIPTIONS_JSON", "{")
    called = {"smtp": 0, "hook": 0}

    def smtp_send(*_args):
        called["smtp"] += 1

    def webhook_post(*_args):
        called["hook"] += 1
        return True, "ok"

    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("network")))
    add_subscription("email", "Desk@example.com", ["601318"], 2, cache_dir=tmp_path)
    with pytest.raises(ValueError):
        add_subscription("email", "not-an-email", ["601318"], 2, cache_dir=tmp_path)
    rows = deliver([_alert()], cache_dir=tmp_path, webhook_post=webhook_post, smtp_send=smtp_send)
    assert rows[0]["status"] == "not_configured"
    assert rows[0]["target"] == "d***@example.com"
    assert "Desk@example.com" not in json.dumps(rows)
    assert called == {"smtp": 0, "hook": 0}
    assert "desk@example.com" in subscriptions_path(tmp_path).read_text(encoding="utf-8")
    assert load_subscriptions(tmp_path)[0]["stored"] == "file"


def test_webhook_is_posted_once_and_mail_uses_the_injected_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/secret-token")
    monkeypatch.setenv("ALERT_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("ALERT_SMTP_FROM", "from@example.com")
    monkeypatch.setenv("ALERT_SMTP_PORT", "")
    monkeypatch.delenv("AH_SUBSCRIPTIONS_JSON", raising=False)
    import smtplib

    monkeypatch.setattr(smtplib, "SMTP", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("socket")))
    monkeypatch.setattr(smtplib, "SMTP_SSL", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("socket")))
    posts = []
    sent = []
    add_subscription("webhook", "https://hooks.example.com/secret-token", ["601318"], 2, cache_dir=tmp_path)
    add_subscription("email", "desk@example.com", ["601318"], 2, cache_dir=tmp_path)
    add_subscription("email", "quiet@example.com", ["601318"], 9, cache_dir=tmp_path)
    rows = deliver(
        [_alert(z=2.4)],
        cache_dir=tmp_path,
        env_entry=2.0,
        webhook_post=lambda url, _body: posts.append(url) or (True, "ok"),
        smtp_send=lambda *args: sent.append(args),
    )
    assert posts == ["https://hooks.example.com/secret-token"]
    assert any(row["status"] == "skipped_duplicate" for row in rows)
    assert sent and sent[0][1] == 587 and sent[0][5] == "desk@example.com"
    assert any(row["status"] == "quiet" and row["target"] == "q***@example.com" for row in rows)
    assert "secret-token" not in json.dumps(rows)
    assert mask_target("webhook", "https://hooks.example.com/secret-token") == "https://hooks.example.com"
    monkeypatch.setenv(
        "AH_SUBSCRIPTIONS_JSON",
        json.dumps({"items": [{"kind": "webhook", "target": "https://other.example.com/x", "codes": ["601318"], "entry_z": 2}]}),
    )
    loaded = load_subscriptions(tmp_path)
    assert any(item["stored"] == "env" for item in loaded)
    record = next(item for item in loaded if item["stored"] == "file" and item["kind"] == "email")
    assert remove_subscription(record["id"], cache_dir=tmp_path)


def test_gitignore_keeps_subscription_addresses_out_of_git():
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "data/cache/" in text
    assert "data/subscriptions.json" in text


def _page(end: date, **changes):
    params = default_desk_params()
    if changes:
        params = replace(params, **changes)
    return SimpleNamespace(start_day=date(2018, 1, 1), end_day=end, params=params)


def _alert(z: float = 2.5) -> dict:
    return {
        "name": "平安",
        "a_code": "601318",
        "h_code": "02318",
        "asof": "2026-11-10",
        "z": z,
        "position": 0,
        "signal": 1,
        "pending": True,
    }
