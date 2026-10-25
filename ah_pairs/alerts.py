"""|z| 越过开仓阈值时，向 ALERT_WEBHOOK_URL 发一条 JSON。

地址只从环境变量读取，不写进仓库。没有地址时只把报文打印出来，退出码仍是 0。
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from ah_pairs.analysis import analyze_frame, warmup_start
from ah_pairs.backtest import StrategyParams, current_state
from ah_pairs.costs import CostParams
from ah_pairs.data import load_pair
from ah_pairs.universe import load_universe


def alert_payload(name: str, a_code: str, h_code: str, state: dict, entry_z: float) -> dict | None:
    z_value = state.get("z")
    if z_value is None or not np.isfinite(z_value) or abs(float(z_value)) < float(entry_z):
        return None
    return {
        "source": "ah-pairs-dashboard",
        "name": name,
        "a_code": a_code,
        "h_code": h_code,
        "asof": None if state.get("asof") is None else str(pd.Timestamp(state["asof"]).date()),
        "z": float(z_value),
        "entry_z": float(entry_z),
        "position": int(state.get("position") or 0),
        "signal": int(state.get("signal") or 0),
        "pending": bool(state.get("pending")),
        "note": "A/H 不能兑换。这是研究提醒，不是下单指令。p 值不是获利概率。",
    }


def send_webhook(url: str, payload: dict, timeout: float = 15) -> tuple[bool, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "ah-pairs-dashboard"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)


def scan_alerts(codes: list[str], start: str, end: str, entry_z: float) -> list[dict]:
    universe = load_universe()
    params = StrategyParams(allow_short_a=False, entry_z=entry_z, costs=CostParams())
    found = []
    for code in codes:
        row = universe.loc[universe["a_code"] == str(code).zfill(6)]
        if row.empty:
            continue
        item = row.iloc[0]
        prices, _notes = load_pair(item.a_code, item.h_code, warmup_start(start, params), end)
        result = analyze_frame(
            prices,
            params,
            eval_start=pd.Timestamp(start),
            full_diagnostics=False,
        )
        state = current_state(result["frame"], result["trades"], params.fill)
        payload = alert_payload(item["name"], item.a_code, item.h_code, state, entry_z)
        if payload is not None:
            found.append(payload)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="|z| 越过阈值时发送 webhook")
    parser.add_argument("--a", action="append", default=["601318", "601398", "600036", "601857"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--entry", type=float, default=2.0)
    args = parser.parse_args()
    try:
        found = scan_alerts(args.a, args.start, args.end, args.entry)
    except Exception as exc:
        print(f"提醒没有算出：{exc}")
        raise SystemExit(0)
    if not found:
        print("没有标的的 |z| 越过开仓阈值。")
        return
    url = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
    print(json.dumps(found, ensure_ascii=False, indent=2))
    if not url:
        print("未设置 ALERT_WEBHOOK_URL，只打印报文，不发送。")
        return
    ok, note = send_webhook(url, {"alerts": found})
    print(note)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
