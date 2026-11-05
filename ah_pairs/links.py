"""把一套研究规格写成地址栏参数，并在打开时读回来。

链接只保存能复现主结果的那一组：股票和日期由调用方另写，这里负责对冲、窗口、
开平仓、成交、是否做空，以及成本是法定档还是零。佣金小数、卡尔曼 δ 和再平衡带不进链接。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import re
from typing import Any, Mapping

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_DEFAULT_START = date(2018, 1, 1)
_KEYS = ("start", "end", "hedge", "hw", "zw", "zin", "zout", "fill", "short", "cost")


@dataclass(frozen=True)
class SpecLink:
    start: date
    end: date
    hedge: str
    hedge_window: int
    z_window: int
    entry_z: float
    exit_z: float
    fill: str
    allow_short_a: bool
    include_costs: bool
    rejected: tuple[str, ...] = ()


def parse_spec_query(raw: Mapping[str, Any], *, today: date | None = None) -> SpecLink:
    """缺省字段用页面默认。出现了但写错的字段记入 rejected，不抛错。"""
    today = today or date.today()
    rejected: list[str] = []
    start = _DEFAULT_START
    end = today
    start_valid = True
    end_valid = True
    if _present(raw, "start"):
        parsed = _as_date(_text(raw.get("start")))
        if parsed is None:
            rejected.append("start")
            start_valid = False
        else:
            start = parsed
    if _present(raw, "end"):
        parsed = _as_date(_text(raw.get("end")))
        if parsed is None:
            rejected.append("end")
            end_valid = False
        else:
            end = parsed
    if start_valid and end_valid and start > end:
        if _present(raw, "start"):
            rejected.append("start")
        if _present(raw, "end"):
            rejected.append("end")
        start = _DEFAULT_START
        end = today

    hedge = "rolling"
    if _present(raw, "hedge"):
        value = _text(raw.get("hedge"))
        if value in {"rolling", "kalman"}:
            hedge = value
        else:
            rejected.append("hedge")

    hedge_window = 120
    if _present(raw, "hw"):
        value = _as_int(_text(raw.get("hw")))
        if value is not None and 40 <= value <= 250:
            hedge_window = value
        else:
            rejected.append("hw")

    z_window = 60
    if _present(raw, "zw"):
        value = _as_int(_text(raw.get("zw")))
        if value is not None and 20 <= value <= 120:
            z_window = value
        else:
            rejected.append("zw")

    entry_z = 2.0
    if _present(raw, "zin"):
        value = _as_tenth(_text(raw.get("zin")))
        if value is not None and 1.0 <= value <= 3.5:
            entry_z = value
        else:
            rejected.append("zin")

    exit_z = 0.5
    if _present(raw, "zout"):
        value = _as_tenth(_text(raw.get("zout")))
        if value is not None and 0.0 <= value <= 1.5 and value < entry_z:
            exit_z = value
        else:
            rejected.append("zout")

    fill = "next_close"
    if _present(raw, "fill"):
        value = _text(raw.get("fill"))
        if value in {"next_close", "same_close"}:
            fill = value
        else:
            rejected.append("fill")

    allow_short_a = False
    if _present(raw, "short"):
        value = _text(raw.get("short"))
        if value == "1":
            allow_short_a = True
        elif value == "0":
            allow_short_a = False
        else:
            rejected.append("short")

    include_costs = True
    if _present(raw, "cost"):
        value = _text(raw.get("cost"))
        if value == "statutory":
            include_costs = True
        elif value == "zero":
            include_costs = False
        else:
            rejected.append("cost")

    ordered = tuple(key for key in _KEYS if key in rejected)
    return SpecLink(
        start=start,
        end=end,
        hedge=hedge,
        hedge_window=hedge_window,
        z_window=z_window,
        entry_z=entry_z,
        exit_z=exit_z,
        fill=fill,
        allow_short_a=allow_short_a,
        include_costs=include_costs,
        rejected=ordered,
    )


def encode_spec(spec: SpecLink, *, pair: str, lang: str) -> dict[str, str]:
    pair_text = str(pair).strip()
    if pair_text.isdigit():
        pair_text = pair_text.zfill(6)
    return {
        "pair": pair_text,
        "lang": str(lang),
        "start": spec.start.isoformat(),
        "end": spec.end.isoformat(),
        "hedge": spec.hedge,
        "hw": str(int(spec.hedge_window)),
        "zw": str(int(spec.z_window)),
        "zin": _format_tenth(spec.entry_z),
        "zout": _format_tenth(spec.exit_z),
        "fill": spec.fill,
        "short": "1" if spec.allow_short_a else "0",
        "cost": "zero" if not spec.include_costs else "statutory",
    }


def _present(raw: Mapping[str, Any], key: str) -> bool:
    return key in raw and raw.get(key) is not None


def _text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[-1] if value else ""
    return str(value).strip()


def _as_int(text: str) -> int | None:
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _as_tenth(text: str) -> float | None:
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    scaled = round(number * 10)
    if abs(number * 10 - scaled) > 1e-6:
        return None
    return scaled / 10


def _as_date(text: str) -> date | None:
    if not _DATE.fullmatch(text):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _format_tenth(value: float) -> str:
    text = f"{float(value):.1f}"
    if text.endswith(".0"):
        return text[:-2]
    return text
