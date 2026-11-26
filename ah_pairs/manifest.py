"""刷新清单。每条序列分开记录数据截止日、状态和错误，写进 data/published/refresh_manifest.json。

数据截止日是序列最后一根 K 线的日期，和任务运行时间分开。任务跑过不等于数据更新到了当天。
状态：updated 正常更新；stale 下载失败沿用旧缓存；failed 没有可用数据；skipped 本轮跳过；empty 下载成功但没有记录。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CORE = "core"
AUX = "aux"

UPDATED = "updated"
STALE = "stale"
FAILED = "failed"
SKIPPED = "skipped"
EMPTY = "empty"


def make_item(
    name: str,
    *,
    kind: str = AUX,
    status: str = UPDATED,
    data_end=None,
    rows: int | None = None,
    source: str = "",
    detail: str = "",
) -> dict:
    end = None
    if data_end is not None:
        try:
            stamp = pd.Timestamp(data_end)
            end = None if pd.isna(stamp) else str(stamp.date())
        except (TypeError, ValueError):
            end = str(data_end)
    return {
        "name": str(name),
        "kind": str(kind),
        "status": str(status),
        "data_end": end,
        "rows": int(rows) if rows is not None else None,
        "source": str(source or ""),
        "detail": str(detail or ""),
    }


def overall_status(items) -> str:
    rows = list(items)
    if any(item["kind"] == CORE and item["status"] == FAILED for item in rows):
        return "failed"
    if any(item["status"] in {FAILED, STALE} for item in rows):
        return "partial"
    return "ok"


def core_failed(items) -> bool:
    return any(item["kind"] == CORE and item["status"] == FAILED for item in items)


def write_manifest(items, path, *, extra: dict | None = None) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    payload = {
        "generated_at_utc": now.isoformat(),
        "generated_at_beijing": now.tz_convert("Asia/Shanghai").isoformat(),
        "status": overall_status(items),
        "items": [dict(item) for item in items],
    }
    if extra:
        payload.update(extra)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return payload


def summary_lines(items) -> list[str]:
    lines = []
    for item in items:
        end = item["data_end"] or "无数据"
        line = f"{item['name']}: {item['status']}，截至 {end}"
        if item["detail"]:
            line += f"。{item['detail']}"
        lines.append(line)
    return lines


def ci_annotations(items) -> list[str]:
    """GitHub Actions 的注解。失败和沿用缓存直接出现在运行摘要页，不用翻日志。"""
    lines = []
    for item in items:
        if item["status"] == UPDATED:
            continue
        end = item["data_end"] or "无数据"
        msg = f"{item['name']}: {item['status']}，数据截至 {end}。{item['detail']}".strip()
        level = "error" if item["kind"] == CORE and item["status"] == FAILED else "warning"
        lines.append(f"::{level}::{msg}")
    return lines
