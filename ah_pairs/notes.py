"""把下载说明拆成可翻译的字段。旧的中文句子仍保留给命令行。"""

from __future__ import annotations


def parse_notes(notes: list[str] | None) -> dict:
    sources: dict[str, str] = {}
    meta = {
        "fx_lag": 0,
        "download_failed": False,
        "adjust_mismatch": False,
    }
    prose: list[str] = []
    for note in notes or []:
        if note.startswith("meta:src:"):
            _meta, _src, leg, val = note.split(":", 3)
            sources[leg] = val
        elif note.startswith("meta:sessions:"):
            _meta, _key, common, one_sided = note.split(":")
            meta["download_common"] = int(common)
            meta["download_one_sided"] = int(one_sided)
        elif note.startswith("meta:fx_lag:"):
            meta["fx_lag"] = int(note.rsplit(":", 1)[-1])
        elif note.startswith("meta:download_failed:"):
            meta["download_failed"] = note.endswith("1")
        elif note.startswith("meta:adjust_mismatch:"):
            meta["adjust_mismatch"] = note.endswith("1")
        elif note.startswith("A股来源："):
            sources.setdefault("a", note.split("：", 1)[1])
        elif note.startswith("H股来源："):
            sources.setdefault("h", note.split("：", 1)[1])
        elif note.startswith("汇率来源："):
            sources.setdefault("fx", note.split("：", 1)[1])
        else:
            prose.append(note)
    return {"sources": sources, "meta": meta, "prose": prose}
