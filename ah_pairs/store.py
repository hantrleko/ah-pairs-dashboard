"""在本地缓存和已发布副本之间选文件。

调用方传入默认缓存目录。只有正在读那个默认目录时，才回退到 data/published。
测试或显式传入的其他目录不会读到仓库里的已发布文件。
"""

from __future__ import annotations

from pathlib import Path


def same_dir(left: Path, right: Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return Path(left) == Path(right)


def prefer(cache_path: Path, published_path: Path, cache_dir: Path, default_cache: Path) -> Path | None:
    """返回较新的那一份。cache_dir 不是默认缓存时，不看已发布副本。"""
    choices: list[Path] = []
    choices.extend(_existing(cache_path))
    if same_dir(cache_dir, default_cache):
        choices.extend(_existing(published_path))
    if not choices:
        return None
    return max(choices, key=lambda path: path.stat().st_mtime)


def _existing(path: Path) -> list[Path]:
    found = []
    if path.exists():
        found.append(path)
    if path.suffix == ".parquet":
        legacy = path.with_suffix(".csv")
        if legacy.exists():
            found.append(legacy)
    return found


def publish_file(src: Path, dest: Path) -> bool:
    """把缓存文件抄到已发布目录。空文件不覆盖已有副本。"""
    if not src.exists() or src.stat().st_size <= 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(src.read_bytes())
    tmp.replace(dest)
    side = src.with_suffix(".source")
    if side.exists():
        dest.with_suffix(".source").write_text(side.read_text(encoding="utf-8"), encoding="utf-8")
    return True
