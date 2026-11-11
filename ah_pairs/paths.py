"""项目内数据路径。

Railway 挂上持久卷之后会设置 RAILWAY_VOLUME_MOUNT_PATH。
也可以用 AH_CACHE_DIR 指到别的目录。未设置时仍写在仓库的 data/cache，该目录不进 git。
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UNIVERSE_PATH = DATA_DIR / "ah_universe.csv"


def _cache_dir() -> Path:
    override = os.environ.get("AH_CACHE_DIR") or os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
    if override:
        return Path(override)
    return DATA_DIR / "cache"


CACHE_DIR = _cache_dir()
SCAN_DIR = CACHE_DIR / "scan"
# 定时任务写入、可以进 git 的副本。冷启动的页面在 data/cache 为空时读这里。
PUBLISHED_DIR = DATA_DIR / "published"
DESK_DIR = DATA_DIR / "desk"
