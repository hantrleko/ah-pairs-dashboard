"""A+H 双上市公司名单。"""

from __future__ import annotations

import pandas as pd

from ah_pairs.paths import UNIVERSE_PATH

# 成交通常较活跃、便于第一轮扫描的 A 股代码。名单里不存在的代码会在运行时丢掉。
LIQUID_A_CODES: tuple[str, ...] = (
    "601398",
    "601939",
    "601288",
    "601988",
    "600036",
    "601328",
    "601318",
    "601628",
    "601601",
    "601336",
    "601857",
    "600028",
    "601088",
    "600938",
    "600941",
    "601728",
    "600050",
    "601899",
    "002594",
    "000333",
    "600585",
    "601919",
    "600030",
    "601888",
    "601633",
    "000063",
    "601600",
    "600362",
    "601186",
    "601390",
    "601766",
    "000002",
    "600600",
    "601012",
    "300750",
)


def a_to_yahoo(code: str) -> str:
    """A 股代码转 Yahoo 代码。上交所为 .SS，深交所为 .SZ。"""
    c = str(code).strip().split(".")[0].zfill(6)
    if c.startswith(("5", "6", "9")):
        return f"{c}.SS"
    return f"{c}.SZ"


def h_to_yahoo(code: str) -> str:
    """H 股代码转 Yahoo 代码，例如 02318 -> 2318.HK。"""
    n = int(str(code).strip().split(".")[0])
    return f"{n:04d}.HK"


def _normalize_universe(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out["name"] = out["name"].astype(str).str.strip()
    out["a_code"] = (
        out["a_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    out["h_code"] = (
        out["h_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    )
    out["premium_snapshot"] = pd.to_numeric(out["premium_snapshot"], errors="coerce")
    out["a_yahoo"] = out["a_code"].map(a_to_yahoo)
    out["h_yahoo"] = out["h_code"].map(h_to_yahoo)
    out = out.dropna(subset=["name", "a_code", "h_code"])
    out = out.drop_duplicates("a_code").sort_values(["name", "a_code"])
    return out.reset_index(drop=True)


def load_universe(path=UNIVERSE_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    return _normalize_universe(frame)


def refresh_universe(path=UNIVERSE_PATH) -> pd.DataFrame:
    """从东财 AH 比价表刷新名单。溢价是刷新当时的报价，不是历史序列。"""
    import akshare as ak

    raw = ak.stock_zh_ah_spot_em()
    snapshot_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    frame = pd.DataFrame(
        {
            "name": raw.iloc[:, 1],
            "h_code": raw.iloc[:, 2],
            "a_code": raw.iloc[:, 5],
            "premium_snapshot": pd.to_numeric(raw.iloc[:, 9], errors="coerce"),
            "ratio_snapshot": pd.to_numeric(raw.iloc[:, 8], errors="coerce"),
            "snapshot_date": snapshot_date,
        }
    )
    frame = _normalize_universe(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["a_yahoo", "h_yahoo"]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    return frame


def liquid_subset(universe: pd.DataFrame) -> pd.DataFrame:
    codes = set(LIQUID_A_CODES)
    picked = universe[universe["a_code"].isin(codes)].copy()
    if picked.empty:
        return universe.head(20).copy()
    order = {code: i for i, code in enumerate(LIQUID_A_CODES)}
    picked["_ord"] = picked["a_code"].map(order)
    return picked.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
