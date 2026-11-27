"""跨页状态：语言、选中股票、规格。与 URL 查询参数同步。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

import os

from ah_pairs.backtest import StrategyParams
from ah_pairs.data import PUBLISHED_DIR
from ah_pairs.i18n import set_lang, tx
from ah_pairs.links import SpecLink, encode_spec, parse_spec_query
from ah_pairs.universe import LIQUID_A_CODES, a_to_yahoo
from ui.cache import cached_universe

SAMPLE_START = date(2018, 1, 1)


def network_allowed() -> bool:
    return os.environ.get("AH_ALLOW_NETWORK", "1").strip().lower() not in {"0", "false", "no", "off"}


def public_mode() -> bool:
    """公开只读部署：关网络或订阅只读时，隐藏维护动作、只看已发布数据。"""
    return not network_allowed()


def init_lang() -> str:
    if "lang" not in st.session_state:
        requested = st.query_params.get("lang")
        st.session_state["lang"] = requested if requested in {"zh", "zh-Hant", "en"} else "zh"
    set_lang(st.session_state["lang"])
    return st.session_state["lang"]


def lang_selector() -> str:
    lang = init_lang()
    pick = st.segmented_control(
        tx("语言", "Language"),
        ["zh", "zh-Hant", "en"],
        default=lang,
        format_func=lambda code: {"zh": "简", "zh-Hant": "繁", "en": "EN"}[code],
        key="lang_pick",
        label_visibility="collapsed",
    )
    if pick and pick != lang:
        st.session_state["lang"] = pick
        set_lang(pick)
        st.query_params["lang"] = pick
        st.rerun()
    return lang


def published_a_codes() -> set[str]:
    """已发布行情覆盖的 A 股代码。"""
    prices = PUBLISHED_DIR / "prices"
    out = set()
    if prices.exists():
        stems = {p.stem for p in prices.glob("*.parquet")}
        for code in LIQUID_A_CODES:
            if a_to_yahoo(code).replace("=", "_").replace("^", "_") in stems:
                out.add(str(code).zfill(6))
    return out


def load_universe_filtered() -> pd.DataFrame:
    universe = cached_universe().copy()
    if public_mode():
        codes = published_a_codes()
        if codes:
            universe = universe[universe["a_code"].astype(str).str.zfill(6).isin(codes)]
    liquid_order = {code: i for i, code in enumerate(LIQUID_A_CODES)}
    universe["_ord"] = universe["a_code"].map(lambda code: liquid_order.get(code, 10_000))
    return universe.sort_values(["_ord", "name"]).reset_index(drop=True)


def current_pair(universe: pd.DataFrame) -> str:
    codes = universe["a_code"].tolist()
    if "pair_select" not in st.session_state or st.session_state["pair_select"] not in codes:
        requested = str(st.query_params.get("pair", st.query_params.get("code", ""))).zfill(6)
        st.session_state["pair_select"] = requested if requested in codes else ("601318" if "601318" in codes else codes[0])
    return st.session_state["pair_select"]


def spec_state() -> tuple[StrategyParams, tuple, bool]:
    """(参数, (开始, 结束), 含成本)。首次从链接种子，之后以会话状态为准。"""
    if "applied_params" not in st.session_state:
        spec = parse_spec_query(dict(st.query_params))
        from ui.spec_dialog import build_params

        st.session_state["applied_params"] = build_params(spec)
        st.session_state["applied_dates"] = (spec.start, spec.end)
        st.session_state["applied_costs_on"] = spec.include_costs
    return (
        st.session_state["applied_params"],
        st.session_state["applied_dates"],
        bool(st.session_state["applied_costs_on"]),
    )


def sync_query(pair: str, start, end) -> None:
    params, _, include_costs = spec_state()
    spec = SpecLink(
        start=pd.Timestamp(start).date(),
        end=pd.Timestamp(end).date(),
        hedge=params.hedge,
        hedge_window=int(params.hedge_window),
        z_window=int(params.z_window),
        entry_z=float(params.entry_z),
        exit_z=float(params.exit_z),
        fill=params.fill,
        allow_short_a=bool(params.allow_short_a),
        include_costs=bool(include_costs),
    )
    desired = encode_spec(spec, pair=str(pair), lang=str(st.session_state.get("lang", "zh")))
    current = {key: st.query_params.get(key) for key in desired}
    if current != desired:
        st.query_params.update(desired)
