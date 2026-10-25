"""重计算缓存。键是参数 JSON 和行情本身，滑块在提交前不会触发重算。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ah_pairs.analysis import analyze_frame, walk_forward
from ah_pairs.backtest import contrast_costs, params_from_json
from ah_pairs.data import load_pair
from ah_pairs.inference import block_bootstrap_sharpe, sharpe_probability
from ah_pairs.market import load_ah_premium_index, load_southbound
from ah_pairs.northbound import load_northbound
from ah_pairs.stability import fragility_report, length_summary, slice_checks, stability_grid
from ah_pairs.universe import load_universe


@st.cache_data(show_spinner=False)
def cached_universe() -> pd.DataFrame:
    return load_universe()


@st.cache_data(show_spinner=False)
def cached_pair(a_code: str, h_code: str, start: str, end: str):
    return load_pair(a_code, h_code, start, end, refresh=False)


@st.cache_data(show_spinner=False)
def cached_northbound() -> pd.Series:
    return load_northbound(refresh=False)


@st.cache_data(show_spinner=False)
def cached_southbound() -> pd.Series:
    return load_southbound(refresh=False)


@st.cache_data(show_spinner=False)
def cached_ah_index() -> pd.Series:
    return load_ah_premium_index(refresh=False)


@st.cache_data(show_spinner=False)
def cached_analysis(a_code: str, h_code: str, warm: str, end: str, eval_start: str, token: str, north_key: str):
    params = params_from_json(token)
    prices, notes = cached_pair(a_code, h_code, warm, end)
    north = None
    if north_key:
        try:
            north = cached_northbound()
        except Exception:
            north = None
    result = analyze_frame(prices, params, eval_start=pd.Timestamp(eval_start), northbound=north)
    return notes, result


@st.cache_data(show_spinner=False)
def cached_walk(token: str, frame: pd.DataFrame) -> dict:
    return walk_forward(frame, params_from_json(token))


@st.cache_data(show_spinner=False)
def cached_costs(token: str, frame: pd.DataFrame) -> pd.DataFrame:
    return contrast_costs(frame, params_from_json(token))


@st.cache_data(show_spinner=False)
def cached_sharpe(values: tuple, n_trials: int) -> tuple[dict, dict]:
    return block_bootstrap_sharpe(values), sharpe_probability(values, n_trials)


@st.cache_data(show_spinner=False)
def cached_stability(frame: pd.DataFrame):
    grid = stability_grid(frame["log_a"], frame["log_h"], frame.index)
    return {
        "grid": grid,
        "report": fragility_report(grid),
        "by_length": length_summary(grid),
        "checks": slice_checks(frame),
    }
