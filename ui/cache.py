"""重计算缓存。信号和回测分开：只改阈值或成本时，不重估协整和 β。"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from ah_pairs.analysis import analyze_frame, build_signals, entry_sensitivity, walk_forward
from ah_pairs.backtest import (
    contrast_costs,
    params_from_json,
    params_from_signal_token,
    params_from_trade_token,
    run_backtest,
)
from ah_pairs.data import load_pair
from ah_pairs.inference import block_bootstrap_sharpe, sharpe_probability
from ah_pairs.market import load_ah_premium_index, load_southbound, read_ah_premium_index, read_southbound
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


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_southbound() -> pd.Series:
    series = read_southbound()
    return series if series is not None else pd.Series(dtype=float, name="southbound")


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_ah_index() -> pd.Series:
    series = read_ah_premium_index()
    return series if series is not None else pd.Series(dtype=float, name="hsahp")


def refresh_southbound_cache() -> str:
    """返回空字符串表示成功，否则是给页面看的说明。"""
    try:
        load_southbound(refresh=True)
    except Exception as exc:
        cached_southbound.clear()
        return str(exc)
    cached_southbound.clear()
    return ""


def refresh_ah_index_cache() -> str:
    try:
        load_ah_premium_index(refresh=True)
    except Exception as exc:
        cached_ah_index.clear()
        return str(exc)
    cached_ah_index.clear()
    return ""


@st.cache_data(show_spinner=False)
def cached_signals(a_code: str, h_code: str, warm: str, end: str, eval_start: str, token: str, north_key: str):
    """协整、β 和 z。token 不含开仓阈值和成本。"""
    params = params_from_signal_token(token)
    prices, notes = cached_pair(a_code, h_code, warm, end)
    north = None
    if north_key:
        try:
            north = cached_northbound()
        except Exception:
            north = None
    prepared = analyze_frame(
        prices,
        params,
        eval_start=pd.Timestamp(eval_start),
        northbound=north,
        full_diagnostics=True,
        alt_hedge=False,
        with_backtest=False,
    )
    return notes, prepared


@st.cache_data(show_spinner=False)
def cached_backtest(view: pd.DataFrame, token: str):
    params = params_from_trade_token(token)
    result, trades, stats = run_backtest(view, params)
    sensitivity = entry_sensitivity(result, params)
    return result, trades, stats, sensitivity


@st.cache_data(show_spinner=False)
def cached_alt_beta(a_code: str, h_code: str, warm: str, end: str, eval_start: str, token: str) -> pd.DataFrame:
    """另一条 β。滚动 OLS 页面补卡尔曼，卡尔曼页面补滚动 OLS。"""
    params = params_from_signal_token(token)
    prices, _notes = cached_pair(a_code, h_code, warm, end)
    other_hedge = "kalman" if params.hedge == "rolling" else "rolling"
    other = replace(params, hedge=other_hedge, kalman_delta_auto=False)
    signals = build_signals(prices, other, alt=False)
    start = pd.Timestamp(eval_start).normalize()
    view = signals.loc[signals.index >= start, ["beta"]].copy()
    name = "beta_kalman" if other_hedge == "kalman" else "beta_rolling"
    return view.rename(columns={"beta": name})


@st.cache_data(show_spinner=False)
def cached_walk(token: str, frame: pd.DataFrame) -> dict:
    return walk_forward(frame, params_from_json(token))


@st.cache_data(show_spinner=False)
def cached_costs(token: str, frame: pd.DataFrame) -> pd.DataFrame:
    return contrast_costs(frame, params_from_json(token))


@st.cache_data(show_spinner=False)
def cached_sharpe(values: tuple, n_trials: int, ann_factor: float) -> tuple[dict, dict]:
    return block_bootstrap_sharpe(values, ann_factor=ann_factor), sharpe_probability(values, n_trials)


@st.cache_data(show_spinner=False)
def cached_stability(frame: pd.DataFrame):
    grid = stability_grid(frame["log_a"], frame["log_h"], frame.index)
    return {
        "grid": grid,
        "report": fragility_report(grid),
        "by_length": length_summary(grid),
        "checks": slice_checks(frame),
    }
