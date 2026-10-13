"""Plotly 图。交易图只用因果价差和已经成交的持仓。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NAVY = "#1f4b3a"
RUST = "#b4532a"
GREEN = "#1d6b45"
RED = "#9b3a3a"
SLATE = "#8a8175"
GOLD = "#a6843d"
INK = "#1c1915"
MUTED = "#6f675e"
GRID = "#efe6d8"
FONT = "Segoe UI, Microsoft YaHei, sans-serif"


def _layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(size=15, color=INK, family=FONT),
            pad=dict(b=8),
        ),
        font=dict(family=FONT, size=12, color=INK),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#fffcf8", bordercolor="#e7dfd2", font=dict(color=INK, family=FONT)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, color=MUTED),
        ),
        margin=dict(l=48, r=18, t=64, b=32),
        height=height,
    )
    axis = dict(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color=MUTED),
        title_font=dict(size=12, color=MUTED),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_annotations(font=dict(size=12, color=MUTED, family=FONT))
    return fig


def _position_bands(fig: go.Figure, frame: pd.DataFrame, row: int | None = None) -> None:
    position = frame["position"].fillna(0).astype(int)
    if position.empty:
        return
    start = None
    current = 0
    index = frame.index
    for i, (ts, pos) in enumerate(position.items()):
        if start is None:
            start = ts
            current = int(pos)
            continue
        if int(pos) != current:
            _band(fig, start, index[i - 1], current, row)
            start = ts
            current = int(pos)
    _band(fig, start, index[-1], current, row)


def _band(fig, start, end, position, row) -> None:
    if position == 0 or start is None:
        return
    color = "rgba(31,107,69,0.10)" if position > 0 else "rgba(155,58,58,0.10)"
    yref = "y domain" if row in (None, 1) else f"y{row} domain"
    fig.add_shape(
        type="rect",
        x0=start,
        x1=end,
        y0=0,
        y1=1,
        yref=yref,
        xref="x",
        fillcolor=color,
        line_width=0,
        layer="below",
    )


def prices_figure(frame: pd.DataFrame) -> go.Figure:
    base_a = frame["a_cny"].iloc[0]
    base_h = frame["h_cny"].iloc[0]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["a_cny"] / base_a * 100,
            name="A 股",
            line=dict(color=NAVY, width=1.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["h_cny"] / base_h * 100,
            name="H 股",
            line=dict(color=RUST, width=1.6),
        )
    )
    return _layout(fig, "价格，起点 = 100")


def premium_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["premium"] * 100,
            name="A/H 溢价",
            line=dict(color=NAVY, width=1.6),
            fill="tozeroy",
            fillcolor="rgba(31,75,58,0.08)",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="#d9d0c3")
    fig.update_yaxes(ticksuffix="%")
    return _layout(fig, "A/H 溢价")


def signal_figure(frame: pd.DataFrame, entry: float, exit_: float, stop: float) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.62, 0.38],
        subplot_titles=("z-score", "成交后的权重"),
    )
    fig.add_trace(
        go.Scatter(x=frame.index, y=frame["z"], name="z-score", line=dict(color=NAVY, width=1.3)),
        row=1,
        col=1,
    )
    for level, name, color, dash in (
        (entry, "开仓", RED, "dash"),
        (-entry, "开仓", GREEN, "dash"),
        (exit_, "平仓", SLATE, "dot"),
        (-exit_, "平仓", SLATE, "dot"),
    ):
        fig.add_hline(y=level, line_dash=dash, line_color=color, line_width=1, row=1, col=1)
    if stop and stop > 0:
        fig.add_hline(y=stop, line_dash="dot", line_color=GOLD, line_width=1, row=1, col=1)
        fig.add_hline(y=-stop, line_dash="dot", line_color=GOLD, line_width=1, row=1, col=1)
    fig.add_trace(
        go.Scatter(x=frame.index, y=frame["w_a"], name="A 股权重", line=dict(color=NAVY, width=1.3)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=frame.index, y=frame["w_h"], name="H 股权重", line=dict(color=RUST, width=1.3)),
        row=2,
        col=1,
    )
    _position_bands(fig, frame, row=1)
    fig.update_yaxes(title="z", row=1, col=1)
    fig.update_yaxes(title="权重", row=2, col=1)
    return _layout(fig, "信号与持仓", height=560)


def spread_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["spread"],
            name="交易价差",
            line=dict(color=NAVY, width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["spread_beta1"],
            name="β = 1",
            line=dict(color=RUST, width=1.1, dash="dot"),
        )
    )
    return _layout(fig, "价差")


def hedge_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=frame.index, y=frame["beta"], name="动态 beta", line=dict(color=NAVY, width=1.5))
    )
    fig.add_hline(y=1, line_dash="dot", line_color="#d9d0c3")
    return _layout(fig, "动态对冲比率")


def equity_figure(frame: pd.DataFrame) -> go.Figure:
    ret_a = frame["ret_a"].fillna(0.0)
    ret_h = frame["ret_h"].fillna(0.0)
    bh_a = (1.0 + ret_a).cumprod()
    bh_h = (1.0 + ret_h).cumprod()
    bh_a = bh_a / bh_a.iloc[0]
    bh_h = bh_h / bh_h.iloc[0]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=("净值", "回撤"),
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["equity"],
            name="配对策略",
            line=dict(color=NAVY, width=2),
            fill="tozeroy",
            fillcolor="rgba(31,75,58,0.08)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=frame.index, y=bh_a, name="只持有 A", line=dict(color=SLATE, width=1, dash="dot")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=bh_h,
            name="只持有 H（人民币）",
            line=dict(color=RUST, width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["drawdown"] * 100,
            name="回撤",
            fill="tozeroy",
            fillcolor="rgba(155,58,58,0.14)",
            line=dict(color=RED, width=1),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title="净值", row=1, col=1)
    fig.update_yaxes(title="%", row=2, col=1)
    return _layout(fig, "净值与回撤", height=520)


def static_resid_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["static_resid"],
            name="静态残差",
            line=dict(color=NAVY, width=1.3),
        )
    )
    fig.add_hline(y=0, line_color=SLATE, line_width=1)
    return _layout(fig, "样本内静态残差")


def northbound_figure(frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["spread"].diff(),
            name="价差日变化",
            line=dict(color=NAVY, width=1),
        ),
        secondary_y=False,
    )
    if "northbound" in frame.columns:
        fig.add_trace(
            go.Bar(
                x=frame.index,
                y=frame["northbound"],
                name="北向净买入（亿元）",
                marker_color="rgba(196,89,17,0.45)",
            ),
            secondary_y=True,
        )
    fig.update_yaxes(title="Δ价差", secondary_y=False)
    fig.update_yaxes(title="亿元", secondary_y=True)
    return _layout(fig, "价差变化与北向资金净买入")
