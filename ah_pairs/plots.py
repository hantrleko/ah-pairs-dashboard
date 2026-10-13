"""Plotly 图。交易图只用因果价差和已经成交的持仓。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NAVY = "#1f4e79"
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
            font=dict(size=13, color=INK),
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
            line=dict(color=NAVY, width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["h_cny"] / base_h * 100,
            name="H 股",
            line=dict(color="#e07a2f", width=2.2),
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
            fillcolor="rgba(31,78,121,0.08)",
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
        go.Scatter(x=frame.index, y=frame["w_h"], name="H 股权重", line=dict(color="#e07a2f", width=1.5)),
        row=2,
        col=1,
    )
    _position_bands(fig, frame, row=1)
    fig.update_yaxes(title="z", row=1, col=1)
    fig.update_yaxes(title="权重", row=2, col=1)
    return _layout(fig, "信号与持仓", height=560)


def spread_figure(frame: pd.DataFrame, z_window: int = 60) -> go.Figure:
    spread = frame["spread"]
    mean = spread.rolling(z_window, min_periods=z_window).mean()
    std = spread.rolling(z_window, min_periods=z_window).std(ddof=1)
    fig = go.Figure()
    for width, opacity in ((2.0, 0.08), (1.5, 0.12), (1.0, 0.16)):
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=mean + width * std,
                name=f"+{width:g}σ",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=mean - width * std,
                name="±2σ" if width == 2.0 else f"−{width:g}σ",
                fill="tonexty",
                fillcolor=f"rgba(31,78,121,{opacity})",
                line=dict(width=0),
                showlegend=width == 2.0,
                legendgroup="band",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=mean,
            name="滚动均值",
            line=dict(color=SLATE, width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=spread,
            name="交易价差",
            line=dict(color=NAVY, width=1.6),
        )
    )
    _trade_markers(fig, frame, spread)
    return _layout(fig, "价差与开平仓", height=420)


def _trade_markers(fig: go.Figure, frame: pd.DataFrame, series: pd.Series, row: int | None = None) -> None:
    if "position" not in frame.columns:
        return
    position = frame["position"].fillna(0)
    previous = position.shift(1).fillna(0)
    entries = frame.loc[(position != 0) & (previous == 0)]
    exits = frame.loc[(position == 0) & (previous != 0)]
    extra = {"row": row, "col": 1} if row else {}
    if len(entries):
        fig.add_trace(
            go.Scatter(
                x=entries.index,
                y=series.reindex(entries.index),
                name="开仓",
                mode="markers",
                marker=dict(symbol="triangle-up", size=9, color=GREEN),
            ),
            **extra,
        )
    if len(exits):
        fig.add_trace(
            go.Scatter(
                x=exits.index,
                y=series.reindex(exits.index),
                name="平仓",
                mode="markers",
                marker=dict(symbol="x", size=8, color=RED),
            ),
            **extra,
        )


def core_signal_figure(frame: pd.DataFrame, z_window: int, entry: float, exit_: float, stop: float) -> go.Figure:
    """价差、σ 带、开平仓和 z-score 放在同一张图里。"""
    spread = frame["spread"]
    mean = spread.rolling(z_window, min_periods=z_window).mean()
    std = spread.rolling(z_window, min_periods=z_window).std(ddof=1)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.62, 0.38],
        subplot_titles=("价差", "z-score"),
    )
    for width, opacity in ((2.0, 0.10), (1.5, 0.14), (1.0, 0.18)):
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=mean + width * std,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=mean - width * std,
                name="±2σ" if width == 2 else "",
                fill="tonexty",
                fillcolor=f"rgba(31,78,121,{opacity})",
                line=dict(width=0),
                showlegend=width == 2,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(x=frame.index, y=mean, name="滚动均值", line=dict(color=SLATE, width=1, dash="dot")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=frame.index, y=spread, name="交易价差", line=dict(color=NAVY, width=1.8)),
        row=1,
        col=1,
    )
    _trade_markers(fig, frame, spread, row=1)
    fig.add_trace(
        go.Scatter(x=frame.index, y=frame["z"], name="z-score", line=dict(color="#e07a2f", width=1.4)),
        row=2,
        col=1,
    )
    for level, color, dash in (
        (entry, RED, "dash"),
        (-entry, GREEN, "dash"),
        (exit_, SLATE, "dot"),
        (-exit_, SLATE, "dot"),
    ):
        fig.add_hline(y=level, line_dash=dash, line_color=color, line_width=1, row=2, col=1)
    if stop and stop > 0:
        fig.add_hline(y=stop, line_dash="dot", line_color=GOLD, line_width=1, row=2, col=1)
        fig.add_hline(y=-stop, line_dash="dot", line_color=GOLD, line_width=1, row=2, col=1)
    _position_bands(fig, frame, row=1)
    _position_bands(fig, frame, row=2)
    return _layout(fig, "价差、开平仓与 z-score", height=640)


def hedge_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "beta_rolling" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["beta_rolling"],
                name="滚动 OLS",
                line=dict(color=NAVY, width=1.6),
            )
        )
    if "beta_kalman" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["beta_kalman"],
                name="卡尔曼",
                line=dict(color=RUST, width=1.5),
            )
        )
    if "beta_rolling" not in frame.columns and "beta_kalman" not in frame.columns:
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["beta"], name="动态 beta", line=dict(color=NAVY, width=1.5))
        )
    fig.add_hline(y=1, line_dash="dot", line_color="#d9d0c3")
    return _layout(fig, "滚动 OLS 与卡尔曼 β")


def equity_figure(frame: pd.DataFrame) -> go.Figure:
    ret_a = frame["ret_a"].fillna(0.0)
    ret_h = frame["ret_h"].fillna(0.0)
    bh_a = (1.0 + ret_a).cumprod()
    bh_h = (1.0 + ret_h).cumprod()
    bh_eq = (1.0 + 0.5 * ret_a + 0.5 * ret_h).cumprod()
    bh_a = bh_a / bh_a.iloc[0]
    bh_h = bh_h / bh_h.iloc[0]
    bh_eq = bh_eq / bh_eq.iloc[0]
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
            fillcolor="rgba(31,78,121,0.08)",
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
            y=bh_eq,
            name="等权 A+H",
            line=dict(color=GOLD, width=1.2, dash="dash"),
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


def stability_heatmap(grid: pd.DataFrame) -> go.Figure:
    """窗口长度 × 窗口终点的协整 p 值。颜色越深越接近拒绝无协整。"""
    if grid.empty:
        return _layout(go.Figure(), "稳定性地图")
    pivot = grid.pivot_table(index="length", columns="end", values="pvalue", aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=[pd.Timestamp(value).strftime("%Y-%m") for value in pivot.columns],
            y=[f"{int(length)} 日" for length in pivot.index],
            colorscale=[
                [0.0, "#1f4e79"],
                [0.05, "#8eacc4"],
                [0.2, "#e07a2f"],
                [1.0, "#9b3a3a"],
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(title="p 值"),
            hovertemplate="终点 %{x}<br>窗口 %{y}<br>p=%{z:.3f}<extra></extra>",
        )
    )
    return _layout(fig, "协整 p 值：窗口长度 × 终点", height=420)


def rolling_p_figure(pvalues: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pvalues.index,
            y=pvalues,
            name="滚动 EG p 值",
            line=dict(color=NAVY, width=1.5),
        )
    )
    fig.add_hline(y=0.05, line_dash="dot", line_color=RED)
    fig.update_yaxes(range=[0, 1])
    return _layout(fig, "滚动 Engle–Granger p 值", height=320)
