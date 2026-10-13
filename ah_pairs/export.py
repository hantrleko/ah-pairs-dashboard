"""研究导出：指标表、LaTeX 和 PNG。"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


def metrics_table(result: dict, name: str, a_code: str, h_code: str) -> pd.DataFrame:
    eg = result["eg"]
    hl = result["half_life_static"]
    adf = result.get("residual_adf") or {}
    stats = result["stats"]
    frame = result["frame"]
    rows = [
        ("公司", name),
        ("A股", a_code),
        ("H股", h_code),
        ("最近交易日", str(pd.Timestamp(frame.index.max()).date())),
        ("重叠交易日", eg["nobs"]),
        ("EG t", eg["t_stat"]),
        ("EG p", eg["pvalue"]),
        ("全样本 beta", eg["beta"]),
        ("AR1 phi", hl.get("phi")),
        ("半衰期", hl.get("half_life")),
        ("残差 ADF 统计量", adf.get("adf_stat")),
        ("残差 ADF p", adf.get("adf_pvalue")),
        ("净收益", stats["total_return"]),
        ("夏普", stats["sharpe"]),
        ("最大回撤", stats["max_drawdown"]),
        ("交易次数", stats["n_trades"]),
    ]
    return pd.DataFrame(rows, columns=["项目", "数值"])


def metrics_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{lr}",
        r"\hline",
    ]
    for _, row in table.iterrows():
        value = row["数值"]
        if isinstance(value, float) and np.isfinite(value):
            text = f"{value:.4g}"
        else:
            text = "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else str(value)
        lines.append(f"{row['项目']} & {text} \\\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def _png(draw) -> bytes | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=140)
    draw(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def spread_png(frame: pd.DataFrame, z_window: int) -> bytes | None:
    def draw(ax):
        spread = frame["spread"]
        mean = spread.rolling(z_window, min_periods=z_window).mean()
        std = spread.rolling(z_window, min_periods=z_window).std()
        ax.fill_between(frame.index, mean - 2 * std, mean + 2 * std, color="#1f4e79", alpha=0.12)
        ax.plot(frame.index, spread, color="#1f4e79", lw=1.1)
        ax.set_title("交易价差")

    return _png(draw)


def equity_png(frame: pd.DataFrame) -> bytes | None:
    def draw(ax):
        ax.plot(frame.index, frame["equity"], color="#1f4e79", lw=1.3, label="策略")
        ax.set_title("策略净值")

    return _png(draw)
