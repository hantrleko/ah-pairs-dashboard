"""把一次分析打成可以放进附录的压缩包。不含密钥。"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict

import pandas as pd

from ah_pairs.export import equity_png, metrics_latex, metrics_table, spread_png


def _method_note(name: str, a_code: str, h_code: str, result: dict) -> str:
    eg = result["eg"]
    stats = result["stats"]
    params = result["params"]
    kalman = result.get("kalman_fit") or {}
    one_sided = result.get("n_one_sided", stats.get("n_one_sided", 0))
    common = result.get("n_common", stats.get("n_common", stats.get("n_days")))
    return "\n".join(
        [
            f"# {name}（{a_code} / {h_code}）方法摘记",
            "",
            "A 股和 H 股不能互相兑换。下面的数字是历史回测，不是无风险套利，也不是投资建议。",
            "",
            f"- 评估区间同时开市 {common} 天，另有 {one_sided} 天只有一边开市。",
            "- 协整、β 和 z 只在同时开市日估计。β[t] 和卡尔曼预测状态都不用当天价格。z 的均值和标准差含当天价差，成交默认在下一个同时开市日收盘。",
            "- 单边开市日不新开仓、不调 β。已有仓位计入开市一边的涨跌，停市一边的收益记 0。",
            f"- 做空 A 股：{'允许。多数 A 股不一定能融到券，这一档偏乐观。' if params.allow_short_a else '不允许。A 股相对偏贵时不开仓。'}",
            f"- Engle–Granger p = {eg['pvalue']:.4g}，β = {eg['beta']:.3f}。p 值不是获利概率。残差 ADF 不能代替它。",
            f"- 扣费后净收益 {stats['total_return']:.2%}，夏普 {stats['sharpe']:.2f}，最大回撤 {stats['max_drawdown']:.2%}，已平仓 {stats['n_trades']} 次。",
            f"- 卡尔曼 δ = {kalman.get('delta', params.kalman_delta)}。{kalman.get('message', '')}",
            "- 稳定性地图的窗口互相重叠。BH 比例只是多重比较的参考，不是独立试验的 q 值。",
            "- Walk-forward 在训练段同时选开仓和平仓阈值。样本够长时按年扩展训练、下一年验证；不够长时只切一刀。验证段不参与挑选。",
            "- 夏普的区间来自区块自助法。压缩夏普概率把试验次数算进去，仍不是未来赚钱的概率。",
            "",
            "价格优先用 Yahoo 复权收盘，失败时用东财前复权。北向是两市合计，日频止于 2024-08-16。南向和恒生 AH 溢价指数是市场层面的描述，不是这只股票的持仓。",
        ]
    )


def research_zip(
    result: dict,
    name: str,
    a_code: str,
    h_code: str,
    z_window: int,
    extra: dict | None = None,
) -> bytes:
    summary = metrics_table(result, name, a_code, h_code)
    frame = result["frame"]
    trades = result["trades"]
    payload = {
        "name": name,
        "a_code": a_code,
        "h_code": h_code,
        "params": asdict(result["params"]),
        "eg_pvalue": result["eg"].get("pvalue"),
        "stats": {key: _plain(value) for key, value in result["stats"].items()},
        "kalman_fit": {key: _plain(value) for key, value in (result.get("kalman_fit") or {}).items()},
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("params.json", json.dumps(payload, ensure_ascii=False, indent=2, default=_plain))
        archive.writestr("metrics.csv", summary.to_csv(index=False))
        archive.writestr("metrics.tex", metrics_latex(summary))
        archive.writestr("trades.csv", trades.to_csv(index=False))
        archive.writestr("daily.csv", frame.reset_index().to_csv(index=False))
        archive.writestr("method.md", _method_note(name, a_code, h_code, result))
        spread = spread_png(frame, z_window)
        equity = equity_png(frame)
        if spread:
            archive.writestr("spread.png", spread)
        if equity:
            archive.writestr("equity.png", equity)
        if isinstance(extra, dict) and isinstance(extra.get("walk_forward"), pd.DataFrame):
            archive.writestr("walk_forward.csv", extra["walk_forward"].to_csv(index=False))
    return buffer.getvalue()


def _plain(value):
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value
