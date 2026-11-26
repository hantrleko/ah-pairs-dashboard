"""信号解释卡的共享组件。前瞻记录页和单对研究页用同一份渲染。

卡片只回答三个问题：观察到了什么、规则允许什么、账户实际做了什么。
理由由引擎在决策时产生，页面不反推。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ah_pairs.i18n import tx
from ui.format import num

REASON_ZH = {
    "enter_long_spread": "价差偏低，达到入场阈值，做多价差（买A卖H）",
    "enter_short_spread": "价差偏高，达到入场阈值，做空价差（卖A买H）",
    "exit_z_out": "价差回到退出阈值内",
    "exit_stop": "止损线触发",
    "exit_max_hold": "持有期限到",
    "short_a_forced_exit": "A股融券名单移除，强制退出",
    "short_a_blocked": "需要做空A股，当前预设不允许",
    "gate_blocked": "开仓闸门不通过（协整或半衰期），不新开仓",
    "beta_blocked": "对冲比例超出允许范围，不新开仓",
    "one_sided": "单边行情日，不产生新信号、不成交，持仓照常记账",
    "hold": "维持",
    "flat": "空仓",
}

REASON_EN = {
    "enter_long_spread": "Spread cheap, entry threshold hit: long spread (long A, short H)",
    "enter_short_spread": "Spread rich, entry threshold hit: short spread (short A, long H)",
    "exit_z_out": "Spread back inside the exit band",
    "exit_stop": "Stop-loss hit",
    "exit_max_hold": "Max holding days reached",
    "short_a_forced_exit": "A-share borrow list removal forces exit",
    "short_a_blocked": "Needs shorting A; this preset disallows it",
    "gate_blocked": "Entry gate failed (cointegration or half-life); no new entries",
    "beta_blocked": "Hedge ratio outside the allowed band; no new entries",
    "one_sided": "One-sided session: no new signal, no fills; open legs still accrue",
    "hold": "Hold",
    "flat": "Flat",
}

CHANGE_ZH = {
    "new_trigger": "新触发",
    "exit_signal": "退出信号",
    "filled_entry": "已开仓",
    "filled_exit": "已平仓",
    "filled_flip": "已翻仓",
    "blocked": "持续受限",
    "one_sided": "单边行情日",
    "unchanged": "无变化",
    "backfilled": "补回放",
    "unknown": "无法判断",
}

CHANGE_EN = {
    "new_trigger": "New trigger",
    "exit_signal": "Exit signal",
    "filled_entry": "Filled entry",
    "filled_exit": "Filled exit",
    "filled_flip": "Flipped",
    "blocked": "Blocked",
    "one_sided": "One-sided day",
    "unchanged": "Unchanged",
    "backfilled": "Backfilled",
    "unknown": "Unknown",
}


def reason_label(reason: str) -> str:
    return tx(REASON_ZH.get(reason, reason or "—"), REASON_EN.get(reason, reason or "—"))


def change_label(change: str) -> str:
    return tx(CHANGE_ZH.get(change, change), CHANGE_EN.get(change, change))


def position_labels() -> dict[int, str]:
    return {
        0: tx("空仓", "flat"),
        1: tx("买A卖H", "long A / short H"),
        -1: tx("卖A买H", "short A / long H"),
    }


def render_ledger_card(row: dict, title: str | None = None) -> None:
    """从账本行渲染解释卡。row 是账本最后一行（或工作台行）。"""
    if title:
        st.markdown(f"### {title}")
    labels = position_labels()
    sig = int(row.get("signal") or 0)
    pending = row.get("pending")
    position = int(row.get("position") or 0)
    reason = row.get("reason") or ""
    blocked = reason in {"short_a_blocked", "gate_blocked", "beta_blocked"}
    col1, col2, col3 = st.columns(3)
    col1.markdown(tx("**观察到了什么**", "**What was observed**"))
    col1.write(
        tx(
            f"截止 {row.get('date', '—')}，z = {num(row.get('z'), 2)}，信号方向：{labels.get(sig, '—')}。",
            f"As of {row.get('date', '—')}, z = {num(row.get('z'), 2)}, signal direction: {labels.get(sig, '—')}.",
        )
    )
    col2.markdown(tx("**规则允许什么**", "**What the rules allowed**"))
    col2.write(
        tx(
            ("不允许新开仓：" if blocked else "规则结论：") + reason_label(reason) + "。",
            ("No new entry: " if blocked else "Rule outcome: ") + reason_label(reason) + ".",
        )
    )
    col3.markdown(tx("**账户实际做了什么**", "**What the account did**"))
    col3.write(
        tx(
            f"当前持仓：{labels.get(position, '—')}；"
            + (f"待成交方向：{labels.get(int(pending[0]), '—')}。" if isinstance(pending, (list, tuple)) and pending else "没有待成交动作。"),
            f"Position: {labels.get(position, '—')}; "
            + (f"pending fill: {labels.get(int(pending[0]), '—')}." if isinstance(pending, (list, tuple)) and pending else "no pending order."),
        )
    )


def render_v14_live_card(page) -> None:
    """侧栏探索口径的 v14 解释卡。按当前页面参数实时重算，不是冻结预设的封存记录。"""
    frame = page.result["frame"]
    if "z" not in frame.columns or "beta" not in frame.columns:
        return
    from ah_pairs.backtest import positions_with_reasons

    tradable = frame["both_open"].astype(bool).to_numpy() if "both_open" in frame.columns else None
    _target, reasons = positions_with_reasons(
        frame["z"].astype("float64").to_numpy(),
        frame["beta"].astype("float64").to_numpy(),
        page.params,
        tradable=tradable,
    )
    reason = reasons[-1]
    labels = position_labels()
    z = frame["z"].astype("float64").iloc[-1]
    position = int(frame["position"].iloc[-1]) if "position" in frame.columns else 0
    col1, col2, col3 = st.columns(3)
    col1.markdown(tx("**观察到了什么**", "**What was observed**"))
    col1.write(tx(f"z = {num(z, 2)}；入场 |z| ≥ {page.params.entry_z:.1f}，退出 |z| < {page.params.exit_z:.1f}，止损 |z| ≥ {page.params.stop_z:.1f}。", f"z = {num(z, 2)}; entry |z| ≥ {page.params.entry_z:.1f}, exit |z| < {page.params.exit_z:.1f}, stop |z| ≥ {page.params.stop_z:.1f}."))
    col2.markdown(tx("**规则给出什么**", "**What the rules say**"))
    col2.write(reason_label(reason))
    col3.markdown(tx("**账户实际是什么**", "**What the account holds**"))
    col3.write(tx(f"当前持仓：{labels.get(position, '—')}。信号事件和成交分开：信号先出，按 T+1 规则在下一次两边都开市的收盘成交。", f"Position: {labels.get(position, '—')}. Signals and fills are separate: the signal comes first and fills at the next close when both markets are open."))
