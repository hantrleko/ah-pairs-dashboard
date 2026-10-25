"""信号页。"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from ah_pairs.alerts import alert_payload, send_webhook
from ah_pairs.backtest import current_state
from ah_pairs.export import spread_png
from ah_pairs.i18n import tx
from ah_pairs.plots import core_signal_figure, hedge_figure, premium_figure, prices_figure
from ui.format import kpis, meta, num, show_chart


def render_signal(page) -> None:
    state = current_state(page.frame, page.result["trades"], page.params.fill)
    names = {
        0: tx("空仓", "Flat"),
        1: tx("多头价差（买 A 卖 H）", "Long spread (long A, short H)"),
        -1: tx("空头价差（卖 A 买 H）", "Short spread (short A, long H)"),
    }
    held_name = names.get(state["position"], names[0])
    signal_name = names.get(state["signal"], names[0])
    if page.params.fill != "next_close":
        pending_text = tx("未按 T+1：信号已按当日收盘计入持仓。", "T+1 is off: the signal is filled at the same close.")
    elif state["pending"]:
        pending_text = tx(
            f"有待成交：收盘信号是{signal_name}，持仓仍是{held_name}。下一个两边都开市的收盘才成交。",
            f"Pending: the close signal is {signal_name}, the position is still {held_name}. It fills at the next close when both markets are open.",
        )
    else:
        pending_text = tx("没有待成交动作。收盘信号和当前持仓一致。", "Nothing is pending. The close signal matches the position.")
    entry_text = "—" if state["last_entry"] is None else str(state["last_entry"].date())
    exit_text = "—" if state["last_exit"] is None else str(state["last_exit"].date())
    kpis(
        [
            (tx("最新 z", "Latest z"), num(state["z"], 2), ""),
            (tx("当前持仓", "Position"), held_name, ""),
            (tx("最近开仓", "Last entry"), entry_text, ""),
            (tx("最近平仓", "Last exit"), exit_text, ""),
        ],
        columns=4,
    )
    st.caption(
        tx(
            f"截止 {pd.Timestamp(state['asof']).date()}，与回测同一套已应用的参数。{pending_text}",
            f"As of {pd.Timestamp(state['asof']).date()}, same applied parameters as the backtest. {pending_text}",
        )
    )
    show_chart(
        core_signal_figure(page.view, page.params.z_window, page.params.entry_z, page.params.exit_z, page.params.stop_z, show_marks=page.show_marks),
        "chart_core",
    )
    show_chart(prices_figure(page.view), "chart_prices")
    left, right = st.columns(2)
    with left:
        show_chart(premium_figure(page.view), "chart_premium")
    with right:
        show_chart(hedge_figure(page.view, active=page.params.hedge), "chart_hedge_signal")
    st.caption(meta(tx("溢价和 β", "Premium and beta"), page.view.index, page.cache_stamp, tx("β 只用当天以前的价格。", "Beta uses only prices before that day.")))
    spread_bytes = spread_png(page.frame, page.params.z_window)
    if spread_bytes:
        st.download_button(tx("下载价差图 PNG", "Download spread PNG"), spread_bytes, file_name=f"{page.picked.a_code}_spread.png", mime="image/png", key="spread_png_signal")
    payload = alert_payload(page.picked["name"], page.picked.a_code, page.picked.h_code, state, page.params.entry_z)
    st.markdown(tx("**提醒**", "**Alert**"))
    if payload is None:
        st.caption(tx(f"最新 |z| 没有越过开仓阈值 {page.params.entry_z:.1f}。定时任务也不会发这条。", f"Latest |z| is inside the entry threshold {page.params.entry_z:.1f}. A scheduled job would not send this."))
    else:
        st.caption(tx("最新 |z| 越过开仓阈值。下面是 webhook 报文，不是下单指令。", "Latest |z| is outside the entry threshold. The payload below is a webhook, not an order."))
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        st.code(text, language="json")
        url = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
        if url and st.button(tx("发送这条提醒", "Send this alert"), key="send_alert"):
            ok, note = send_webhook(url, payload)
            st.write(note if ok else tx(f"没有发出去：{note}", f"Not sent: {note}"))
        elif not url:
            st.caption(tx("环境变量 ALERT_WEBHOOK_URL 为空，所以这里只展示报文。定时任务见仓库里的 alert workflow。", "ALERT_WEBHOOK_URL is empty, so this only shows the payload. The scheduled job is the alert workflow in the repo."))
