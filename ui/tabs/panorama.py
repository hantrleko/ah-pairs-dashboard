"""今日全景。打开时只读快照。"""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from ah_pairs.eflab_calendar import last_completed_session
from ah_pairs.i18n import tx
from ah_pairs.panorama import build_panorama, is_default_spec, load_panorama, write_panorama
from ah_pairs.universe import liquid_subset
from ui.format import follow_pair_selection, num, pct, show_table


def _view_key(page) -> str:
    payload = json.dumps(asdict(page.params), sort_keys=True, default=str)
    return f"{page.start_day}|{page.end_day}|{payload}"


def _labels() -> dict[int, str]:
    return {
        0: tx("空仓", "Flat"),
        1: tx("买A卖H", "Long A, short H"),
        -1: tx("卖A买H", "Short A, long H"),
    }


def _show(table: pd.DataFrame) -> None:
    labels = _labels()
    shown = pd.DataFrame(
        {
            "A": table["a_code"].astype(str),
            "H": table["h_code"].astype(str),
            tx("名称", "Name"): table["name"].astype(str),
            tx("截止", "As of"): table["asof"].astype(str),
            tx("溢价", "Premium"): table["premium"].map(pct),
            "z": table["z"].map(lambda value: num(value, 2)),
            tx("持仓", "Position"): table["position"].map(lambda value: labels.get(int(value), labels[0])),
            tx("信号", "Signal"): table["signal"].map(lambda value: labels.get(int(value), labels[0])),
            tx("待成交", "Pending"): table["pending"].map(
                lambda value: tx("是", "Yes") if bool(value) else tx("否", "No")
            ),
        }
    )
    event = show_table(shown, key="panorama_table", selectable=True)
    follow_pair_selection(event, shown)


def render_panorama(page) -> None:
    session_day = last_completed_session()
    st.caption(
        tx(
            f"日历上的最近已收盘日是 {session_day}。表上的截止日期是本地行情最后一根同时开市的 K 线，两边不一致时以表为准。这张表不是协整 p 值，也不替换单对回测。默认不做空 A。",
            f"The last completed session on the calendar is {session_day}. The table date is the last common session in the local file. This table is not a cointegration p-value and does not replace the pair backtest. Short A-shares stay off by default.",
        )
    )
    key = _view_key(page)
    saved = st.session_state.get("panorama_view")
    table = None
    meta = None
    source = ""
    if isinstance(saved, dict) and saved.get("key") == key and isinstance(saved.get("table"), pd.DataFrame):
        table = saved["table"]
        meta = saved.get("meta") or {}
        source = "session"
    else:
        loaded, file_meta, status = load_panorama(page)
        if status == "ok":
            table = loaded
            meta = file_meta or {}
            source = "file"
        elif status == "earlier_end":
            st.caption(tx("已有快照，但里面的行情晚于侧栏结束日，所以不显示。", "A snapshot exists, but its prices are after the sidebar end date, so it is hidden."))
        elif status == "spec":
            st.caption(tx("已有默认规格快照。侧栏已经改过，不用那张表冒充当前规格。", "A default-spec snapshot exists. The sidebar has changed, so that table is not shown as the current specification."))
        elif status == "window":
            st.caption(tx("已有快照，但起始日或结束日对不上，所以不显示。", "A snapshot exists, but the start or end date does not match, so it is hidden."))
        else:
            st.caption(tx("还没有全景表。", "There is no panorama yet."))
    if st.button(tx("用本地缓存重算", "Recompute from the local cache"), key="panorama_compute", use_container_width=True):
        with st.spinner(tx("只读本地行情…", "Reading local prices…")):
            fresh, fresh_meta = build_panorama(
                liquid_subset(page.universe),
                page.start_day,
                page.end_day,
                page.params,
            )
        wrote = False
        if is_default_spec(page.params, page.start_day):
            wrote = write_panorama(fresh, fresh_meta) is not None
        st.session_state["panorama_view"] = {
            "key": key,
            "table": fresh,
            "meta": fresh_meta,
            "saved": wrote,
        }
        st.rerun()
    if table is None or table.empty:
        if source == "session":
            skipped = (meta or {}).get("skipped") or []
            if skipped == ["FX"]:
                st.caption(tx("没有本地汇率，不算。", "No local FX file, so nothing was computed."))
            else:
                st.caption(tx("本地缓存里没有能拼起来的行情。", "The local cache has no pair that could be assembled."))
        return
    sample_end = (meta or {}).get("sample_end")
    skipped = (meta or {}).get("skipped") or []
    saved_now = source == "session" and isinstance(saved, dict) and saved.get("saved")
    if source == "file":
        st.caption(
            tx(
                f"这是默认规格的快照，样本截止 {sample_end}。结束日可以比这一天晚最多 10 天。改了参数就要重算，不能把这张表当成改过的规格。",
                f"This is the default-spec snapshot. The sample ends {sample_end}. The end date may be at most 10 days later. A changed specification has to be recomputed.",
            )
        )
    elif saved_now:
        st.caption(
            tx(
                f"已按默认规格写入快照，样本截止 {sample_end}。这次没有下载。",
                f"The default-spec snapshot was written. The sample ends {sample_end}. Nothing was downloaded.",
            )
        )
    else:
        st.caption(
            tx(
                f"这是按当前侧栏从本地缓存重算的，样本截止 {sample_end}。这次没有下载，也没有覆盖默认规格快照。",
                f"Recomputed from the local cache with the current sidebar. The sample ends {sample_end}. Nothing was downloaded, and the default snapshot was left as it was.",
            )
        )
    st.caption(
        tx(
            f"本地有行情 {len(table)} 只，跳过 {len(skipped)} 只。z 从低到高，越靠前 A 相对越便宜。持仓按次日收盘，信号当天还没成交时两列会不一样。",
            f"{len(table)} names had local prices, {len(skipped)} were skipped. z is sorted low to high, so a cheaper A-share is higher up. Positions fill at the next close, so the signal column can differ.",
        )
    )
    if skipped and skipped != ["FX"]:
        st.caption(tx("跳过：" + "、".join(str(code) for code in skipped), "Skipped: " + ", ".join(str(code) for code in skipped)))
    _show(table)
