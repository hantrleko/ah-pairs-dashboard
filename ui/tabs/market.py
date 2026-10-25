"""北向、南向和恒生 AH 溢价指数。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ah_pairs.freshness import northbound_stops_before
from ah_pairs.i18n import tx
from ah_pairs.market import describe_market
from ah_pairs.northbound import latest_valid
from ah_pairs.plots import northbound_figure
from ui.cache import cached_ah_index, cached_southbound
from ui.format import meta, show_chart, show_table, span_frame


def _series(loader, error_key: str):
    if st.session_state.get(error_key):
        return None, st.session_state[error_key]
    try:
        return loader(), ""
    except Exception as exc:
        st.session_state[error_key] = str(exc)
        return None, str(exc)


def _block(page, series, error: str, column: str, kind: str, source: str, note: str, error_key: str = "") -> None:
    if error:
        st.info(error)
        if error_key and st.button(tx("再试一次", "Try again"), key=f"retry_{column}"):
            st.session_state.pop(error_key, None)
            st.rerun()
        return
    if series is None or series.dropna().empty:
        st.info(tx("这条序列是空的。", "This series is empty."))
        return
    common = page.common.copy()
    common[column] = series.reindex(common.index)
    usable = common.dropna(subset=["spread", column])
    view = span_frame(usable, page.chart_span)
    end = latest_valid(series)
    end_text = "—" if end is None else str(pd.Timestamp(end).date())
    stale = northbound_stops_before(page.last_px, end) if column == "northbound" else False
    if stale:
        st.markdown(
            tx(
                f'<p class="health warn">北向是两市合计，不是 {page.picked["name"]} 的持仓。共同有数的日期止于 {end_text}，行情到 {pd.Timestamp(page.last_px).date()}。</p>',
                f'<p class="health warn">Northbound flow is the two-market total, not the holding in {page.picked["name"]}. Overlap ends {end_text}; prices run to {pd.Timestamp(page.last_px).date()}.</p>',
            ),
            unsafe_allow_html=True,
        )
    st.caption(tx(f"序列截至 {end_text}。{note}", f"Series through {end_text}. {note}"))
    if len(view) == 0:
        st.warning(tx("当前图表区间里没有重叠数据。可以改成全期间。", "No overlap in the current chart span. Try the full sample."))
        return
    st.caption(meta(source, view.index, page.cache_stamp))
    show_chart(northbound_figure(view, column=column, kind=kind), f"chart_{column}")
    factor = usable[column].diff() if column == "hsahp" else usable[column]
    flow = describe_market(usable["spread"], factor)
    if not flow.get("ok"):
        st.info(flow.get("message") or tx("重叠不够，没有做回归。", "Not enough overlap for a regression."))
        return
    show_table(
        pd.DataFrame(
            [
                {
                    tx("样本天数", "Days"): flow["nobs"],
                    tx("起始", "Start"): pd.Timestamp(flow["start"]).date(),
                    tx("结束", "End"): pd.Timestamp(flow["end"]).date(),
                    tx("系数", "Coefficient"): flow["beta"],
                    tx("标准误", "SE"): flow["se"],
                    "t": flow["t_stat"],
                    "p": flow["pvalue"],
                    "R²": flow["r2"],
                }
            ]
        )
    )
    st.caption(tx("描述性回归，HAC 标准误，滞后 5 日。不是这条序列造成价差的证据。系数是同一天的关系。", "A descriptive regression with HAC standard errors, 5 lags. It is not evidence that this series causes the spread. The coefficient is same-day."))


def render_market(page) -> None:
    st.markdown(
        tx(
            "三列都是市场层面的序列，不是这只股票自己的持仓。北向日频已经停止。南向和恒生沪深港通 AH 股溢价指数（HSAHP）用来接上停更之后的对照。指数高于 100 表示样本内 A 股平均更贵。",
            "All three series are market-wide. None is this stock’s own holding. Daily northbound disclosure has stopped. Southbound flow and the Hang Seng Stock Connect China AH Premium Index (HSAHP) continue the comparison. An index above 100 means A-shares are richer than H-shares on average.",
        )
    )
    south, south_error = _series(cached_southbound, "south_failed")
    ah, ah_error = _series(cached_ah_index, "ah_failed")
    st.markdown(tx("**北向资金**", "**Northbound**"))
    _block(page, page.north, page.north_error, "northbound", "bar", tx("东财沪深港通北向合计", "East Money northbound total"), tx("单位亿元，两市合计。", "CNY 100 million, both mainland markets."))
    st.markdown(tx("**南向资金**", "**Southbound**"))
    _block(page, south, south_error, "southbound", "bar", tx("东财沪深港通南向合计", "East Money southbound total"), tx("单位亿元，港股通合计，不是这只 H 股的持仓。", "CNY 100 million, Stock Connect total, not this H-share’s holding."), "south_failed")
    st.markdown(tx("**恒生 AH 溢价指数**", "**Hang Seng AH premium index**"))
    _block(page, ah, ah_error, "hsahp", "line", "HSAHP", tx("回归用的是指数的日变化，不是指数水平。", "The regression uses the daily change in the index, not its level."), "ah_failed")
