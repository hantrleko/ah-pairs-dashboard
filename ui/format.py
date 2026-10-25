"""数字、表格和图表的共用写法。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ah_pairs.analysis import pick_best_sharpe, pick_lowest_p, screen_one
from ah_pairs.i18n import tx


def pct(value) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.2%}"


def num(value, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def pval(value) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def tone(value) -> str:
    if value is None or not np.isfinite(value) or value == 0:
        return ""
    return "neg" if value < 0 else "pos"


def kpis(items: list[tuple[str, str, str]], columns: int = 6) -> None:
    kind = {4: "kpi-row four", 5: "kpi-row five"}.get(columns, "kpi-row")
    cells = "".join(
        f'<div class="kpi {item_tone}"><span>{label}</span><strong>{value}</strong></div>'
        for label, value, item_tone in items
    )
    st.markdown(f'<div class="{kind}">{cells}</div>', unsafe_allow_html=True)


def show_chart(fig, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": True})


def show_table(frame, column_config=None) -> None:
    st.dataframe(frame, use_container_width=True, hide_index=True, column_config=column_config)


def span_frame(data: pd.DataFrame, span: str) -> pd.DataFrame:
    if data is None or len(data) == 0:
        return data
    focus = st.session_state.get("chart_focus")
    if isinstance(focus, dict) and focus.get("start") and focus.get("end"):
        start = pd.Timestamp(focus["start"])
        end = pd.Timestamp(focus["end"])
        sliced = data.loc[(data.index >= start) & (data.index <= end)]
        if len(sliced):
            return sliced
    if span == "all":
        return data
    days = 365 if span == "1y" else 365 * 3
    end = data.index.max()
    return data.loc[data.index >= end - pd.Timedelta(days=days)]


def meta(source: str, index, cache_stamp: str, extra: str = "") -> str:
    if index is None or len(index) == 0:
        return tx("这段图上没有数据。", "This chart has no data.")
    start = pd.Timestamp(index.min()).date()
    end = pd.Timestamp(index.max()).date()
    return tx(
        f"来源：{source}。图上 {start} 至 {end}，最后日期 {end}。行情缓存写入 {cache_stamp}。{extra}",
        f"Source: {source}. Plotted {start} to {end}. Cache written {cache_stamp}. {extra}",
    )


def half_life_text(result: dict | None) -> str:
    if not result:
        return "—"
    if result.get("mean_reverting") and np.isfinite(result.get("half_life", np.nan)):
        return num(result["half_life"], 1)
    return tx("不收敛", "not mean-reverting")


def rank_lines(table: pd.DataFrame) -> None:
    best_p = pick_lowest_p(table)
    best_s = pick_best_sharpe(table)
    if best_p is not None:
        st.markdown(
            tx(
                f"协整诊断较好：**{best_p['公司']}**，EG p = {pval(best_p['eg_pvalue'])}。",
                f"Closer to cointegration: **{best_p['公司']}**, EG p = {pval(best_p['eg_pvalue'])}.",
            )
        )
    if best_s is not None:
        st.markdown(
            tx(
                f"扣费后回测表现较好：**{best_s['公司']}**，夏普 {num(best_s['sharpe'], 2)}，净收益 {pct(best_s.get('total_return', np.nan))}。",
                f"Better after costs: **{best_s['公司']}**, Sharpe {num(best_s['sharpe'], 2)}, net {pct(best_s.get('total_return', np.nan))}.",
            )
        )
    if best_p is not None and best_s is not None and best_p["公司"] != best_s["公司"]:
        st.caption(
            tx(
                "这两类排名不是同一只。p 值低不代表扣完费用后的策略更好。",
                "These two rankings need not name the same firm. A smaller p-value is not a better after-cost strategy.",
            )
        )


def compare_key(page) -> str:
    params = page.params
    return "|".join(
        [
            str(page.start_day),
            str(page.end_day),
            params.hedge,
            str(params.hedge_window),
            str(params.z_window),
            f"{params.entry_z:.1f}",
            f"{params.exit_z:.1f}",
            params.fill,
            "1" if params.allow_short_a else "0",
            "1" if params.kalman_delta_auto else "0",
        ]
    )


QUICK = (("601318", "平安", "Ping An"), ("601398", "工行", "ICBC"), ("600036", "招行", "CMB"), ("601857", "石油", "PetroChina"))


def render_quick_compare(page, button_key: str) -> None:
    from ui.cache import cached_pair

    st.markdown(tx("**平安、工行、招行、石油**", "**Ping An, ICBC, CMB, PetroChina**"))
    st.caption(
        tx(
            "用已经应用的日期和参数现算，已经扣了侧栏里的成本。p 值更小只表示更接近协整，不表示扣费后更好。",
            "Uses the applied dates and parameters, after the sidebar costs. A smaller p-value means closer to cointegration, not a better after-cost result.",
        )
    )
    if st.button(tx("按当前区间比较这四只", "Compare these four"), key=button_key):
        rows = []
        progress = st.progress(0.0, text=tx("准备比较", "Preparing"))
        for i, (code, short_zh, short_en) in enumerate(QUICK):
            short_name = tx(short_zh, short_en)
            progress.progress(i / len(QUICK), text=short_name)
            match = page.universe.loc[page.universe["a_code"] == code]
            if match.empty:
                rows.append({"简称": short_name, "公司": code, "eg_pvalue": np.nan, "说明": tx("名单里没有", "Not in the universe")})
                continue
            item = match.iloc[0]
            try:
                pair_prices, _notes = cached_pair(item.a_code, item.h_code, page.warm, str(page.end_day))
                summary = screen_one(pair_prices, page.params, pd.Timestamp(page.start_day))
                rows.append({"简称": short_name, "公司": item["name"], "A": item.a_code, "H": item.h_code, **summary, "说明": ""})
            except Exception as exc:
                rows.append({"简称": short_name, "公司": item["name"], "A": item.a_code, "H": item.h_code, "eg_pvalue": np.nan, "说明": str(exc)})
        progress.empty()
        st.session_state["quick_compare"] = {"key": compare_key(page), "table": pd.DataFrame(rows)}
    saved = st.session_state.get("quick_compare")
    if not isinstance(saved, dict) or not isinstance(saved.get("table"), pd.DataFrame):
        return
    if saved.get("key") != compare_key(page):
        st.caption(tx("日期或参数已经变了，需要再比一次。", "Dates or parameters changed. Compare again."))
        return
    table = saved["table"]
    rank_lines(table)
    columns = [col for col in ["简称", "公司", "A", "H", "eg_pvalue", "total_return", "max_drawdown", "n_trades", "total_cost", "sharpe", "说明"] if col in table.columns]
    show_table(
        table[columns],
        {
            "eg_pvalue": st.column_config.NumberColumn(tx("EG p 值", "EG p-value"), format="%.3f"),
            "total_return": st.column_config.NumberColumn(tx("净收益", "Net return"), format="%.2%"),
            "max_drawdown": st.column_config.NumberColumn(tx("最大回撤", "Max drawdown"), format="%.2%"),
            "n_trades": st.column_config.NumberColumn(tx("交易次数", "Trades"), format="%d"),
            "total_cost": st.column_config.NumberColumn(tx("成本合计", "Costs"), format="%.2%"),
            "sharpe": st.column_config.NumberColumn(tx("夏普", "Sharpe"), format="%.2f"),
        },
    )
