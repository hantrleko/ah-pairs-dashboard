"""前瞻记录。每只股票 × 每个预设的封存时间线，只读。

记录两个时间：行情截止日和系统实际观察时间。补回放行单独标记，
不进段内净值；缺口事件永久留痕。“按今天数据重算”只作对照，
不回写账本，两条曲线不拼接。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ah_pairs.forward import (
    FORWARD_DIR,
    annotate_segments,
    build_pair_frame,
    read_rows,
    read_state,
    rows_path,
    state_path,
    v14_daily_rows,
    v15_daily_rows,
)
from ah_pairs.i18n import tx
from ah_pairs.presets import PRESETS, get_preset
from ah_pairs.universe import load_universe
from ui.explain import reason_label, render_ledger_card
from ui.format import num


def _ledgers_index() -> dict:
    """{(preset_id, a_code): n_rows}，只扫文件名，不读内容。"""
    out = {}
    ledgers = FORWARD_DIR / "ledgers"
    if not ledgers.exists():
        return out
    for preset_dir in sorted(ledgers.iterdir()):
        if not preset_dir.is_dir():
            continue
        for path in sorted(preset_dir.glob("*.jsonl")):
            try:
                with open(path, encoding="utf-8") as handle:
                    count = sum(1 for _ in handle)
            except OSError:
                count = 0
            out[(preset_dir.name, path.stem)] = count
    return out


def _names() -> dict:
    try:
        universe = load_universe()
    except Exception:
        return {}
    return {str(item["a_code"]).zfill(6): str(item.get("name") or item["a_code"]) for item in universe.to_dict(orient="records")}


def _position_word(value: int) -> str:
    return {0: tx("空仓", "Flat"), 1: tx("买A卖H", "Long A/H-"), -1: tx("卖A买H", "Short A/H+")}.get(int(value or 0), "—")


def _timeline(rows: list[dict]) -> None:
    annotated = annotate_segments(rows)
    frame = pd.DataFrame(annotated)
    for column in ("z", "signal", "position", "reason", "net", "equity", "seg_equity", "n_fills", "observed_at"):
        if column not in frame.columns:
            frame[column] = None
    frame["backfilled"] = frame.get("backfilled", False)
    frame["backfilled"] = frame["backfilled"].fillna(False).astype(bool)

    live = frame[frame["seg_equity"].notna()]
    if len(live) >= 2:
        chart = live.set_index(pd.to_datetime(live["date"]))["seg_equity"].astype(float)
        st.line_chart(chart, height=180)
        segs = sorted({int(x) for x in live["seg_id"] if pd.notna(x)})
        if len(segs) > 1:
            st.caption(
                tx(
                    f"净值分 {len(segs)} 段：每段是连续实时观察的区间，从 1 重新归一。段间隔着补回放，不能拼成一条曲线。",
                    f"Equity in {len(segs)} segments: each is a continuous real-time window renormalized to 1. Segments are separated by backfilled rows and cannot be stitched.",
                )
            )
        else:
            st.caption(tx("段内净值从 1 起步，只含实时观察的行。", "Segment equity starts at 1 and contains only real-time rows."))
    elif len(frame) and int(frame["backfilled"].sum()) == len(frame):
        st.caption(tx("目前全是补回放行，还没有实时前瞻净值。", "All rows so far are backfilled; there is no real-time forward equity yet."))

    shown = frame.iloc[::-1].head(60)
    table = pd.DataFrame(
        {
            tx("日期", "Date"): shown["date"].astype(str),
            tx("观察时间", "Observed"): shown["observed_at"].astype(str).str.slice(0, 16),
            "z": shown["z"].map(lambda v: num(v, 2)),
            tx("信号", "Signal"): shown["signal"].map(lambda v: _position_word(int(v or 0))),
            tx("持仓", "Position"): shown["position"].map(lambda v: _position_word(int(v or 0))),
            tx("原因", "Reason"): shown["reason"].map(lambda r: reason_label(r or "")),
            tx("成交", "Fills"): shown["n_fills"].map(lambda v: int(v or 0)),
            tx("净收益", "Net"): shown["net"].map(lambda v: "" if v is None or pd.isna(v) else f"{float(v) * 100:.3f}%"),
            tx("段内净值", "Seg. equity"): shown["seg_equity"].map(lambda v: "" if v is None or pd.isna(v) else num(v, 4)),
            tx("补回放", "Backfilled"): shown["backfilled"].map(lambda v: tx("是", "Yes") if v else ""),
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def _gaps_section(state: dict) -> None:
    gaps = state.get("gaps") or []
    if not gaps:
        return
    with st.expander(tx(f"缺口事件（{len(gaps)}）", f"Gap events ({len(gaps)})")):
        for gap in reversed(gaps[-10:]):
            st.caption(
                tx(
                    f"{gap.get('from', '?')} → {gap.get('to', '?')}：{int(gap.get('rows') or 0)} 行补回放，其中按规则会发生 {int(gap.get('fills') or 0)} 笔成交；系统当时未实时观察，记录于 {str(gap.get('observed_at', ''))[:16]}。",
                    f"{gap.get('from', '?')} → {gap.get('to', '?')}: {int(gap.get('rows') or 0)} backfilled rows with {int(gap.get('fills') or 0)} rule-implied fills; not observed in real time, recorded {str(gap.get('observed_at', ''))[:16]}.",
                )
            )


def _recompute_compare(preset_id: str, a_code: str, h_code: str, rows: list[dict], state: dict) -> None:
    preset = get_preset(preset_id)
    inception = state.get("inception")
    frame, problem = build_pair_frame(a_code, h_code)
    if frame is None or frame.empty:
        st.warning(tx(f"重算不了：{problem}", f"Cannot recompute: {problem}"))
        return
    if preset.engine == "v14":
        new_rows = v14_daily_rows(frame, preset, inception=inception)
    else:
        locks = (state.get("v15") or {}).get("locks")
        new_rows, _live = v15_daily_rows(frame, preset, locks, inception=inception)
    sealed = {row["date"]: row for row in rows}
    diffs = []
    checked = 0
    for row in new_rows:
        old = sealed.get(row["date"])
        if old is None:
            continue
        checked += 1
        issues = []
        if int(old.get("position") or 0) != int(row.get("position") or 0):
            issues.append(tx("持仓不同", "position differs"))
        old_net = old.get("net")
        new_net = row.get("net")
        if old_net is not None and new_net is not None and abs(float(old_net) - float(new_net)) > 1e-10:
            issues.append(tx("净收益不同", "net differs"))
        if issues:
            diffs.append({"date": row["date"], "issues": "、".join(issues), "sealed_pos": old.get("position"), "now_pos": row.get("position")})
    if not diffs:
        st.success(
            tx(
                f"对照了 {checked} 个共同日期：持仓和净收益完全一致。这说明封存记录与当前数据口径一致。",
                f"Compared {checked} common dates: positions and net returns all match. The sealed records are consistent with current data.",
            )
        )
    else:
        st.warning(
            tx(
                f"{len(diffs)} 个日期不一致（共对照 {checked} 天）。通常是上游更正了历史行情；封存行不改，差异只作对照。",
                f"{len(diffs)} dates differ (out of {checked}). Usually upstream revised history; sealed rows stay, differences are for comparison only.",
            )
        )
        st.dataframe(pd.DataFrame(diffs), use_container_width=True, hide_index=True)




def _attribution_section(preset_id: str, a_code: str, rows: list[dict], state: dict) -> None:
    """账本逐笔收益拆解。v14 按预设重算权重；v15 用 ±0.5 等金额两腿，借券费标未建模。"""
    from ah_pairs.attribution import (
        attribute_frame,
        attribute_window,
        frame_from_v15_ledger,
        ledger_episodes,
        reconcile_daily,
        v14_attribution_frame,
    )
    from ah_pairs.presets import get_preset as _get

    episodes = ledger_episodes(rows)
    if not episodes:
        st.caption(tx("这条账本还没有开过仓，没有可拆解的交易。", "This ledger has not opened a trade yet; nothing to attribute."))
        return
    frame, problem = build_pair_frame(a_code, str(rows[-1].get("h_code") or ""))
    if frame is None or frame.empty:
        st.caption(tx(f"行情不齐，拆不了：{problem}", f"Prices incomplete: {problem}"))
        return
    preset = _get(preset_id)
    inception = state.get("inception")
    if preset.engine == "v14":
        attr_frame = v14_attribution_frame(frame, preset, inception=inception)
        borrow_note = ""
    else:
        attr_frame = frame_from_v15_ledger(rows, frame)
        borrow_note = tx(" v15 的借券费没有建模，这一栏是未计入，不是真实零成本。", " v15 borrow cost is not modelled; that column is unaccounted, not a true zero.")
    attr = attribute_frame(attr_frame)
    if not reconcile_daily(attr):
        st.warning(tx("每日归因与账本净收益对不上，下面数字不可信。", "Daily attribution does not reconcile with the ledger; numbers below are unreliable."))
        return
    labels = []
    for ep in episodes:
        side = tx("做多价差", "Long") if int(ep["direction"]) == 1 else tx("做空价差", "Short")
        end = ep["exit_date"] or tx("持仓中", "open")
        labels.append(f"{ep['entry_date']} → {end} · {side}")
    pick = st.selectbox(tx("选择一笔交易", "Pick a trade"), range(len(labels)), format_func=lambda i: labels[i], key="fwd_attr_pick")
    ep = episodes[int(pick)]
    end_date = ep["exit_date"] or rows[-1]["date"]
    out = attribute_window(attr, ep["entry_date"], end_date)
    if not out.get("ok"):
        st.caption(tx(f"不可归因：{out.get('reason')}", f"Not attributable: {out.get('reason')}"))
        return
    table = pd.DataFrame(
        {
            tx("来源", "Source"): [tx("A 股价格", "A price"), tx("H 股本币价格", "H local price"), tx("汇率", "FX"), tx("交易费用", "Fees"), tx("借券费用", "Borrow"), tx("合计", "Total")],
            tx("贡献", "Contribution"): [f"{out['a_price'] * 100:.3f}%", f"{out['h_price'] * 100:.3f}%", f"{out['fx'] * 100:.3f}%", f"{out['cost'] * 100:.3f}%", f"{out['borrow'] * 100:.3f}%", f"{out['total'] * 100:.3f}%"],
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    note = tx("对账一致：各项加总等于该窗口的复合净收益。", "Reconciled: the parts sum to the compounded return of the window.") if out.get("reconciled") else tx("对账不一致。", "Not reconciled.")
    if not ep["closed"]:
        note += tx(" 未平仓部分是浮动盈亏，估值截止到账本最后一行。", " The open part is unrealized, valued at the last ledger row.")
    note += borrow_note
    if preset.engine == "v14":
        note += tx(" 若上游修订过行情，重算权重反映的是当前数据。", " If upstream revised prices, the recomputed weights reflect current data.")
    st.caption(note)

def render_forward(page) -> None:
    st.caption(
        tx(
            "前瞻账本从上线日空仓起步，逐日只追加，不因重跑改写。每行同时记录行情截止日和系统实际观察时间，不倒签。",
            "The forward ledger starts flat on launch day, appends only, and is never rewritten by reruns. Each row carries both the data cutoff and the actual observation time — never backdated.",
        )
    )
    index = _ledgers_index()
    if not index:
        st.info(
            tx(
                "尚未启账。正式启用后，这里按股票、引擎和预设展示封存时间线；本地演练记录不作为正式前瞻。",
                "Not live yet. Once enabled, this page shows the sealed timeline per pair, engine and preset; local rehearsal records do not count as official forward history.",
            )
        )
        return
    names = _names()
    keys = sorted(index.keys())
    labels = [f"{names.get(a, a)} · {preset}" for preset, a in keys]
    pick = st.selectbox(tx("账本", "Ledger"), range(len(keys)), format_func=lambda i: labels[i], key="forward_ledger_pick")
    preset_id, a_code = keys[int(pick)]
    rows = read_rows(rows_path(FORWARD_DIR, preset_id, a_code))
    state = read_state(state_path(FORWARD_DIR, preset_id, a_code)) or {}
    if not rows:
        st.info(tx("这条账本是空的。", "This ledger is empty."))
        return
    preset = get_preset(preset_id)
    h_code = str(rows[-1].get("h_code") or "")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tx("上线日", "Inception"), str(state.get("inception") or "—"))
    c2.metric(tx("行情截止", "Data as of"), str(state.get("data_asof") or "—"))
    c3.metric(tx("当前持仓", "Position"), _position_word(state.get("position") or 0))
    c4.metric(tx("记录行数", "Rows"), str(len(rows)))
    st.caption(
        tx(
            f"预设版本 v{preset.version}，生效 {preset.effective}。最近实际观察 {str(state.get('last_observed_at', '—'))[:16]}。",
            f"Preset v{preset.version}, effective {preset.effective}. Last observed {str(state.get('last_observed_at', '—'))[:16]}.",
        )
    )
    render_ledger_card(rows[-1])
    st.divider()
    _timeline(rows)
    _gaps_section(state)
    with st.expander(tx("逐笔收益拆解", "Per-trade attribution")):
        _attribution_section(preset_id, a_code, rows, state)
    if st.button(tx("按今天数据重算对照", "Recompute with current data"), key="forward_recompute"):
        with st.spinner(tx("只读本地缓存重算…", "Recomputing from the local cache…")):
            _recompute_compare(preset_id, a_code, h_code, rows, state)
        st.caption(tx("对照不写账本；当时记录与重算结果不能拼成一条曲线。", "The comparison does not write to the ledger; sealed records and recomputed results cannot be stitched into one curve."))
