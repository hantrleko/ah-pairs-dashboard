"""刷新缓存、已发布副本，以及四只快捷股票的默认规格预计算。

页面打开时不要调用。工作日收盘后由 GitHub Actions 运行。
`--codes` 只限制行情、分红和预计算，不会把离线扫描和今日全景收成少数几只。
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import pandas as pd

from ah_pairs.backtest import StrategyParams, run_backtest
from ah_pairs.costs import CostParams
from ah_pairs.data import _cache_file, dividend_path, load_h_dividends, load_pair
from ah_pairs.desk import DESK_A_CODES, default_desk_params, write_result
from ah_pairs.paths import CACHE_DIR, PUBLISHED_DIR
from ah_pairs.store import publish_file
from ah_pairs.universe import a_to_yahoo, h_to_yahoo, liquid_subset, load_universe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="刷新行情、分红、市场序列、离线扫描，并预计算默认规格")
    parser.add_argument("--codes", nargs="*", help="只刷新这些 A 股代码的行情、分红和预计算。扫描仍用全部流动性子集。")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-dividends", action="store_true")
    parser.add_argument("--skip-market", action="store_true")
    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--skip-desk", action="store_true")
    parser.add_argument("--skip-borrow", action="store_true")
    parser.add_argument("--skip-panorama", action="store_true")
    args = parser.parse_args(argv)
    codes = [str(code).zfill(6) for code in args.codes] if args.codes else None
    errors: list[str] = []
    wrote = False

    if not args.skip_market:
        wrote = _market(errors) or wrote
    universe = _universe(codes)
    if not args.skip_prices:
        wrote = _prices(universe, args.start, args.end, errors) or wrote
    if not args.skip_dividends:
        wrote = _dividends(universe, errors) or wrote
    if not args.skip_scan:
        wrote = _scan(args.start, args.end, errors) or wrote
    if not args.skip_desk:
        wrote = _desk(codes, args.start, args.end, errors) or wrote
    if not args.skip_borrow:
        wrote = _borrow(errors) or wrote
    if not args.skip_panorama:
        wrote = _panorama(args.start, args.end, errors) or wrote

    if errors:
        print(f"有 {len(errors)} 项没有完成。", flush=True)
        for item in errors:
            print(f"- {item}", flush=True)
    if errors and not wrote:
        return 1
    print("刷新结束。", flush=True)
    return 0


def _universe(codes: list[str] | None) -> pd.DataFrame:
    universe = liquid_subset(load_universe())
    if codes:
        universe = universe[universe["a_code"].isin(codes)].reset_index(drop=True)
    return universe


def _market(errors: list[str]) -> bool:
    from ah_pairs.market import load_ah_premium_index, load_southbound
    from ah_pairs.northbound import load_northbound

    wrote = False
    jobs = (
        ("northbound.parquet", lambda: load_northbound(refresh=True)),
        ("southbound.parquet", lambda: load_southbound(refresh=True)),
        ("hsahp.parquet", lambda: load_ah_premium_index(refresh=True)),
    )
    for name, fetch in jobs:
        try:
            series = fetch()
            if series is None or series.dropna().empty:
                raise RuntimeError("下载结果是空的")
            src = CACHE_DIR / name
            wrote = publish_file(src, PUBLISHED_DIR / name) or wrote
            end = pd.Timestamp(series.dropna().index.max()).date()
            print(f"{name} 截至 {end}，{len(series)} 行。", flush=True)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            print(f"{name} 失败：{exc}", flush=True)
    return wrote


def _prices(universe: pd.DataFrame, start: str, end: str, errors: list[str]) -> bool:
    wrote = False
    if universe.empty:
        errors.append("行情：没有可刷新的股票")
        return False
    for row in universe.itertuples(index=False):
        try:
            load_pair(row.a_code, row.h_code, "2010-01-01", end, refresh=True)
            for ticker in (a_to_yahoo(row.a_code), h_to_yahoo(row.h_code), "HKDCNY=X"):
                src = _cache_file(CACHE_DIR, ticker)
                wrote = publish_file(src, _cache_file(PUBLISHED_DIR / "prices", ticker)) or wrote
            print(f"行情 {row.a_code} 已发布。", flush=True)
        except Exception as exc:
            errors.append(f"行情 {row.a_code}: {exc}")
            print(f"行情 {row.a_code} 失败：{exc}", flush=True)
    return wrote


def _dividends(universe: pd.DataFrame, errors: list[str]) -> bool:
    wrote = False
    seen: set[str] = set()
    for row in universe.itertuples(index=False):
        h_code = str(row.h_code).zfill(5)
        if h_code in seen:
            continue
        seen.add(h_code)
        try:
            series = load_h_dividends(h_code, refresh=True)
            if series is None or series.empty:
                print(f"分红 {h_code} 没有现金红利记录。", flush=True)
                continue
            ticker = h_to_yahoo(h_code)
            src = dividend_path(CACHE_DIR, ticker)
            wrote = publish_file(src, dividend_path(PUBLISHED_DIR / "dividends", ticker)) or wrote
            print(f"分红 {h_code} {len(series)} 条。", flush=True)
        except Exception as exc:
            errors.append(f"分红 {h_code}: {exc}")
            print(f"分红 {h_code} 失败：{exc}", flush=True)
    return wrote


def _borrow(errors: list[str]) -> bool:
    from ah_pairs.borrow import load_borrow

    try:
        frame = load_borrow(refresh=True)
        day = pd.Timestamp(frame["date"].max()).date()
        print(f"融券名单 {day}，{frame['a_code'].nunique()} 只。", flush=True)
        return True
    except Exception as exc:
        errors.append(f"融券名单: {exc}")
        print(f"融券名单失败：{exc}", flush=True)
        return False


def _scan(start: str, end: str, errors: list[str]) -> bool:
    from ah_pairs.scan import scan_pairs, with_scan_meta, write_scan

    os.environ["AH_PUBLISH_SCAN"] = "1"
    try:
        universe = liquid_subset(load_universe())
        params = StrategyParams(allow_short_a=False, costs=CostParams())
        print(f"开始扫描 {len(universe)} 只，{start} 至 {end}。", flush=True)
        summary, returns = scan_pairs(universe, start, end, params)
        summary = with_scan_meta(summary, start, end, allow_short_a=False)
        write_scan(summary, returns)
        print(f"扫描写出 {len(summary)} 行。", flush=True)
        return not summary.empty
    except Exception as exc:
        errors.append(f"扫描: {exc}")
        print(f"扫描失败：{exc}", flush=True)
        traceback.print_exc()
        return False


def _desk(requested: list[str] | None, start: str, end: str, errors: list[str]) -> bool:
    from ah_pairs.analysis import analyze_frame, walk_forward, warmup_start
    from ah_pairs.stability import slice_checks, stability_grid

    wanted = requested if requested is not None else list(DESK_A_CODES)
    codes = [code for code in wanted if code in DESK_A_CODES]
    if not codes:
        print("这次没有要预计算的快捷股票。", flush=True)
        return False
    names = load_universe().set_index("a_code")
    params = default_desk_params()
    wrote = False
    for code in codes:
        if code not in names.index:
            errors.append(f"预计算 {code}: 股票池里没有")
            continue
        h_raw = names.loc[code, "h_code"]
        if isinstance(h_raw, pd.Series):
            h_raw = h_raw.iloc[0]
        h_code = str(h_raw).zfill(5)
        try:
            print(f"开始预计算 {code}。", flush=True)
            warm = warmup_start(start, params).strftime("%Y-%m-%d")
            prices, _notes = load_pair(code, h_code, warm, end, refresh=False)
            prepared = analyze_frame(
                prices,
                params,
                eval_start=start,
                full_diagnostics=False,
                alt_hedge=False,
                with_backtest=False,
            )
            frame, _trades, _stats = run_backtest(prepared["frame"], params)
            walk = walk_forward(frame, params)
            if not walk.get("ok"):
                raise RuntimeError(walk.get("message") or "walk-forward 没有结果")
            common = frame.loc[frame["both_open"]] if "both_open" in frame.columns else frame
            grid = stability_grid(common["log_a"], common["log_h"], common.index)
            checks = slice_checks(common)
            sample_end = common.index.max()
            write_result(code, params, start, end, sample_end, walk, grid, checks, include_costs=True)
            print(f"预计算 {code} 完成，样本到 {pd.Timestamp(sample_end).date()}。", flush=True)
            wrote = True
        except Exception as exc:
            errors.append(f"预计算 {code}: {exc}")
            print(f"预计算 {code} 失败：{exc}", flush=True)
            traceback.print_exc()
    return wrote


def _panorama(start: str, end: str, errors: list[str]) -> bool:
    from ah_pairs.desk import default_desk_params
    from ah_pairs.panorama import build_panorama, write_panorama

    try:
        universe = liquid_subset(load_universe())
        table, meta = build_panorama(universe, start, end, default_desk_params())
        path = write_panorama(table, meta)
        skipped = meta.get("skipped") or []
        print(f"全景 {len(table)} 行，跳过 {len(skipped)}。", flush=True)
        if path is None:
            errors.append("全景: 本地没有可用行情")
            return False
        return True
    except Exception as exc:
        errors.append(f"全景: {exc}")
        print(f"全景失败：{exc}", flush=True)
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(main())
