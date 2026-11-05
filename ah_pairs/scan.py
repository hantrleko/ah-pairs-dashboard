"""离线扫描股票池，写成 parquet。页面只读结果，不在打开时扫完全部公司。

默认只扫流动性名单。`--all` 才扫 ah_universe.csv 里的全部公司，需要网络，时间很长。
"""

from __future__ import annotations

import argparse

import pandas as pd

from ah_pairs.analysis import analyze_frame, warmup_start
from ah_pairs.backtest import StrategyParams
from ah_pairs.costs import CostParams
from ah_pairs.data import load_pair
from ah_pairs.paths import SCAN_DIR
from ah_pairs.portfolio import portfolio_summary
from ah_pairs.universe import liquid_subset, load_universe

SCAN_SPEC_DEFAULT = "short=0;cost=statutory;fill=next_close"


def scan_pairs(
    universe: pd.DataFrame,
    start,
    end,
    params: StrategyParams | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = params or StrategyParams(allow_short_a=False, costs=CostParams())
    rows = []
    returns = {}
    eval_start = pd.Timestamp(start)
    warm = warmup_start(eval_start, params)
    for item in universe.itertuples(index=False):
        label = f"{item.name} {item.a_code}"
        try:
            prices, _notes = load_pair(item.a_code, item.h_code, warm, end)
            result = analyze_frame(prices, params, eval_start=eval_start, full_diagnostics=False)
            eg = result["eg"]
            stats = result["stats"]
            hl = result["half_life_static"]
            premium = result["frame"]["premium"].dropna()
            summary = {
                "nobs": eg["nobs"],
                "eg_pvalue": eg["pvalue"],
                "eg_beta": eg["beta"],
                "eg_t": eg["t_stat"],
                "half_life": hl["half_life"],
                "sharpe": stats["sharpe"],
                "cagr": stats["cagr"],
                "total_return": stats["total_return"],
                "max_drawdown": stats["max_drawdown"],
                "n_trades": stats["n_trades"],
                "total_cost": stats["total_cost"],
                "latest_premium": float(premium.iloc[-1]) if len(premium) else float("nan"),
            }
            rows.append({"公司": item.name, "A": item.a_code, "H": item.h_code, **summary, "说明": ""})
            returns[label] = result["frame"]["ret_strat"]
            print(f"完成 {label}  EG p={summary['eg_pvalue']:.3f}")
        except Exception as exc:
            rows.append({"公司": item.name, "A": item.a_code, "H": item.h_code, "说明": str(exc)})
            print(f"跳过 {label}：{exc}")
    summary_frame = pd.DataFrame(rows)
    return_frame = pd.DataFrame(returns).sort_index()
    return summary_frame, return_frame


def scan_spec(allow_short_a: bool = False) -> str:
    short = "1" if allow_short_a else "0"
    return f"short={short};cost=statutory;fill=next_close"


def with_scan_meta(summary: pd.DataFrame, start, end, asof=None, allow_short_a: bool = False) -> pd.DataFrame:
    """给摘要加上生成日、样本区间和规格。页面用这三列说明这不是当日排名。"""
    out = summary.copy()
    out["asof"] = str(pd.Timestamp(asof).date()) if asof is not None else pd.Timestamp.today().strftime("%Y-%m-%d")
    out["start"] = str(pd.Timestamp(start).date())
    out["end"] = str(pd.Timestamp(end).date())
    out["spec"] = scan_spec(allow_short_a)
    return out


def scan_is_older(asof, cache_stamp) -> bool:
    """扫描生成日早于行情缓存的写入日。无法解析时不当作过期。"""
    try:
        scanned = pd.Timestamp(str(asof)[:10]).normalize()
        cached = pd.Timestamp(str(cache_stamp)[:10]).normalize()
    except (TypeError, ValueError):
        return False
    if pd.isna(scanned) or pd.isna(cached):
        return False
    return bool(scanned < cached)


def write_scan(summary: pd.DataFrame, returns: pd.DataFrame, folder=SCAN_DIR) -> None:
    import os

    from ah_pairs.paths import DATA_DIR

    folder.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(folder / "summary.parquet", index=False)
    returns.to_parquet(folder / "returns.parquet")
    if os.environ.get("AH_PUBLISH_SCAN") == "1":
        published = DATA_DIR / "scan_snapshot"
        published.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(published / "summary.parquet", index=False)
        returns.to_parquet(published / "returns.parquet")


def _scan_folders(folder=None):
    from ah_pairs.paths import DATA_DIR

    if folder is not None:
        return [folder]
    return [SCAN_DIR, DATA_DIR / "scan_snapshot"]


def read_scan(folder=None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """先看缓存目录，再看仓库里的 data/scan_snapshot。两边都有时用较新的一份。"""
    found = []
    for candidate in _scan_folders(folder):
        summary_path = candidate / "summary.parquet"
        if summary_path.exists():
            found.append((summary_path.stat().st_mtime, candidate))
    if not found:
        return None, None
    chosen = max(found)[1]
    summary = pd.read_parquet(chosen / "summary.parquet")
    returns_path = chosen / "returns.parquet"
    returns = pd.read_parquet(returns_path) if returns_path.exists() else None
    return summary, returns


def main() -> None:
    parser = argparse.ArgumentParser(description="离线扫描 A+H 股票池")
    parser.add_argument("--all", action="store_true", help="扫描名单里的全部公司")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--allow-short-a", action="store_true")
    args = parser.parse_args()
    universe = load_universe()
    if not args.all:
        universe = liquid_subset(universe)
    params = StrategyParams(allow_short_a=bool(args.allow_short_a), costs=CostParams())
    summary, returns = scan_pairs(universe, args.start, args.end, params)
    summary = with_scan_meta(summary, args.start, args.end, allow_short_a=bool(args.allow_short_a))
    write_scan(summary, returns)
    portfolio = portfolio_summary(returns) if not returns.empty else {"ok": False}
    print(f"写出 {SCAN_DIR / 'summary.parquet'} ，{len(summary)} 行。")
    if portfolio.get("ok"):
        print(
            f"等风险组合 {portfolio['n_pairs']} 对，净收益 {portfolio['total_return']:.2%}，"
            f"夏普 {portfolio['sharpe']:.2f}。"
        )
    else:
        print(portfolio.get("message", "等风险组合没有算出。"))


if __name__ == "__main__":
    main()
