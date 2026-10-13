"""命令行跑一对股票，方便不打开页面时核对结果。"""

from __future__ import annotations

import argparse

import pandas as pd

from ah_pairs.analysis import analyze_frame, warmup_start
from ah_pairs.backtest import StrategyParams
from ah_pairs.costs import CostParams
from ah_pairs.data import load_pair
from ah_pairs.northbound import load_northbound
from ah_pairs.universe import load_universe


def main() -> None:
    parser = argparse.ArgumentParser(description="跑一对 A+H 的协整与回测")
    parser.add_argument("--a", default="601318", help="A 股代码")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--hedge", default="rolling", choices=["rolling", "kalman"])
    parser.add_argument("--no-short-a", action="store_true")
    args = parser.parse_args()

    universe = load_universe()
    row = universe.loc[universe["a_code"] == str(args.a).zfill(6)]
    if row.empty:
        raise SystemExit(f"名单里没有 {args.a}")
    name = row.iloc[0]["name"]
    h_code = row.iloc[0]["h_code"]
    params = StrategyParams(
        hedge=args.hedge,
        allow_short_a=not args.no_short_a,
        costs=CostParams(),
    )
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    prices, warnings = load_pair(args.a, h_code, warmup_start(start, params), end)
    north = None
    try:
        north = load_northbound()
    except Exception as exc:
        warnings.append(f"北向资金没有载入：{exc}")
    result = analyze_frame(prices, params, eval_start=start, northbound=north)
    eg = result["eg"]
    hl = result["half_life_static"]
    stats = result["stats"]
    print(f"{name}  {args.a} / {h_code}")
    for note in warnings:
        print(" -", note)
    if hl["mean_reverting"]:
        half_text = f"半衰期={hl['half_life']:.1f} 日"
    else:
        half_text = "半衰期=样本内不收敛"
    print(
        f"EG beta={eg['beta']:.3f}  t={eg['t_stat']:.2f}  p={eg['pvalue']:.4g}  {half_text}"
    )
    print(
        f"收益={stats['total_return']:.2%}  夏普={stats['sharpe']:.2f}  "
        f"回撤={stats['max_drawdown']:.2%}  交易={stats['n_trades']}"
    )
    joh = result["johansen"]
    if joh.get("ok"):
        print(
            f"Johansen 迹统计={joh['trace_r0']:.2f}  5%临界值={joh['trace_crit_95']:.2f}  "
            f"拒绝无协整={joh['trace_reject_r0_95']}"
        )


if __name__ == "__main__":
    main()
