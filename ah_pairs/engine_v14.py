"""V1.4 回测入口。

计算仍在 ah_pairs.backtest。V1.5 不改这些函数。把 BACKTEST_VERSION 设回 v14 就走这里。
"""

from __future__ import annotations

from ah_pairs.backtest import StrategyParams, run_backtest

ENGINE = "v14"


def run(frame, params: StrategyParams):
    return run_backtest(frame, params)
