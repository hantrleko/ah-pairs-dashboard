"""A 股、H 股交易成本。

印花税按法定档期；佣金是券商参数，不是法定税率。
权重是净值比例，成本也是净值比例。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# A 股印花税（卖方/买方）。2008-09-19 起改为卖方单边。
_A_STAMP_SELLER_ONLY = pd.Timestamp("2008-09-19")
_A_STAMP_HALVED = pd.Timestamp("2023-08-28")
_A_STAMP_CUT_TO_10BP = pd.Timestamp("2008-04-24")
_A_STAMP_RAISED_TO_30BP = pd.Timestamp("2007-05-30")

# 港股股票交易印花税，买卖双方各付。
_H_STAMP_UP = pd.Timestamp("2021-08-01")
_H_STAMP_DOWN = pd.Timestamp("2023-11-17")


@dataclass
class CostParams:
    """佣金、融券费和可选的平坦印花税。"""

    a_commission: float = 0.00025
    h_commission: float = 0.0004
    a_borrow_annual: float = 0.05
    h_borrow_annual: float = 0.02
    use_statutory_stamp: bool = True
    a_stamp_sell: float = 0.0005
    h_stamp: float = 0.001
    slippage: float = 0.0
    trading_days: int = 252


def _as_day(day) -> pd.Timestamp:
    stamp = pd.Timestamp(day)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


def a_share_stamp(day, params: CostParams) -> tuple[float, float]:
    """返回 (买入印花税, 卖出印花税)。"""
    if not params.use_statutory_stamp:
        return 0.0, float(params.a_stamp_sell)
    d = _as_day(day)
    if d >= _A_STAMP_HALVED:
        return 0.0, 0.0005
    if d >= _A_STAMP_SELLER_ONLY:
        return 0.0, 0.001
    if d >= _A_STAMP_CUT_TO_10BP:
        return 0.001, 0.001
    if d >= _A_STAMP_RAISED_TO_30BP:
        return 0.003, 0.003
    return 0.001, 0.001


def h_share_stamp(day, params: CostParams) -> float:
    """港股单边印花税。买卖都付。"""
    if not params.use_statutory_stamp:
        return float(params.h_stamp)
    d = _as_day(day)
    if d >= _H_STAMP_DOWN:
        return 0.001
    if d >= _H_STAMP_UP:
        return 0.0013
    return 0.001


def leg_turnover_cost(
    prev_w: float,
    new_w: float,
    commission: float,
    stamp_buy: float,
    stamp_sell: float,
) -> float:
    """一条腿从旧权重调到新权重的成本，占净值的比例。

    正权重为多头，负权重为空头。跨越零轴时先平仓再开仓。
    """
    if prev_w == new_w:
        return 0.0
    if prev_w * new_w < 0:
        return leg_turnover_cost(
            prev_w, 0.0, commission, stamp_buy, stamp_sell
        ) + leg_turnover_cost(0.0, new_w, commission, stamp_buy, stamp_sell)
    if new_w > prev_w:
        traded = new_w - prev_w
        return traded * (commission + stamp_buy)
    traded = prev_w - new_w
    return traded * (commission + stamp_sell)


def rebalance_cost(
    prev_a: float,
    prev_h: float,
    new_a: float,
    new_h: float,
    day,
    params: CostParams,
) -> float:
    a_buy, a_sell = a_share_stamp(day, params)
    h_stamp = h_share_stamp(day, params)
    cost_a = leg_turnover_cost(prev_a, new_a, params.a_commission, a_buy, a_sell)
    cost_h = leg_turnover_cost(prev_h, new_h, params.h_commission, h_stamp, h_stamp)
    slip = params.slippage * (abs(new_a - prev_a) + abs(new_h - prev_h))
    return cost_a + cost_h + slip
