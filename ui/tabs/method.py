"""方法说明。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ah_pairs.i18n import tx


def render_method() -> None:
    st.markdown(
        tx(
            """
**价格。** A 股、H 股用复权收盘价。H 股乘以港元兑人民币。协整、β 和 z 只在两边都有成交的日子上估计。单边开市日把已有仓位按开市一边盯市，停市一边收益记 0，当天不新开仓、也不调 β。汇率最多向前填 5 个交易日。

**协整。** Engle–Granger 的原假设是「没有协整」。回归是 `log A = α + β log H + e`，滞后用 AIC。p 值是 MacKinnon 协整 p 值。残差 ADF 用了同一段残差，会偏乐观，不能代替 EG p 值。Johansen 迹检验含常数项。半衰期来自 AR(1)，只在 0 < φ < 1 时有限。

**交易和前视。** 全样本 β、EG 和 Johansen 用了评估区间内全部同时开市日，只能当描述。回测 β 是滚动 OLS 或卡尔曼滤波，写入 t 日的是 t 之前的状态。z 的滚动均值含当天价差，成交默认在下一个两边都开市的收盘。选「未考虑：当日收盘」会更乐观。

**卡尔曼 δ。** 可以按预热样本的一步预测误差似然，在 10⁻⁶ 到 10⁻² 的对数网格上选取。预热不够就沿用侧栏里的数，并在诊断页说明。用到评估样本本身的估计不是样本外。

**做空 A 股。** 默认关闭。打开之后，A 股偏贵时会卖 A，页顶会写明多数标的不一定能融到券。

**稳定性。** 地图的滞后固定为 1 阶，和摘要页 AIC 不是同一次检验。窗口按步长滑动并且互相重叠，检验次数不是独立样本数。BH 比例是 Benjamini–Hochberg 的参考，不是独立试验的 q 值。

**Walk-forward。** 样本跨过三个日历年并且足够长时，训练段逐年变长，下一年做验证。每一折同时选开仓和平仓阈值。样本不够就只切成前后两半。验证段不参与挑选。

**夏普。** 点估计是日收益均值除以标准差再乘 √252，无风险利率按 0。区间是区块自助法。PSR 是这组夏普大于 0 的概率。压缩夏普再把阈值搜索的次数算进去。它们都不是获利概率。

**多标的。** 页面上的四只和大约十二只是现算。全部股票池用 `python -m ah_pairs.scan --all` 写到缓存目录，页面只读那份 parquet。等风险组合用各对策略日收益的滚动波动倒数，权重滞后一天。

**多重比较。** 阈值网格、多折挑选和按 p 值给多只股票排序，都会把同一段行情试很多次。最好的那一格偏乐观。

**不要读成什么。** 样本内协整不保证以后还在。A/H 不能转换。北向是两市合计，日频止于 2024-08-16。南向和恒生 AH 溢价指数也是市场序列，不是这只股票的持仓，回归只描述同一天是否同向。
            """,
            """
**Prices.** Adjusted closes. The H-share is multiplied by the yuan price of one Hong Kong dollar. Cointegration, beta and z use only sessions when both markets traded. On a one-sided day an existing position is marked on the open leg, the closed leg earns zero, and there is no new trade or beta change. The FX rate is carried forward at most five sessions.

**Cointegration.** The Engle–Granger null is “no cointegration”. The regression is `log A = α + β log H + e`, with AIC lags. The p-value is MacKinnon’s cointegration p-value. A residual ADF uses residuals from that same sample, so it is optimistic and must not replace the EG p-value. The Johansen trace test includes a constant. The half-life is an AR(1) and is finite only when 0 < φ < 1.

**Trading and look-ahead.** Full-sample beta, EG and Johansen use every common session in the evaluation window. They are descriptive. The traded beta is rolling OLS or a Kalman filter; the value stored on day t was known before day t. The z-score’s mean includes today’s spread, and the default fill is the next close when both markets are open. Same-close fill is more optimistic.

**Kalman delta.** It can be chosen on a log grid from 1e-6 to 1e-2 by the Gaussian prediction-error likelihood of the warm-up sample. If that sample is short, the sidebar value is kept and the diagnostics say so. An estimate that uses the evaluation sample is not out of sample.

**Short A-shares.** Off by default. When it is on, a rich A-share is sold, and the page says that many names cannot be borrowed.

**Stability.** The map fixes the lag at 1. The summary EG test uses AIC. Those are different specifications. Windows overlap, so the number of tests is not an independent sample size. The BH rate is a Benjamini–Hochberg reference, not a q-value from independent trials.

**Walk-forward.** When the sample covers three calendar years and is long enough, training expands by year and the next year is the validation slice. Each fold picks both the entry and the exit threshold. Shorter samples use one half-split. Validation does not choose the thresholds.

**Sharpe.** The point estimate is the mean daily return over its standard deviation, times √252, with a zero risk-free rate. The interval is a block bootstrap. The PSR is the probability that this Sharpe exceeds zero. The deflated Sharpe also counts how many thresholds were searched. None of these is the probability of making money.

**Several pairs.** The four-name and twelve-name tables are computed on demand. The full universe is an offline job, `python -m ah_pairs.scan --all`, written into the cache directory. The page only reads that parquet. The equal-risk book uses inverse rolling volatility of each pair’s strategy return, lagged one day.

**Multiple testing.** Threshold grids, fold-by-fold selection and ranking many pairs on one sample all reuse the same prices. The best cell is biased upward.

**What this is not.** In-sample cointegration need not persist. A and H shares are not convertible. Northbound flow is the two-market total and the daily series stops on 2024-08-16. Southbound flow and the Hang Seng AH premium index are also market series, not this stock’s holding. The regressions only describe same-day association.
            """,
        )
    )
    changelog = Path(__file__).resolve().parents[2] / "CHANGELOG.md"
    if changelog.exists():
        with st.expander(tx("版本记录", "Changelog")):
            st.markdown(changelog.read_text(encoding="utf-8"))
