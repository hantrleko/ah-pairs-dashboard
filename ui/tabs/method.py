"""方法说明。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ah_pairs.i18n import tx


def render_method() -> None:
    st.markdown(
        tx(
            """
**价格。** A 股、H 股用复权收盘价。Yahoo 这条含分红再投资，东财回退用前复权。港股通持有 H 股要缴红利税，复权价不是税后可交易价格。稳定性页可以另下一份未复权收盘，看协整结论变不变。H 股乘以港元兑人民币。协整、β 和 z 只在两边都有成交的日子上估计。单边开市日把已有仓位按开市一边盯市，停市一边收益记 0，当天不新开仓、也不调 β。汇率最多向前填 5 个交易日。

**协整。** Engle–Granger 的原假设是「没有协整」。回归是 `log A = α + β log H + e`，滞后用 AIC。p 值是 MacKinnon 协整 p 值。残差 ADF 用了同一段残差，会偏乐观，不能代替 EG p 值。Johansen 迹检验含常数项。半衰期来自 AR(1)，只在 0 < φ < 1 时有限。

**交易和前视。** 全样本 β、EG 和 Johansen 用了评估区间内全部同时开市日，只能当描述。回测 β 是滚动 OLS 或卡尔曼滤波，写入 t 日的是 t 之前的状态。z 的滚动均值含当天价差，成交默认在下一个两边都开市的收盘。选「未考虑：当日收盘」会更乐观。

**卡尔曼 δ。** 可以按预热样本的一步预测误差似然，在 10⁻⁶ 到 10⁻² 的对数网格上选取。预热不够就沿用侧栏里的数，并在诊断页说明。用到评估样本本身的估计不是样本外。

**做空 A 股。** 默认关闭。打开之后，A 股偏贵时会卖 A，页顶会写明多数标的不一定能融到券。

**稳定性。** 地图的滞后固定为 1 阶，和摘要页 AIC 不是同一次检验。窗口按步长滑动并且互相重叠，检验次数不是独立样本数。BH 比例是 Benjamini–Hochberg 的参考，不是独立试验的 q 值。

**Walk-forward。** 样本跨过三个日历年并且足够长时，训练段逐年变长，下一年做验证。每一折同时选开仓和平仓阈值。样本不够就只切成前后两半。验证段不参与挑选。各折验证段的日收益会接成一条样本外净值。这一步在回测页点按钮后才计算。

**夏普。** 年数是第一根和最后一根 K 线的日历跨度除以 365.25。年化因子是这段行数除以这个年数，不再固定乘 √252。只在同时开市日上重算的夏普单独列出。无风险利率按 0。区间用同一个年化因子做区块自助法。PSR 是未年化夏普大于 0 的概率。压缩夏普再把开仓阈值的搜索次数算进去。它们都不是获利概率。融券费仍按一年 252 天计提。

**成本。** 佣金、印花税、滑点和融券费分开。方向变化那天记入开平仓，方向不变、只改 β 的记入再平衡。回测页可以另下一行 H 股现金红利的 20% 预扣：只扣前一日已经持有 H 股多头的日子，按当日汇率换成人民币。没有分红序列时这一行留空。主回测仍用复权价。

**链接。** 点「应用参数」后，地址栏写入股票、语言、日期、对冲、两个窗口、开仓和平仓阈值、成交时点、是否做空，以及成本是法定档还是零。打开链接就用这套规格。佣金小数、卡尔曼 δ 和再平衡带不在地址里。写错的字段会被丢掉，侧栏标出代号。

**多标的。** 页面上的四只和大约十二只是现算。流动性子集的离线摘要放在 `data/scan_snapshot/`，表上写明生成日、样本区间、不做空、法定成本和次日收盘。扫描日早于行情缓存时，页面会标明。全部股票池仍用 `python -m ah_pairs.scan --all`，不要在页面里跑。等风险组合用各对策略日收益的滚动波动倒数，权重滞后一天。p 值更小不是「最协整」的发现。

**多重比较。** 阈值网格、多折挑选和按 p 值给多只股票排序，都会把同一段行情试很多次。最好的那一格偏乐观。

**不要读成什么。** 样本内协整不保证以后还在。A/H 不能转换。北向是两市合计，日频止于 2024-08-16。南向和恒生 AH 溢价指数也是市场序列，不是这只股票的持仓，回归只描述同一天是否同向。
            """,
            """
**Prices.** Adjusted closes. Yahoo reinvests dividends; the East Money fallback is a forward adjustment. Stock Connect holders of H shares pay dividend tax, so the adjusted close is not an after-tax tradable price. The stability page can download unadjusted closes and rerun the check. The H-share is multiplied by the yuan price of one Hong Kong dollar. Cointegration, beta and z use only sessions when both markets traded. On a one-sided day an existing position is marked on the open leg, the closed leg earns zero, and there is no new trade or beta change. The FX rate is carried forward at most five sessions.

**Cointegration.** The Engle–Granger null is “no cointegration”. The regression is `log A = α + β log H + e`, with AIC lags. The p-value is MacKinnon’s cointegration p-value. A residual ADF uses residuals from that same sample, so it is optimistic and must not replace the EG p-value. The Johansen trace test includes a constant. The half-life is an AR(1) and is finite only when 0 < φ < 1.

**Trading and look-ahead.** Full-sample beta, EG and Johansen use every common session in the evaluation window. They are descriptive. The traded beta is rolling OLS or a Kalman filter; the value stored on day t was known before day t. The z-score’s mean includes today’s spread, and the default fill is the next close when both markets are open. Same-close fill is more optimistic.

**Kalman delta.** It can be chosen on a log grid from 1e-6 to 1e-2 by the Gaussian prediction-error likelihood of the warm-up sample. If that sample is short, the sidebar value is kept and the diagnostics say so. An estimate that uses the evaluation sample is not out of sample.

**Short A-shares.** Off by default. When it is on, a rich A-share is sold, and the page says that many names cannot be borrowed.

**Stability.** The map fixes the lag at 1. The summary EG test uses AIC. Those are different specifications. Windows overlap, so the number of tests is not an independent sample size. The BH rate is a Benjamini–Hochberg reference, not a q-value from independent trials.

**Walk-forward.** When the sample covers three calendar years and is long enough, training expands by year and the next year is the validation slice. Each fold picks both the entry and the exit threshold. Shorter samples use one half-split. Validation does not choose the thresholds. The validation returns are stitched into one out-of-sample equity curve. The backtest page computes this when you press the button.

**Sharpe.** Years are the calendar span from the first bar to the last, divided by 365.25. The annualization factor is the number of rows divided by those years, not a fixed √252. A second Sharpe uses common sessions only. The risk-free rate is zero. The interval is a block bootstrap with the same factor. The PSR is the probability that the non-annualized Sharpe exceeds zero. The deflated Sharpe also counts how many entry thresholds were searched. None of these is the probability of making money. Borrow still accrues on a 252-day year.

**Costs.** Commission, stamp, slippage and borrow are shown separately. A day that changes direction is an open or a close. A day that keeps the direction and only moves beta is a rebalance. The backtest page can add one row that withholds 20% of H-share cash dividends, only on days the book was already long H, converted at that day's exchange rate. The row stays blank when there is no dividend series. The main backtest still uses adjusted closes.

**Link.** Apply writes the pair, language, dates, hedge, both windows, entry and exit, fill, whether A-shares can be shorted, and whether costs are the statutory schedule or zero. Opening that link uses this specification. Commission decimals, the Kalman delta and the rebalance band stay out of the address. Invalid fields are dropped and the sidebar names their codes.

**Several pairs.** The four-name and twelve-name tables are computed on demand. An offline summary of the liquid subset lives in `data/scan_snapshot/`, with the generation date, sample window, short-A off, statutory costs and next-close fill. If that date is older than the price cache, the page says so. The full universe remains `python -m ah_pairs.scan --all` and does not run in the page. The equal-risk book uses inverse rolling volatility of each pair’s strategy return, lagged one day. A smaller p-value is not a discovery of the most cointegrated name.

**Multiple testing.** Threshold grids, fold-by-fold selection and ranking many pairs on one sample all reuse the same prices. The best cell is biased upward.

**What this is not.** In-sample cointegration need not persist. A and H shares are not convertible. Northbound flow is the two-market total and the daily series stops on 2024-08-16. Southbound flow and the Hang Seng AH premium index are also market series, not this stock’s holding. The regressions only describe same-day association.
            """,
        )
    )
    changelog = Path(__file__).resolve().parents[2] / "CHANGELOG.md"
    if changelog.exists():
        with st.expander(tx("版本记录", "Changelog")):
            st.markdown(changelog.read_text(encoding="utf-8"))
