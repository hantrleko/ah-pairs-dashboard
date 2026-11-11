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

**Walk-forward。** 样本跨过三个日历年并且足够长时，训练段逐年变长，下一年做验证。每一折同时选开仓和平仓阈值。样本不够就只切成前后两半。验证段不参与挑选。各折验证段的日收益会接成一条样本外净值。默认规格已经预计算的，打开就能看到。规格不同时，这一步在回测页点按钮后才计算。

**夏普。** 年数是第一根和最后一根 K 线的日历跨度除以 365.25。年化因子是这段行数除以这个年数，不再固定乘 √252。只在同时开市日上重算的夏普单独列出。无风险利率按 0。区间用同一个年化因子做区块自助法。PSR 是未年化夏普大于 0 的概率。压缩夏普再把开仓阈值的搜索次数算进去。它们都不是获利概率。融券费仍按一年 252 天计提。

**成本。** 佣金、印花税、滑点和融券费分开。方向变化那天记入开平仓，方向不变、只改 β 的记入再平衡。回测页可以另下一行 H 股现金红利的 20% 预扣：只扣前一日已经持有 H 股多头的日子，按当日汇率换成人民币。没有分红序列时这一行留空。主回测仍用复权价。

**V1.5。** 默认引擎仍是 v14。侧栏切到 v15 后，页顶改成净溢价 z、样本外净收益、样本外夏普、样本外回撤、Reality Check 的 p 值和半衰期。样本外折要在回测页点按钮才算，打开页面不算，也不在每日预计算里。训练 756 个同时开市日，测试 126 日，中间隔 121 日，只用测试段之前的数据选开仓、平仓和估计窗口。折起点重算一次 Engle-Granger，半衰期按日更新；不是每个参数、每一天都重算 EG。没有股息日历时，楔子不算，页面会写明。成交成本按 100 万人民币名义本金换成收益率，最低佣金不会按零点几元去扣。H 股直接账户红利税率和港交所征费合计仍标成未核实。RC p 不是获利概率。样本外不足 4 折时，统计结论不可靠。v15 不改 v14 的数字。

**预计算。** 工作日收盘后的定时任务刷新行情、H 股分红、北向、南向、恒生 AH 溢价指数、融券标的名单和流动性扫描，写入 `data/published/`。页面先读 `data/cache`，没有文件时读已发布副本，打开页面不下载。四只快捷股票在默认规格下的 walk-forward 和稳定性地图放在 `data/desk/`。规格一致、结束日不早于样本截止日并且比该日晚不超过 10 天时打开就能看到，并写明这一天。改了对冲、窗口、开平仓、成本或起始日之后，仍要点按钮重算。预计算结果不会冒充另一套规格。

**链接。** 点「应用参数」后，地址栏写入股票、语言、日期、对冲、两个窗口、开仓和平仓阈值、成交时点、是否做空，以及成本是法定档还是零。打开链接就用这套规格。佣金小数、卡尔曼 δ 和再平衡带不在地址里。写错的字段会被丢掉，侧栏标出代号。

**多标的。** 页面上的四只和大约十二只是现算。流动性子集的离线摘要放在 `data/scan_snapshot/`，表上写明生成日、样本区间、不做空、法定成本和次日收盘。扫描日早于行情缓存时，页面会标明。全部股票池仍用 `python -m ah_pairs.scan --all`，不要在页面里跑。等风险组合用各对策略日收益的滚动波动倒数，权重滞后一天。p 值更小不是「最协整」的发现。

**多重比较。** 阈值网格、多折挑选和按 p 值给多只股票排序，都会把同一段行情试很多次。最好的那一格偏乐观。

**相对市场。** 诊断页把个股溢价减去恒生 AH 溢价指数。指数 100 当作平价，减的是指数/100 − 1，系数固定为 1。接着做的是普通 ADF，原假设是相对溢价有单位根。它不替换 EG p 值。指数还没有缓存时这一项不算。

**结构断点。** Gregory–Hansen 允许 log A 和 log H 的关系变一次。模型 C 只改截距，模型 C/T 加趋势，模型 C/S 连斜率一起改。临界值用 Gregory and Hansen (1996) 表 1 的 ADF*，一个回归元；模型 C 的 5% 是 −4.61。两端各去掉 15%，各断点用同一个 Schwert 滞后。点按钮才算。拒绝不替换 EG p 值。

**纯多头。** 回测页另给三行：永远持 A、永远持 H、按溢价 z 切换。z < 0 持 A，z ≥ 0 持 H，不做空。成交时点和主回测一样。这是已经要持有这家公司时的比较，不替换配对回测。

**月度截面。** 一组对照页点按钮后，用本地缓存给流动性子集按溢价 z 排序。每月做多 z 最低四分之一的 A 股，并做空它们的 H 股。不做空 A。没有缓存的股票跳过，页面不下载。

**融券名单。** 更新按钮下载沪深融资融券标的和保证金比例。那不是融券年化费率，回测费率仍用侧栏。默认不做空，名单不改数字。打开做空之后，只有名单覆盖到的日期才按名单限制做空；更早的日子仍按开关。

**今日全景。** 第一页列出流动性子集的最新溢价、z 和持仓。打开时只读定时任务写下的表。点按钮才按当前侧栏从本地缓存重算，不下载，也不跑协整。结束日早于样本截止日，或规格和默认规格不同时，不拿这张表冒充。点一行打开那一对。默认不做空 A。

**提醒订阅。** 信号页可以保存邮件地址或 webhook。地址写在本地缓存，不进 git。定时任务仍会向环境变量 ALERT_WEBHOOK_URL 发送。邮件只在设置了 ALERT_SMTP_HOST 和 ALERT_SMTP_FROM 时发送；没设置就只保存，不假装已经发出。

**日期口径。** 行情日期是最近一根已收盘 K 线。沪深 15:00、港股 16:00，16:10 以后的工作日才算当天，周末往前推。法定假日不在这张表里，表上的日期仍以行情文件为准。入口是 eugenefinance.tech。公开地址只包括本站和 Fin-Analysis。理财站点还没有公开地址，不编链接。

**不要读成什么。** 样本内协整不保证以后还在。A/H 不能转换。北向是两市合计，日频止于 2024-08-16。南向和恒生 AH 溢价指数也是市场序列，不是这只股票的持仓，回归只描述同一天是否同向。
            """,
            """
**Prices.** Adjusted closes. Yahoo reinvests dividends; the East Money fallback is a forward adjustment. Stock Connect holders of H shares pay dividend tax, so the adjusted close is not an after-tax tradable price. The stability page can download unadjusted closes and rerun the check. The H-share is multiplied by the yuan price of one Hong Kong dollar. Cointegration, beta and z use only sessions when both markets traded. On a one-sided day an existing position is marked on the open leg, the closed leg earns zero, and there is no new trade or beta change. The FX rate is carried forward at most five sessions.

**Cointegration.** The Engle–Granger null is “no cointegration”. The regression is `log A = α + β log H + e`, with AIC lags. The p-value is MacKinnon’s cointegration p-value. A residual ADF uses residuals from that same sample, so it is optimistic and must not replace the EG p-value. The Johansen trace test includes a constant. The half-life is an AR(1) and is finite only when 0 < φ < 1.

**Trading and look-ahead.** Full-sample beta, EG and Johansen use every common session in the evaluation window. They are descriptive. The traded beta is rolling OLS or a Kalman filter; the value stored on day t was known before day t. The z-score’s mean includes today’s spread, and the default fill is the next close when both markets are open. Same-close fill is more optimistic.

**Kalman delta.** It can be chosen on a log grid from 1e-6 to 1e-2 by the Gaussian prediction-error likelihood of the warm-up sample. If that sample is short, the sidebar value is kept and the diagnostics say so. An estimate that uses the evaluation sample is not out of sample.

**Short A-shares.** Off by default. When it is on, a rich A-share is sold, and the page says that many names cannot be borrowed.

**Stability.** The map fixes the lag at 1. The summary EG test uses AIC. Those are different specifications. Windows overlap, so the number of tests is not an independent sample size. The BH rate is a Benjamini–Hochberg reference, not a q-value from independent trials.

**Walk-forward.** When the sample covers three calendar years and is long enough, training expands by year and the next year is the validation slice. Each fold picks both the entry and the exit threshold. Shorter samples use one half-split. Validation does not choose the thresholds. The validation returns are stitched into one out-of-sample equity curve. A precomputed default specification is shown on open. Any other specification runs when you press the button on the backtest page.

**Sharpe.** Years are the calendar span from the first bar to the last, divided by 365.25. The annualization factor is the number of rows divided by those years, not a fixed √252. A second Sharpe uses common sessions only. The risk-free rate is zero. The interval is a block bootstrap with the same factor. The PSR is the probability that the non-annualized Sharpe exceeds zero. The deflated Sharpe also counts how many entry thresholds were searched. None of these is the probability of making money. Borrow still accrues on a 252-day year.

**Costs.** Commission, stamp, slippage and borrow are shown separately. A day that changes direction is an open or a close. A day that keeps the direction and only moves beta is a rebalance. The backtest page can add one row that withholds 20% of H-share cash dividends, only on days the book was already long H, converted at that day's exchange rate. The row stays blank when there is no dividend series. The main backtest still uses adjusted closes.

**V1.5.** The default engine is still v14. Switching the sidebar to v15 replaces the headline with net-premium z, out-of-sample net, out-of-sample Sharpe, out-of-sample drawdown, the reality-check p-value and the half-life. The out-of-sample folds run only from the button on the backtest page. They are not part of the daily precompute. Training is 756 common sessions, the test is 126, and 121 sessions are left out in between. Entry, exit and the estimation window are chosen only on data before the test. Engle–Granger is recomputed once at the start of a fold; the half-life updates daily. A missing dividend calendar means the wedge is not applied, and the page says so. Trading costs are converted to a return on a one-million-yuan book, so the cash minimum commission is not charged on a fraction of one yuan. The H-share direct-account dividend tax and the HKEX levy total remain unverified. The RC p-value is not the probability of profit. Fewer than four out-of-sample folds is not a reliable claim. v15 does not change the v14 numbers.

**Precompute.** After the close on weekdays, a scheduled job refreshes prices, H-share dividends, northbound, southbound, the Hang Seng AH premium index, the margin-eligible list and the liquid scan, and writes them to `data/published/`. The page reads `data/cache` first and the published copy when that file is missing. Opening the page does not download. Walk-forward and the stability map for the four shortcut names, at the default specification, live in `data/desk/`. They appear on open when the specification matches, the end date is no earlier than the sample end, and it is at most 10 days later. The page names that sample end. A different hedge, window, entry, exit, cost or start date still needs the button. A precomputed result is not presented as a different specification.

**Link.** Apply writes the pair, language, dates, hedge, both windows, entry and exit, fill, whether A-shares can be shorted, and whether costs are the statutory schedule or zero. Opening that link uses this specification. Commission decimals, the Kalman delta and the rebalance band stay out of the address. Invalid fields are dropped and the sidebar names their codes.

**Several pairs.** The four-name and twelve-name tables are computed on demand. An offline summary of the liquid subset lives in `data/scan_snapshot/`, with the generation date, sample window, short-A off, statutory costs and next-close fill. If that date is older than the price cache, the page says so. The full universe remains `python -m ah_pairs.scan --all` and does not run in the page. The equal-risk book uses inverse rolling volatility of each pair’s strategy return, lagged one day. A smaller p-value is not a discovery of the most cointegrated name.

**Multiple testing.** Threshold grids, fold-by-fold selection and ranking many pairs on one sample all reuse the same prices. The best cell is biased upward.

**Versus the market.** Diagnostics subtract the Hang Seng AH premium index from the stock premium. A reading of 100 is parity, so the index enters as level/100 − 1 and the coefficient is fixed at 1. The next step is an ordinary ADF whose null is a unit root in that relative premium. It does not replace the EG p-value. If the index is not cached, this block is skipped.

**One break.** Gregory–Hansen lets the relation between log A and log H change once. Model C shifts the intercept, model C/T adds a trend, and model C/S also shifts the slope. Critical values are the ADF* column of Gregory and Hansen (1996), Table 1, for one regressor. Model C’s 5% value is −4.61. Each end drops 15%, and every break uses the same Schwert lag. It runs from the button. A rejection does not replace the EG p-value.

**Long only.** The backtest page adds three rows: always hold A, always hold H, and switch on the premium z. Hold A when z < 0 and H when z ≥ 0. Nothing is shorted. The fill is the same as the main backtest. This is the comparison for someone who would hold the company anyway. It does not replace the pairs backtest.

**Monthly cross-section.** The button on the pairs page ranks the liquid subset by premium z, using only local cache. Each month it buys the cheapest quartile of A-shares and shorts those names’ H-shares. A-shares are not shorted. Names without a cache file are skipped. The page does not download them.

**Borrow list.** The update button downloads the margin-eligible list and the margin ratio. That ratio is not the annual borrow fee; the backtest still uses the sidebar fee. Shorting stays off by default, so the list does not change the figures. After shorting is turned on, only dates covered by the list can block a short. Earlier dates still follow the switch.

**Today.** The first page lists the latest premium, z and position for the liquid subset. Opening it only reads the table written by the scheduled job. The button recomputes from the local cache with the current sidebar. It does not download and it does not run a cointegration test. An earlier end date, or a specification other than the default, does not reuse that table. Clicking a row opens the pair. Short A-shares stay off by default.

**Subscriptions.** The signal page can save an email address or a webhook. The address stays in the local cache and is not committed. The scheduled job still posts to ALERT_WEBHOOK_URL. Mail is sent only when ALERT_SMTP_HOST and ALERT_SMTP_FROM are set. Otherwise the address is saved and nothing is sent.

**Dates.** A market date is the last completed bar. Shanghai closes at 15:00 and Hong Kong at 16:00. A weekday counts only after 16:10. Weekends step back to Friday. Exchange holidays are not in this calendar, so a table still uses the last bar in the price file. The lab entry is eugenefinance.tech. The public list is this site and Fin-Analysis. The planning site has no public address, so it is not given a made-up link.

**What this is not.** In-sample cointegration need not persist. A and H shares are not convertible. Northbound flow is the two-market total and the daily series stops on 2024-08-16. Southbound flow and the Hang Seng AH premium index are also market series, not this stock’s holding. The regressions only describe same-day association.
            """,
        )
    )
    changelog = Path(__file__).resolve().parents[2] / "CHANGELOG.md"
    if changelog.exists():
        with st.expander(tx("版本记录", "Changelog")):
            st.markdown(changelog.read_text(encoding="utf-8"))
