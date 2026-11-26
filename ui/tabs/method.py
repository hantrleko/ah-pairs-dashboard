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

**持仓权重。** 同一方向持仓期间，名义权重 $w_a$ 和 $w_h$ 在回测里固定不变，等于每个交易日都把仓位调回目标比例，而且这部分权重漂移不计入佣金与印花税。真实头寸会随价格变化。净值与夏普是在这一约定下算出来的。

**溢价水平。** 主图与溢价 z 用复权收盘。A 股与 H 股除息日、分红金额不同，历史溢价的水平和可交易 AH 溢价会有偏差；最近一天用于信号时受影响较小。对照未复权收盘请看稳定性页。

**Walk-forward 中位数。** 某一折验证段若一笔交易也没有，该折夏普记为 NaN 并在汇总中丢掉，不算作 0，中位数因此略偏乐观。每一折样本外回测从空仓开始，跨年持仓不会在折之间接续。

**卡尔曼 δ。** 可以按预热样本的一步预测误差似然，在 10⁻⁶ 到 10⁻² 的对数网格上选取。预热不够就沿用侧栏里的数，并在诊断页说明。用到评估样本本身的估计不是样本外。

**做空 A 股。** 默认关闭。打开之后，A 股偏贵时会卖 A，页顶会写明多数标的不一定能融到券。

**稳定性。** 地图的滞后固定为 1 阶，和摘要页 AIC 不是同一次检验。窗口按步长滑动并且互相重叠，检验次数不是独立样本数。BH 比例是 Benjamini–Hochberg 的参考，不是独立试验的 q 值。

**Walk-forward。** 样本跨过三个日历年并且足够长时，训练段逐年变长，下一年做验证。每一折同时选开仓和平仓阈值。样本不够就只切成前后两半。验证段不参与挑选。各折验证段的日收益会接成一条样本外净值。默认规格已经预计算的，打开就能看到。规格不同时，这一步在回测页点按钮后才计算。

**夏普。** 年数是第一根和最后一根 K 线的日历跨度除以 365.25。年化因子是这段行数除以这个年数，不再固定乘 √252。只在同时开市日上重算的夏普单独列出。无风险利率按 0。区间用同一个年化因子做区块自助法。PSR 是未年化夏普大于 0 的概率。压缩夏普再把开仓阈值的搜索次数算进去。它们都不是获利概率。融券费仍按一年 252 天计提。

**成本。** 佣金、印花税、滑点和融券费分开。方向变化那天记入开平仓，方向不变、只改 β 的记入再平衡。回测页可以另下一行 H 股现金红利的 20% 预扣：只扣前一日已经持有 H 股多头的日子，按当日汇率换成人民币。没有分红序列时这一行留空。主回测仍用复权价。

**V1.5。** 默认引擎仍是 v14。侧栏切到 v15 后，页顶改成复权价溢价 z、样本外净收益、样本外夏普、样本外回撤、Reality Check 的 p 值和半衰期。样本外折要在回测页点按钮才算，打开页面不算，也不在每日预计算里。训练 756 个同时开市日，测试 126 日，中间隔 121 日，只用测试段之前的数据选开仓、平仓和估计窗口。折起点重算一次 Engle-Granger，半衰期按日更新；不是每个参数、每一天都重算 EG。折起点 EG 不通过时，这一折的所有候选都不新开仓。z 完全来自复权价溢价；税后股息差只作参考显示，要进入信号需要逐日变化的楔子，这一步还没做。测试段用两地交易日的并集记账：持仓在单边开市日记开市一边的涨跌，当天不成交、不收调仓费。分折边界和信号仍按共同开市日确定。成交成本按 100 万人民币名义本金换成收益率，最低佣金不会按零点几元去扣。H 股直接账户红利税率和港交所征费合计仍标成未核实。RC p 不是获利概率。样本外不足 4 折时，统计结论不可靠。v15 不改 v14 的数字。

**预计算。** 工作日收盘后的定时任务刷新行情、H 股分红、北向、南向、恒生 AH 溢价指数、融券标的名单和流动性扫描，写入 `data/published/`。页面先读 `data/cache`，没有文件时读已发布副本，打开页面不下载。四只快捷股票在默认规格下的 walk-forward 和稳定性地图放在 `data/desk/`。规格一致、结束日不早于样本截止日并且比该日晚不超过 10 天时打开就能看到，并写明这一天。改了对冲、窗口、开平仓、成本或起始日之后，仍要点按钮重算。预计算结果不会冒充另一套规格。

**链接。** 点「应用参数」后，地址栏写入股票、语言、日期、对冲、两个窗口、开仓和平仓阈值、成交时点、是否做空，以及成本是法定档还是零。打开链接就用这套规格。佣金小数、卡尔曼 δ 和再平衡带不在地址里。写错的字段会被丢掉，侧栏标出代号。

**多标的。** 页面上的四只和大约十二只是现算。流动性子集的离线摘要放在 `data/scan_snapshot/`，表上写明生成日、样本区间、不做空、法定成本和次日收盘。扫描日早于行情缓存时，页面会标明。全部股票池仍用 `python -m ah_pairs.scan --all`，不要在页面里跑。等风险组合用各对策略日收益的滚动波动倒数，权重滞后一天。p 值更小不是「最协整」的发现。

**多重比较。** 阈值网格、多折挑选和按 p 值给多只股票排序，都会把同一段行情试很多次。最好的那一格偏乐观。

**相对市场。** 诊断页把个股溢价减去恒生 AH 溢价指数。指数 100 当作平价，减的是指数/100 − 1，系数固定为 1。接着做的是普通 ADF，原假设是相对溢价有单位根。它不替换 EG p 值。指数还没有缓存时这一项不算。

**结构断点。** Gregory–Hansen 允许 log A 和 log H 的关系变一次。模型 C 只改截距，模型 C/T 加趋势，模型 C/S 连斜率一起改。临界值用 Gregory and Hansen (1996) 表 1 的 ADF*，一个回归元；模型 C 的 5% 是 −4.61。两端各去掉 15%，各断点用同一个 Schwert 滞后。点按钮才算。拒绝不替换 EG p 值。

**纯多头。** 回测页另给三行：永远持 A、永远持 H、按溢价 z 切换。z < 0 持 A，z ≥ 0 持 H，不做空。成交时点和主回测一样。这是已经要持有这家公司时的比较，不替换配对回测。

**月度截面。** 一组对照页点按钮后，用本地缓存给流动性子集按溢价 z 排序。每月做多 z 最低四分之一的 A 股，并做空它们的 H 股。不做空 A。没有缓存的股票跳过，页面不下载。

**融券名单。** 更新按钮下载沪深融资融券标的和保证金比例。那不是融券年化费率，回测费率仍用侧栏。默认不做空，名单不改数字。打开做空之后，只有名单覆盖到的日期才按名单限制做空；更早的日子仍按开关。

**前瞻账本。** 前瞻记录页列出每只股票 × 每个预设的模拟账户。账本从上线日空仓起步，历史行情只用于指标预热和合规训练，不把上线前的回测仓位或收益算进前瞻业绩。每个工作日收盘后由定时任务追加一行：记录行情截止日和系统实际观察时间两个时间，后者不倒签。账本只追加；上游更正行情时旧行不动，冲突会写进刷新清单。漏观察的区间标 backfilled 行并记缺口事件，事件里写明这段按规则会发生几笔成交；成交是规则对数据的确定函数，但系统当时并未实时观察，所以展示端的前瞻净值在缺口之后重新归一，分成若干段，不把这段接进“真实前瞻”。页面打开、手动重算和导出都不写账本；正式写入需要在刷新任务里显式启用，默认关闭。每只股票、每个引擎、每个预设独立记账，不是一个能同时使用全部资金的组合。v15 连续账户沿用 756 个同时开市日训练、121 日隔离、126 日锁参；换参不清仓、不抹掉挂单，已有仓位按入场版本的退出规则走完，平仓后才按新版本开仓。v15 的借券费没有建模，显示的是未计入，不是实际为零。首批预设：v14-default 与 v14-strict（入场阈值 2.5），v15-default 与 v15-strict（训练网格只留更高入场档）；数值在上线前冻结，不按回测赢家反选。

**今日变化。** 今日工作台首屏回答的是今天有什么变化，不是谁的 z 最大。每行比较账本里最近两个决策日：新触发、退出信号、已开仓、已平仓、持续受限、单边行情日分开列。连续越阈只在首次触发日标新触发。信号事件和模拟成交分开记：先出信号，按预设的成交规则在下一次执行时点成交，两列不混。数据没更新或原因不明时如实显示，不猜。

**信号解释卡。** 单对研究页和前瞻记录页各有一张解释卡，回答三个问题：观察到了什么、规则允许什么、账户实际做了什么。理由由引擎在决策时产生，页面不反推。被闸门或做空限制拦下的信号是有效解释，不会被压缩成含义不清的空仓。

**逐笔收益拆解。** 回测页或前瞻记录页选一笔交易，把收益拆成 A 股价格、H 股本币价格、汇率、交易费和借券费五块。H 股人民币收益已含汇率，拆解时港股本币贡献是 wH 乘本币收益，汇率贡献是 wH 乘（1+本币收益）再乘汇率收益，交叉项计入汇率，不重复计算。每日各项加总等于引擎净收益，逐笔各项先用前一日净值换成金额再除以该笔起始本金，加总等于该笔复合净收益。开仓费归新交易、平仓费归旧交易、调仓费归存续交易，同一笔费用只记一次。未平仓交易单独标注，估值截止到最后一根 K 线。

**今日全景。** 今日工作台的下半部分列出流动性子集的最新溢价、z 和持仓。打开时只读定时任务写下的表。点按钮才按当前侧栏从本地缓存重算，不下载，也不跑协整。结束日早于样本截止日，或规格和默认规格不同时，不拿这张表冒充。点一行打开那一对。默认不做空 A。

**提醒订阅。** 信号页可以保存邮件地址或 webhook。地址写在本地缓存，不进 git。公开部署可设 `AH_SUBSCRIPTIONS_READONLY=1`，此时不能保存、删除或手动发送；订阅请用 `AH_SUBSCRIPTIONS_JSON`。定时任务仍会向环境变量 ALERT_WEBHOOK_URL 发送。邮件只在设置了 ALERT_SMTP_HOST 和 ALERT_SMTP_FROM 时发送；没设置就只保存，不假装已经发出。

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

**V1.5.** The default engine is still v14. Switching the sidebar to v15 replaces the headline with adjusted-price premium z, out-of-sample net, out-of-sample Sharpe, out-of-sample drawdown, the reality-check p-value and the half-life. The out-of-sample folds run only from the button on the backtest page. They are not part of the daily precompute. Training is 756 common sessions, the test is 126, and 121 sessions are left out in between. Entry, exit and the estimation window are chosen only on data before the test. Engle–Granger is recomputed once at the start of a fold; the half-life updates daily. If the fold-start EG test fails, no candidate in that fold opens. The z comes entirely from the adjusted-price premium; the after-tax dividend gap is display-only reference and would need a time-varying wedge to enter the signal, which is not implemented. Test folds are accounted on the union calendar: an open position earns the side that traded on a one-sided day, with no fill and no rebalance cost that day. Fold boundaries and signals still use common sessions. Trading costs are converted to a return on a one-million-yuan book, so the cash minimum commission is not charged on a fraction of one yuan. The H-share direct-account dividend tax and the HKEX levy total remain unverified. The RC p-value is not the probability of profit. Fewer than four out-of-sample folds is not a reliable claim. v15 does not change the v14 numbers.

**Precompute.** After the close on weekdays, a scheduled job refreshes prices, H-share dividends, northbound, southbound, the Hang Seng AH premium index, the margin-eligible list and the liquid scan, and writes them to `data/published/`. The page reads `data/cache` first and the published copy when that file is missing. Opening the page does not download. Walk-forward and the stability map for the four shortcut names, at the default specification, live in `data/desk/`. They appear on open when the specification matches, the end date is no earlier than the sample end, and it is at most 10 days later. The page names that sample end. A different hedge, window, entry, exit, cost or start date still needs the button. A precomputed result is not presented as a different specification.

**Link.** Apply writes the pair, language, dates, hedge, both windows, entry and exit, fill, whether A-shares can be shorted, and whether costs are the statutory schedule or zero. Opening that link uses this specification. Commission decimals, the Kalman delta and the rebalance band stay out of the address. Invalid fields are dropped and the sidebar names their codes.

**Several pairs.** The four-name and twelve-name tables are computed on demand. An offline summary of the liquid subset lives in `data/scan_snapshot/`, with the generation date, sample window, short-A off, statutory costs and next-close fill. If that date is older than the price cache, the page says so. The full universe remains `python -m ah_pairs.scan --all` and does not run in the page. The equal-risk book uses inverse rolling volatility of each pair’s strategy return, lagged one day. A smaller p-value is not a discovery of the most cointegrated name.

**Multiple testing.** Threshold grids, fold-by-fold selection and ranking many pairs on one sample all reuse the same prices. The best cell is biased upward.

**Versus the market.** Diagnostics subtract the Hang Seng AH premium index from the stock premium. A reading of 100 is parity, so the index enters as level/100 − 1 and the coefficient is fixed at 1. The next step is an ordinary ADF whose null is a unit root in that relative premium. It does not replace the EG p-value. If the index is not cached, this block is skipped.

**One break.** Gregory–Hansen lets the relation between log A and log H change once. Model C shifts the intercept, model C/T adds a trend, and model C/S also shifts the slope. Critical values are the ADF* column of Gregory and Hansen (1996), Table 1, for one regressor. Model C’s 5% value is −4.61. Each end drops 15%, and every break uses the same Schwert lag. It runs from the button. A rejection does not replace the EG p-value.

**Long only.** The backtest page adds three rows: always hold A, always hold H, and switch on the premium z. Hold A when z < 0 and H when z ≥ 0. Nothing is shorted. The fill is the same as the main backtest. This is the comparison for someone who would hold the company anyway. It does not replace the pairs backtest.

**Monthly cross-section.** The button on the pairs page ranks the liquid subset by premium z, using only local cache. Each month it buys the cheapest quartile of A-shares and shorts those names’ H-shares. A-shares are not shorted. Names without a cache file are skipped. The page does not download them.

**Borrow list.** The update button downloads the margin-eligible list and the margin ratio. That ratio is not the annual borrow fee; the backtest still uses the sidebar fee. Shorting stays off by default, so the list does not change the figures. After shorting is turned on, only dates covered by the list can block a short. Earlier dates still follow the switch.

**The forward ledger.** The Forward record page shows one simulated account per pair per preset. Ledgers start flat on launch day; earlier prices only warm up indicators and legal training, so no backtest position or return is inherited. Each weekday close, the scheduled job appends one row recording both the data cutoff and the actual observation time; the latter is never backdated. Rows are append-only: if upstream revises prices, old rows stay and the conflict is written to the refresh manifest. Missed windows that cannot be verified are recorded as gaps, not filled with invented trades. Rows replayed across a missed window are marked backfilled, the gap event records how many rule-implied fills it contained, and the displayed forward equity renormalizes after the gap instead of absorbing it. Opening the page, recomputing and exporting never write the ledger. Each pair, engine and preset is an independent simulation — not a portfolio allocating capital. v15 runs continuously with 756 common sessions of training, a 121-session embargo and 126-session locked parameters; a parameter roll neither liquidates positions nor discards pending orders, and an open position finishes under the exit rules of the version that opened it. The v15 borrow cost is not modelled and is labelled as such, never shown as a true zero. The first presets are v14-default and v14-strict (entry 2.5), v15-default and v15-strict (training grid restricted to the higher entry band); values are frozen before launch, never picked ex post from winners.

**Today's changes.** The workbench answers what changed today, not which z is largest. Each row compares the last two decision days in the ledger: new trigger, exit signal, filled entry, filled exit, persistent block and one-sided day are separate. A repeated crossing is only new on the first trigger day. Signal events and simulated fills are separate columns. Missing or unverifiable data is shown as such, not guessed.

**The explanation card.** The pair-research page and the forward record page answer three questions: what was observed, what the rules allowed, and what the account actually did. Reasons are produced by the engine at decision time; the page does not infer them. A signal blocked by the gate or the short-sale rule is a valid explanation, never compressed into a meaningless "flat".

**Per-trade attribution.** Both the backtest page and the forward record page can split a trade into A-share price, H-share local price, FX, fees and borrow cost. The CNY return of the H leg already contains FX, so the split assigns wH times the local return to H and wH times (1 + local return) times the FX return to FX; the cross term goes to FX and is never double counted. Daily parts sum to the engine net return; per-trade parts are converted to amounts with the previous day's equity and divided by the trade's starting capital, summing to the compounded trade return. Entry fees belong to the new trade, exit fees to the closing one, rebalance fees to the ongoing one; every fee is counted exactly once. Open trades are marked as unrealized, valued at the last row.

**Today.** The lower half of the workbench lists the latest premium, z and position for the liquid subset. Opening it only reads the table written by the scheduled job. The button recomputes from the local cache with the current sidebar. It does not download and it does not run a cointegration test. An earlier end date, or a specification other than the default, does not reuse that table. Clicking a row opens the pair. Short A-shares stay off by default.

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
