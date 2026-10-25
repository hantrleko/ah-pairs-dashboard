"""摘要。"""

from __future__ import annotations

import streamlit as st

from ah_pairs.i18n import tx
from ui.format import half_life_text, pct, pval, render_quick_compare


def render_summary(page) -> None:
    company = page.picked["name"]
    p_text = pval(page.eg["pvalue"])
    life = half_life_text(page.static_hl)
    same_or_not = tx(
        "更像会回到平常水平" if page.eg["reject_5"] else "还不能说它会回到平常水平",
        "looks more like it returns toward a usual gap" if page.eg["reject_5"] else "cannot yet be said to return toward a usual gap",
    )
    st.markdown(
        tx(
            f"这页看的是**同一家公司的两只股票**：内地的 A 股，和香港的 H 股。现在打开的是 **{company}**。\n\n它们不能互相兑换。所以两边价格不一样是常态，不是捡钱的机会。这个看板只回答一件事：价差偏离之后，历史上有没有回到附近。",
            f"This page is about **two listings of the same company**: the mainland A-share and the Hong Kong H-share. Open now: **{company}**.\n\nThey are not convertible into each other. A price gap is normal, not a free profit. The board asks one question: after the gap moved away, did it historically come back.",
        )
    )
    with st.expander(tx("这几个词是什么意思", "What these words mean")):
        st.markdown(
            tx(
                """
**EG p 值。** Engle–Granger 检验的原假设是「没有协整」。p 值小，是更有理由拒绝这个假设。它不是未来赚钱的概率，也不保证价差会收回来。

**残差 ADF。** 把上面那次回归的残差再做平稳性检验。残差是同一段样本估出来的，这个 p 值会偏乐观，不能代替 EG p 值。

**Johansen 检验。** 两条价格放在一起看有没有协整关系。它和 EG 可以一个拒绝、一个不拒绝。

**半衰期。** 如果价差服从 AR(1)，`半衰期 = −ln(2) / ln φ`。意思是走完一半回家路程大约要多少个交易日。
                """,
                """
**EG p-value.** The Engle–Granger null is “no cointegration”. A small p-value is evidence against that null. It is not the probability of making money, and it does not promise that the gap closes.

**Residual ADF.** A stationarity test on the residuals of that same regression. Because the residuals were estimated on the same sample, the p-value is optimistic and must not replace the EG p-value.

**Johansen.** A systems test of the two prices. It can reject when EG does not, and the reverse.

**Half-life.** If the spread is an AR(1), half-life = −ln(2) / ln φ, in trading days.
                """,
            )
        )
    st.markdown(
        tx(
            """
<div class="lesson">
  <article><strong>1. 两地各有一个价格</strong><p>A 股用人民币。H 股用港元，这里先乘汇率，换成人民币再比较。协整只用两地都开市的日子。单边开市日只给已有仓位记账。</p></article>
  <article><strong>2. 看价差会不会回家</strong><p>协整的大白话是：两条价格可以各自乱走，但它们的差距常常会回到一条旧轨道附近。</p></article>
  <article><strong>3. 偏得太多就试着对做</strong><p>A 股相对贵，就假设以后变便宜：卖 A、买 H。默认不允许卖 A。A 股相对便宜，才买 A、卖 H。回到附近就平仓。</p></article>
</div>
            """,
            """
<div class="lesson">
  <article><strong>1. Two prices</strong><p>The A-share is in yuan. The H-share is in Hong Kong dollars, multiplied by the exchange rate. Cointegration uses days when both markets are open. A one-sided day only marks an existing position to market.</p></article>
  <article><strong>2. Does the gap come home</strong><p>Cointegration means the two prices can wander, while the gap between them often returns toward an old path.</p></article>
  <article><strong>3. Fade a large gap</strong><p>If A is rich, the bet is that it cheapens: short A, long H. Shorting A is off by default. If A is cheap, buy A and sell H. Close when the gap comes back.</p></article>
</div>
            """,
        ),
        unsafe_allow_html=True,
    )
    # st.markdown doesn't take unsafe via tx; call it directly.
    # The previous call wrongly passed unsafe_allow_html to tx. Fix below if needed.
    c1, c2, c3 = st.columns(3)
    c1.markdown(tx(f"**最近溢价 {pct(page.premium_last)}**\n\nA 股比换成人民币后的 H 股贵多少。正数是 A 更贵。", f"**Premium {pct(page.premium_last)}**\n\nHow much more expensive the A-share is than the H-share in yuan. Positive means A is richer."))
    c2.markdown(tx(f"**EG p 值 {p_text}**\n\n越小，越能说这段时间价差像会回家。0.05 是常用分界。现在这页{same_or_not}。", f"**EG p-value {p_text}**\n\nSmaller means this sample looks more like a gap that comes home. 0.05 is the usual line. This page {same_or_not}."))
    if life not in {"—", tx("不收敛", "not mean-reverting")}:
        c3.markdown(tx(f"**半衰期 {life} 个交易日**\n\n如果价差真会回家，大概要多久走完一半路程。", f"**Half-life {life} trading days**\n\nIf the gap really comes home, about how long half of that trip takes."))
    else:
        c3.markdown(tx("**半衰期还看不出来**\n\n这段价差不像会稳定地回家。", "**No usable half-life**\n\nThis gap does not look like a stable return home."))
    st.markdown(
        tx(
            """
- **β（贝塔）**：H 股动 1%，A 股大约跟着动多少。用来决定两边各买多少。
- **z 分数**：价差离开最近平均水平有几个标准差。
- **夏普**：收益相对波动。负数就是这段回测在亏。旁边的区间来自区块自助法，不是获利概率。
- **最大回撤**：净值从高点掉下去最深的那段。
- **北向 / 南向 / AH 溢价指数**：市场层面的对照，不是这只股票自己的持仓，也不是价差变动的原因。
            """,
            """
- **Beta:** how much the A-share moves when the H-share moves 1%. It sets the hedge, not equal share counts.
- **z-score:** how many standard deviations the spread is from its recent mean.
- **Sharpe:** return relative to volatility. Negative means this backtest lost money. The interval beside it is a block bootstrap, not a probability of profit.
- **Max drawdown:** the deepest drop from a peak.
- **Northbound / southbound / AH premium index:** market-wide series, not this stock’s own flow and not a cause of the spread.
            """,
        )
    )
    render_quick_compare(page, "compare_from_easy")
