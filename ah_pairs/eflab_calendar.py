"""EF Lab 三个站点共用的行情日期口径。

三份副本保持同一套规则：本仓库的 ah_pairs/eflab_calendar.py、
OmniFinance 的 core/eflab_calendar.py、Fin-Analysis 的 src/eflab_calendar.py。

行情的截止日期是最后一根已经收盘的 K 线，不是日历上的今天。
沪深 15:00 收盘，港股 16:00 收盘（北京时间）。16:10 之后的工作日才把当天算进去。
周末往前推到周五。法定假日不在这张表里，真实行情仍以文件里的最后一根 K 线为准。
价格比「今天和所选结束日里较早的那天」落后超过 7 个自然日，算过期。
A+H 研究样本默认从 2018-01-01 起。页面结束日可以比样本截止日晚最多 10 天。
一年 252 天只用于融券计提，以及另外两个站点里按日收益年化的地方。
A+H 看板的夏普用首尾日历跨度，不用 252。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    SHANGHAI: tzinfo = ZoneInfo("Asia/Shanghai")
except ZoneInfoNotFoundError:
    SHANGHAI = timezone(timedelta(hours=8))

SESSION_READY = time(16, 10)
PRICE_LAG_DAYS = 7
SAMPLE_START = date(2018, 1, 1)
END_SLACK_DAYS = 10
TRADING_DAYS = 252

LAB_ENTRY = "https://eugenefinance.tech"
# 只放仓库里已经出现过的公开地址。OmniFinance 没有公开地址，不编一个。
SITES: tuple[dict[str, str], ...] = (
    {
        "id": "ah-pairs",
        "name": "A+H",
        "url": "https://ah-pairs-dashboard-production.up.railway.app",
    },
    {
        "id": "fin-analysis",
        "name": "Fin-Analysis",
        "url": "https://financial-analysis-agent-eugenefinance02.streamlit.app/",
    },
)


def as_shanghai(now: datetime | date | None = None) -> datetime:
    if now is None:
        return datetime.now(SHANGHAI)
    if isinstance(now, datetime):
        if now.tzinfo is None:
            return now.replace(tzinfo=SHANGHAI)
        return now.astimezone(SHANGHAI)
    if isinstance(now, date):
        return datetime.combine(now, SESSION_READY, tzinfo=SHANGHAI)
    raise TypeError("now 需要是日期或时间")


def last_completed_session(now: datetime | date | None = None) -> date:
    """最近一个 A 股和港股都已收盘的工作日。"""
    current = as_shanghai(now)
    day = current.date()
    if current.weekday() >= 5 or current.time() < SESSION_READY:
        day = day - timedelta(days=1)
    while day.weekday() >= 5:
        day = day - timedelta(days=1)
    return day


def yahoo_exclusive_end(now: datetime | date | None = None) -> date:
    """结束日不含当天的下载接口，用最近已收盘日的下一天。"""
    return last_completed_session(now) + timedelta(days=1)
