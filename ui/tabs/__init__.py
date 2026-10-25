"""标签页导出。"""

from ui.tabs.backtest import render_backtest
from ui.tabs.diagnose import render_diagnose
from ui.tabs.market import render_market
from ui.tabs.method import render_method
from ui.tabs.scan import render_scan
from ui.tabs.signal import render_signal
from ui.tabs.stability import render_stability
from ui.tabs.summary import render_summary

__all__ = [
    "render_backtest",
    "render_diagnose",
    "render_market",
    "render_method",
    "render_scan",
    "render_signal",
    "render_stability",
    "render_summary",
]
