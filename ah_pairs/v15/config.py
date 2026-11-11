"""V1.5 开关。默认保持 v14，等双跑对比之后再改这一行。"""

from __future__ import annotations

DEFAULT_ENGINE = "v14"
ENGINES = ("v14", "v15")

# 上线前仍需人工核对的两项。未核对前不要把默认引擎改成 v15。
UNVERIFIED = (
    "tau_H_direct",
    "h_levy_total",
)
