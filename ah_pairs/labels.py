"""表格行名和列名。计算层保留稳定代号，界面再用当前语言显示。"""

from __future__ import annotations

from ah_pairs.i18n import tx

SLICE_ZH = {
    "full": "全区间",
    "first_half": "前半段",
    "second_half": "后半段",
    "drop_first_year": "去掉最早约一年",
    "drop_last_year": "去掉最近约一年",
    "drop_move_5": "去掉单日涨跌超过 5% 的日子",
    "drop_move_3": "去掉单日涨跌超过 3% 的日子",
    "drop_spread_p99": "去掉价差跳动最大的 1% 日子",
    "unadjusted": "未复权收盘",
}

SLICE_EN = {
    "full": "Full sample",
    "first_half": "First half",
    "second_half": "Second half",
    "drop_first_year": "Drop about the first year",
    "drop_last_year": "Drop about the last year",
    "drop_move_5": "Drop days that moved more than 5%",
    "drop_move_3": "Drop days that moved more than 3%",
    "drop_spread_p99": "Drop the largest 1% of spread jumps",
    "unadjusted": "Unadjusted closes",
}

COST_CASE = {
    "current": ("当前成本", "Current costs"),
    "zero": ("不计成本", "Zero costs"),
    "slip10": ("滑点再加 10 bps", "Slippage plus 10 bps"),
}

WF_ROLE = {
    "train": ("训练 {fold}", "Train {fold}"),
    "valid": ("验证 {fold}", "Validation {fold}"),
    "train_half": ("训练（选出参数）", "Train (parameters chosen)"),
    "valid_half": ("验证（锁定参数）", "Validation (parameters locked)"),
    "full_locked": ("全样本（锁定参数）", "Full sample (locked parameters)"),
    "full_sidebar": ("全样本（侧栏当前参数）", "Full sample (sidebar parameters)"),
}


def slice_label(code: str) -> str:
    return tx(SLICE_ZH.get(code, code), SLICE_EN.get(code, code))


def cost_case_label(code: str) -> str:
    zh, en = COST_CASE.get(code, (code, code))
    return tx(zh, en)


def walk_row_label(role: str, fold: str = "", fallback: str = "") -> str:
    pair = WF_ROLE.get(role)
    if pair is None:
        return fallback or role
    zh, en = pair
    if "{fold}" in zh or "{fold}" in en:
        zh = zh.format(fold=fold)
        en = en.format(fold=fold)
    return tx(zh, en)
