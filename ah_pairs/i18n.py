"""界面语言。简体是原文，繁体在运行时由简体转出，英文单独写在调用处。"""

from __future__ import annotations

from contextvars import ContextVar

_LANG: ContextVar[str] = ContextVar("ah_lang", default="zh")
LANGS = ("zh", "zh-Hant", "en")


def get_lang() -> str:
    return _LANG.get()


def set_lang(lang: str) -> None:
    _LANG.set(lang if lang in LANGS else "zh")


def to_traditional(text: str) -> str:
    try:
        import zhconv
    except Exception:
        return text
    return zhconv.convert(text, "zh-hant")


def tx(zh: str, en: str) -> str:
    """按当前语言返回一句界面文字。"""
    lang = get_lang()
    if lang == "en":
        return en
    if lang == "zh-Hant":
        return to_traditional(zh)
    return zh
