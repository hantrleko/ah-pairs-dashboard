"""把收盘提醒订到邮件或 webhook。

地址只写在缓存目录的 subscriptions.json，或环境变量 AH_SUBSCRIPTIONS_JSON。
两处都不进 git。没有 ALERT_SMTP_HOST 和 ALERT_SMTP_FROM 时，邮件只保存，不发送。
"""

from __future__ import annotations

from datetime import date
from email.message import EmailMessage
import json
import os
from pathlib import Path
import uuid

from ah_pairs.paths import CACHE_DIR


def subscriptions_path(cache_dir: Path | None = None) -> Path:
    directory = CACHE_DIR if cache_dir is None else cache_dir
    return directory / "subscriptions.json"


def load_subscriptions(cache_dir: Path | None = None) -> list[dict]:
    items = _read_file(subscriptions_path(cache_dir))
    items.extend(_read_env())
    return items


def add_subscription(
    kind: str,
    target: str,
    codes: list[str],
    entry_z: float,
    cache_dir: Path | None = None,
) -> dict:
    cleaned = _validate(kind, target, codes, entry_z)
    path = subscriptions_path(cache_dir)
    items = _read_file(path)
    for item in items:
        if item.get("kind") == cleaned["kind"] and item.get("target") == cleaned["target"]:
            raise ValueError("这个地址已经订过了")
    if len(items) >= 20:
        raise ValueError("订阅最多 20 条")
    record = {
        "id": uuid.uuid4().hex[:12],
        "created": date.today().isoformat(),
        "stored": "file",
        **cleaned,
    }
    items.append(record)
    _write(path, items)
    return record


def remove_subscription(sub_id: str, cache_dir: Path | None = None) -> bool:
    path = subscriptions_path(cache_dir)
    items = _read_file(path)
    kept = [item for item in items if item.get("id") != sub_id]
    if len(kept) == len(items):
        return False
    _write(path, kept)
    return True


def subscribed_codes(cache_dir: Path | None = None) -> list[str]:
    """文件和环境变量里要检查的代码。出现 * 就表示整个流动性子集。"""
    found: list[str] = []
    for item in load_subscriptions(cache_dir):
        raw = [str(code) for code in item.get("codes") or []]
        if "*" in raw:
            return ["*"]
        for code in raw:
            if code not in found:
                found.append(code)
    return found


def mask_target(kind: str, target: str) -> str:
    """日志和页面只留邮箱首字母或 webhook 的主机名。"""
    text = str(target or "")
    if kind == "email" and "@" in text:
        name, _, domain = text.partition("@")
        return f"{name[:1]}***@{domain}"
    if "://" in text:
        host = text.split("://", 1)[1].split("/", 1)[0]
        scheme = text.split("://", 1)[0]
        return f"{scheme}://{host}"
    return "已保存"


def smtp_configured() -> bool:
    host = os.environ.get("ALERT_SMTP_HOST", "").strip()
    sender = os.environ.get("ALERT_SMTP_FROM", "").strip()
    return bool(host and sender)


def deliver(
    payloads: list[dict],
    cache_dir: Path | None = None,
    env_entry: float | None = None,
    webhook_post=None,
    smtp_send=None,
) -> list[dict]:
    """向环境变量和订阅地址发送。没配 SMTP 的邮件记为 not_configured，不打开网络。"""
    results = []
    env_url = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
    seen_hooks: set[str] = set()
    if env_url:
        chosen = _above(payloads, env_entry)
        if not chosen:
            results.append(_status("webhook", "env", "quiet", env_url))
        else:
            ok, note = _post(env_url, chosen, webhook_post)
            results.append(_status("webhook", "env", "sent" if ok else "failed", env_url, note))
        seen_hooks.add(env_url)
    seen_mail: set[str] = set()
    for item in load_subscriptions(cache_dir):
        matched = _for_subscription(item, payloads)
        kind = str(item.get("kind") or "")
        target = str(item.get("target") or "")
        sub_id = str(item.get("id") or "")
        if not matched:
            results.append(_status(kind, sub_id, "quiet", target))
            continue
        if kind == "webhook":
            if target in seen_hooks:
                results.append(_status(kind, sub_id, "skipped_duplicate", target))
                continue
            ok, note = _post(target, matched, webhook_post)
            seen_hooks.add(target)
            results.append(_status(kind, sub_id, "sent" if ok else "failed", target, note))
            continue
        if target in seen_mail:
            results.append(_status(kind, sub_id, "skipped_duplicate", target))
            continue
        seen_mail.add(target)
        if not smtp_configured():
            results.append(_status(kind, sub_id, "not_configured", target, "未设置 ALERT_SMTP_HOST 或 ALERT_SMTP_FROM"))
            continue
        try:
            _send_email(target, matched, smtp_send)
        except Exception as exc:
            results.append(_status(kind, sub_id, "failed", target, str(exc)))
        else:
            results.append(_status(kind, sub_id, "sent", target))
    return results


def _validate(kind: str, target: str, codes: list[str], entry_z: float) -> dict:
    kind_text = str(kind or "").strip()
    target_text = str(target or "").strip()
    if kind_text not in {"email", "webhook"}:
        raise ValueError("订阅只能是邮件或 webhook")
    if not target_text or len(target_text) > 500:
        raise ValueError("地址为空或过长")
    if kind_text == "email":
        target_text = target_text.lower()
        name, sep, domain = target_text.partition("@")
        if not sep or not name or "." not in domain or " " in target_text:
            raise ValueError("邮件地址看起来不对")
    elif not (target_text.startswith("https://") or target_text.startswith("http://")):
        raise ValueError("webhook 需要 http 或 https 地址")
    elif " " in target_text:
        raise ValueError("webhook 地址里不能有空格")
    try:
        entry = float(entry_z)
    except (TypeError, ValueError) as exc:
        raise ValueError("开仓阈值需要是数字") from exc
    if not (0 < entry <= 10):
        raise ValueError("开仓阈值需要在 0 和 10 之间")
    cleaned: list[str] = []
    for code in codes:
        text = str(code).strip()
        if text == "*":
            return {"kind": kind_text, "target": target_text, "codes": ["*"], "entry_z": entry}
        if not text.isdigit():
            raise ValueError("股票代码需要是数字")
        cleaned.append(text.zfill(6))
    if not cleaned:
        raise ValueError("至少要有一只股票")
    return {"kind": kind_text, "target": target_text, "codes": cleaned, "entry_z": entry}


def _read_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _read_env() -> list[dict]:
    raw = os.environ.get("AH_SUBSCRIPTIONS_JSON", "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    found = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            cleaned = _validate(
                str(item.get("kind") or ""),
                str(item.get("target") or ""),
                list(item.get("codes") or []),
                item.get("entry_z", 2.0),
            )
        except ValueError:
            continue
        found.append(
            {
                "id": str(item.get("id") or f"env-{index}"),
                "created": item.get("created"),
                "stored": "env",
                **cleaned,
            }
        )
    return found


def _write(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _above(payloads: list[dict], entry: float | None) -> list[dict]:
    if entry is None:
        return list(payloads)
    return [item for item in payloads if _finite_z(item) >= float(entry)]


def _for_subscription(item: dict, payloads: list[dict]) -> list[dict]:
    entry = float(item.get("entry_z") or 0)
    codes = {str(code).zfill(6) for code in item.get("codes") or [] if str(code) != "*"}
    star = "*" in [str(code) for code in item.get("codes") or []]
    matched = []
    for payload in payloads:
        if _finite_z(payload) < entry:
            continue
        code = str(payload.get("a_code") or "").zfill(6)
        if star or not codes or code in codes:
            matched.append(payload)
    return matched


def _finite_z(payload: dict) -> float:
    try:
        value = float(payload.get("z"))
    except (TypeError, ValueError):
        return 0.0
    if value != value:
        return 0.0
    return abs(value)


def _post(url: str, payloads: list[dict], webhook_post) -> tuple[bool, str]:
    body = {"alerts": payloads}
    if webhook_post is not None:
        ok, note = webhook_post(url, body)
        return bool(ok), str(note)
    from ah_pairs.alerts import send_webhook

    return send_webhook(url, body)


def _send_email(recipient: str, payloads: list[dict], smtp_send) -> None:
    sender = os.environ.get("ALERT_SMTP_FROM", "").strip()
    host = os.environ.get("ALERT_SMTP_HOST", "").strip()
    raw_port = os.environ.get("ALERT_SMTP_PORT", "").strip()
    port = int(raw_port) if raw_port else 587
    user = os.environ.get("ALERT_SMTP_USER", "").strip()
    password = os.environ.get("ALERT_SMTP_PASSWORD", "")
    message = _email(sender, recipient, payloads)
    if smtp_send is not None:
        smtp_send(host, port, user, password, sender, recipient, message)
        return
    import smtplib

    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=20) as client:
            if user:
                client.login(user, password)
            client.sendmail(sender, [recipient], message)
        return
    with smtplib.SMTP(host, port, timeout=20) as client:
        client.ehlo()
        client.starttls()
        client.ehlo()
        if user:
            client.login(user, password)
        client.sendmail(sender, [recipient], message)


def _email(sender: str, recipient: str, payloads: list[dict]) -> str:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = "A+H 研究提醒"
    lines = ["这是研究提醒，不是下单指令。A/H 不能兑换。p 值不是获利概率。", ""]
    for item in payloads:
        lines.append(
            f"{item.get('name')} {item.get('a_code')} {item.get('asof')} "
            f"z={float(item.get('z')):.2f} 持仓={int(item.get('position') or 0)}"
        )
    message.set_content("\n".join(lines))
    return message.as_string()


def _status(kind: str, sub_id: str, status: str, target: str, note: str = "") -> dict:
    return {
        "kind": kind,
        "id": sub_id,
        "status": status,
        "target": mask_target(kind, target),
        "note": note,
    }
