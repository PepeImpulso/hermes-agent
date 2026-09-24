"""Opt-in, private evidence of a completed native Telegram text turn.

This records an observed provider reply ACK, not an authorization to send or a
claim about streaming, voice, media, tools, or other outbound messages.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat

from hermes_constants import get_hermes_home


_ID = re.compile(r"^[1-9][0-9]*$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_RECEIPT_KEYS = {
    "schema", "observedAt", "gatewaySourceSha", "botId", "userId", "chatId",
    "updateId", "inboundMessageId", "sessionId", "providerReplyMessageId",
    "replyToInboundMessageId", "turnCompleted", "deliveryKind",
}


def _positive_id(value) -> str | None:
    text = str(value) if value is not None and not isinstance(value, bool) else ""
    return text if _ID.fullmatch(text) else None


def _canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def native_telegram_facts(receipt: dict) -> dict:
    """Derive #2503's ``nativeTelegram`` facts from a reviewed private receipt.

    The surrounding NitrOdoo capability observation still needs independent
    authorization, functional facts, a dedicated Hermes signer and signature.
    This function alone never certifies a live migration.
    """
    if (not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS
            or receipt.get("schema") != "hermes-native-telegram-text-ack-v1"):
        raise ValueError("invalid native Telegram receipt")
    for field in ("botId", "userId", "chatId", "updateId", "inboundMessageId",
                  "providerReplyMessageId", "replyToInboundMessageId"):
        if not isinstance(receipt.get(field), str) or _positive_id(receipt[field]) is None:
            raise ValueError(f"invalid native Telegram {field}")
    if (receipt["userId"] != receipt["chatId"]
            or receipt["inboundMessageId"] != receipt["replyToInboundMessageId"]
            or receipt.get("turnCompleted") is not True
            or receipt.get("deliveryKind") != "nonstreamed_text"
            or not isinstance(receipt.get("gatewaySourceSha"), str)
            or not _SHA.fullmatch(receipt["gatewaySourceSha"])
            or not isinstance(receipt.get("sessionId"), str) or not receipt["sessionId"]):
        raise ValueError("native Telegram causal chain is incomplete")

    def digest(value):
        return hashlib.sha256(_canonical(value)).hexdigest()

    message = {"chatId": receipt["chatId"], "messageId": receipt["inboundMessageId"]}
    return {
        "ingressKind": "hermes_native_gateway",
        "botIdentityDigest": digest({"botId": receipt["botId"]}),
        "gatewaySourceDigest": digest({"gatewaySourceSha": receipt["gatewaySourceSha"]}),
        "inboundUpdateDigest": digest({"botId": receipt["botId"], "updateId": receipt["updateId"]}),
        "inboundMessageDigest": digest(message),
        "hermesTaskDigest": digest({"sessionId": receipt["sessionId"],
                                    "updateId": receipt["updateId"],
                                    "inboundMessageId": receipt["inboundMessageId"]}),
        "replyMessageDigest": digest({"chatId": receipt["chatId"],
                                      "messageId": receipt["providerReplyMessageId"]}),
        "replyToInboundMessageDigest": digest(message),
        "providerAcknowledged": True,
        "hermesCompleted": True,
    }


def _private_directory(home: Path) -> Path:
    get_effective_uid = getattr(os, "geteuid", None)
    if os.name != "posix" or get_effective_uid is None:
        raise ValueError("native Telegram receipt capture requires POSIX file permissions")
    owner_uid = get_effective_uid()
    home = Path(os.path.abspath(home))
    for ancestor in (home, *home.parents):
        info = ancestor.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("native Telegram receipt path traverses a link or non-directory")
    if home.lstat().st_uid != owner_uid:
        raise ValueError("native Telegram receipt home has a different owner")
    directory = home / "native_telegram_receipts"
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = directory.lstat()
    if (not stat.S_ISDIR(info.st_mode) or directory.is_symlink()
            or info.st_uid != owner_uid or info.st_mode & 0o077):
        raise ValueError("native Telegram receipt directory is not private")
    return directory


def capture_native_text_ack(*, config, event, session_key, result, adapter, code_sha=None, home=None):
    """Write one immutable-by-convention receipt; return its path or None.

    The caller invokes this only on the normal final-text delivery lane. Every
    missing causal witness fails closed without affecting message delivery.
    """
    if not isinstance(config, dict) or config.get("enabled") is not True:
        return None
    owner_id = _positive_id(config.get("user_id"))
    source = getattr(event, "source", None)
    if (not owner_id or _positive_id(getattr(source, "user_id", None)) != owner_id
            or _positive_id(getattr(source, "chat_id", None)) != owner_id):
        return None
    if getattr(getattr(source, "platform", None), "value", None) != "telegram":
        return None
    if getattr(source, "chat_type", None) not in ("private", "dm") or getattr(event, "internal", False):
        return None
    if not getattr(event, "_native_telegram_text_only", False):
        return None
    turn = getattr(event, "_native_telegram_completed_turn", None)
    if not isinstance(turn, tuple) or len(turn) != 2 or turn[0] != session_key or not turn[1]:
        return None
    if getattr(event, "ledger_message_id", None) not in (None, getattr(event, "message_id", None)):
        return None  # queued chains need their own causal receipt contract
    update_id = _positive_id(getattr(event, "platform_update_id", None))
    inbound_id = _positive_id(getattr(event, "message_id", None))
    chat_id = _positive_id(getattr(source, "chat_id", None))
    bot_id = _positive_id(getattr(getattr(adapter, "_bot", None), "id", None))
    reply_id = _positive_id(getattr(result, "message_id", None))
    response = getattr(result, "raw_response", None)
    if not getattr(result, "success", False) or not isinstance(response, dict):
        return None
    message_ids = response.get("message_ids")
    anchor = _positive_id(response.get("ack_reply_to_message_id"))
    if (not all((update_id, inbound_id, chat_id, bot_id, reply_id))
            or message_ids != [reply_id] or anchor != inbound_id):
        return None
    if code_sha is None:
        from hermes_cli.build_info import get_code_identity
        code_sha = get_code_identity().get("sha")
    if not isinstance(code_sha, str) or not _SHA.fullmatch(code_sha):
        return None
    receipt = {
        "schema": "hermes-native-telegram-text-ack-v1",
        "observedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "gatewaySourceSha": code_sha,
        "botId": bot_id,
        "userId": owner_id,
        "chatId": chat_id,
        "updateId": update_id,
        "inboundMessageId": inbound_id,
        "sessionId": str(turn[1]),
        "providerReplyMessageId": reply_id,
        "replyToInboundMessageId": anchor,
        "turnCompleted": True,
        "deliveryKind": "nonstreamed_text",
    }
    native_telegram_facts(receipt)
    directory = _private_directory(Path(home) if home is not None else get_hermes_home())
    name = hashlib.sha256(_canonical({"botId": bot_id, "chatId": chat_id,
                                      "updateId": update_id, "inboundMessageId": inbound_id})).hexdigest()
    path = directory / f"{name}.json"
    temporary = directory / f".{name}.{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    linked = False
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            return None  # a duplicate inbound update cannot create fresh proof
        linked = True
        temporary.unlink()
        if os.name == "posix":
            directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return path
    except Exception:
        if linked:
            path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
