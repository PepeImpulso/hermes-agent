"""Native Telegram evidence is emitted only for a causal, provider-ACKed text reply."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms import base as platform_base
from gateway import native_telegram_receipts as receipts
from gateway.native_telegram_receipts import capture_native_text_ack
from plugins.platforms.telegram.adapter import TelegramAdapter


def _case():
    event = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.TELEGRAM, user_id="123", chat_id="123", chat_type="private"),
        message_id="41", platform_update_id=90, internal=False,
        _native_telegram_text_only=True,
        _native_telegram_completed_turn=("telegram:123", "session-1"),
        ledger_message_id=None,
    )
    result = SimpleNamespace(
        success=True, message_id="42",
        raw_response={"message_ids": ["42"], "ack_reply_to_message_id": 41},
    )
    adapter = SimpleNamespace(_bot=SimpleNamespace(id=777))
    return event, result, adapter


def test_captures_private_causal_ack_once_without_message_content(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "_private_directory", lambda home: tmp_path)
    event, result, adapter = _case()
    args = dict(config={"enabled": True, "user_id": "123"}, event=event,
                session_key="telegram:123", result=result, adapter=adapter,
                code_sha="a" * 40, home=tmp_path)
    path = capture_native_text_ack(**args)
    assert path is not None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["inboundMessageId"] == receipt["replyToInboundMessageId"] == "41"
    assert receipt["providerReplyMessageId"] == "42"
    assert receipt["sessionId"] == "session-1" and receipt["gatewaySourceSha"] == "a" * 40
    assert receipt["turnCompleted"] is True and receipt["deliveryKind"] == "nonstreamed_text"
    assert "content" not in receipt and "token" not in receipt
    assert path.read_bytes().endswith(b"\n")
    facts = receipts.native_telegram_facts(receipt)
    assert facts["ingressKind"] == "hermes_native_gateway"
    assert facts["inboundMessageDigest"] == facts["replyToInboundMessageDigest"]
    assert facts["providerAcknowledged"] is True and facts["hermesCompleted"] is True
    assert capture_native_text_ack(**args) is None


def test_fact_derivation_rejects_forged_reply_or_extra_content(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "_private_directory", lambda home: tmp_path)
    event, result, adapter = _case()
    path = capture_native_text_ack(
        config={"enabled": True, "user_id": "123"}, event=event,
        session_key="telegram:123", result=result, adapter=adapter,
        code_sha="a" * 40, home=tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["replyToInboundMessageId"] = "40"
    with pytest.raises(ValueError, match="causal"):
        receipts.native_telegram_facts(value)
    value["replyToInboundMessageId"] = "41"
    value["content"] = "untrusted text"
    with pytest.raises(ValueError, match="invalid"):
        receipts.native_telegram_facts(value)


@pytest.mark.linux_only
def test_receipt_directory_and_file_are_private_on_linux(tmp_path):
    event, result, adapter = _case()
    path = capture_native_text_ack(
        config={"enabled": True, "user_id": "123"}, event=event,
        session_key="telegram:123", result=result, adapter=adapter,
        code_sha="a" * 40, home=tmp_path)
    assert path is not None
    assert path.parent.stat().st_mode & 0o077 == 0
    assert path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("mutation", [
    lambda event, result, adapter: setattr(event, "platform_update_id", None),
    lambda event, result, adapter: setattr(event, "_native_telegram_completed_turn", None),
    lambda event, result, adapter: setattr(event, "_native_telegram_text_only", False),
    lambda event, result, adapter: setattr(event, "ledger_message_id", "99"),
    lambda event, result, adapter: setattr(event.source, "user_id", "456"),
    lambda event, result, adapter: setattr(event.source, "chat_id", "456"),
    lambda event, result, adapter: setattr(event.source, "chat_type", "group"),
    lambda event, result, adapter: setattr(result, "success", False),
    lambda event, result, adapter: result.raw_response.update(ack_reply_to_message_id=None),
    lambda event, result, adapter: result.raw_response.update(ack_reply_to_message_id=40),
    lambda event, result, adapter: result.raw_response.update(message_ids=["42", "43"]),
    lambda event, result, adapter: setattr(adapter._bot, "id", None),
])
def test_missing_or_mismatched_causal_witness_never_writes(tmp_path, mutation):
    event, result, adapter = _case()
    mutation(event, result, adapter)
    assert capture_native_text_ack(
        config={"enabled": True, "user_id": "123"}, event=event,
        session_key="telegram:123", result=result, adapter=adapter,
        code_sha="a" * 40, home=tmp_path) is None
    assert not (tmp_path / "native_telegram_receipts").exists()


def test_opt_in_is_required(tmp_path):
    event, result, adapter = _case()
    assert capture_native_text_ack(
        config=None, event=event, session_key="telegram:123", result=result,
        adapter=adapter, code_sha="a" * 40, home=tmp_path) is None


def test_interrupted_file_write_leaves_no_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "_private_directory", lambda home: tmp_path)
    monkeypatch.setattr(receipts.os, "fsync", lambda descriptor: (_ for _ in ()).throw(OSError("disk")))
    event, result, adapter = _case()
    with pytest.raises(OSError, match="disk"):
        capture_native_text_ack(
            config={"enabled": True, "user_id": "123"}, event=event,
            session_key="telegram:123", result=result, adapter=adapter,
            code_sha="a" * 40, home=tmp_path)
    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.linux_only
def test_symlinked_receipt_home_is_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="link"):
        receipts._private_directory(link)


@pytest.mark.asyncio
async def test_telegram_transport_exposes_provider_confirmed_reply_anchor():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(
            message_id=42, reply_to_message=SimpleNamespace(message_id=41))))
    adapter._should_attempt_rich = lambda *args, **kwargs: False
    result = await adapter.send("123", "hello", reply_to="41", metadata={"notify": True})
    assert result.success is True and result.message_id == "42"
    assert result.raw_response["ack_reply_to_message_id"] == 41
    assert adapter._bot.send_message.await_args.kwargs["reply_to_message_id"] == 41


@pytest.mark.asyncio
async def test_transport_ack_without_reply_anchor_is_not_native_proof(tmp_path):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)))
    adapter._should_attempt_rich = lambda *args, **kwargs: False
    result = await adapter.send("123", "hello", reply_to="41", metadata={"notify": True})
    event, _, _ = _case()
    assert result.success is True
    assert result.raw_response["ack_reply_to_message_id"] is None
    assert capture_native_text_ack(
        config={"enabled": True, "user_id": "123"}, event=event,
        session_key="telegram:123", result=result, adapter=adapter,
        code_sha="a" * 40, home=tmp_path) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("text_only", [True, False])
async def test_final_delivery_invokes_capture_only_for_text_lane(monkeypatch, text_only):
    event, result, delivery_adapter = _case()
    event._native_telegram_text_only = text_only
    capture = Mock()
    monkeypatch.setattr(receipts, "capture_native_text_ack", capture)
    monkeypatch.setattr(platform_base, "_config_section", lambda section: {
        "native_telegram_receipts": {"enabled": True, "user_id": "123"}})
    handler = SimpleNamespace(send_final_ledgered=AsyncMock(return_value=(result, delivery_adapter)))
    deliveries = []
    await platform_base.BasePlatformAdapter._send_final_text(
        handler, event, "telegram:123", "hello", {}, False, 0, deliveries.append)
    assert deliveries == [result]
    assert capture.call_count == int(text_only)
    if text_only:
        assert capture.call_args.kwargs["adapter"] is delivery_adapter
        assert capture.call_args.kwargs["event"] is event
