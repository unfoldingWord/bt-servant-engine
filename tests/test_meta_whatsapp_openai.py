"""API integration test for /meta-whatsapp using real OpenAI for intent/selection.

We simulate a Meta webhook POST, compute a valid HMAC signature, and stub outbound
Meta sends. The brain runs normally and will invoke OpenAI for language,
preprocessor, intent classification, and selection parsing. We choose a
keywords-style query to avoid costly summarization calls.
"""
# pylint: disable=missing-function-docstring,line-too-long,duplicate-code,unused-argument
from __future__ import annotations

import os
import hmac
import hashlib
import json
import time
import pytest
from fastapi.testclient import TestClient

import bt_servant as api
import messaging


def _has_real_openai() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key and key != "test")


pytestmark = [
    pytest.mark.openai,
    pytest.mark.skipif(not _has_real_openai(), reason="OPENAI_API_KEY not set for live OpenAI tests"),
]


def _make_signature(app_secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _meta_text_payload(text: str) -> dict:
    ts = str(int(time.time()))
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "15555555555",
                                    "id": "wamid.TEST",
                                    "timestamp": ts,
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }


def test_meta_whatsapp_keywords_flow_with_openai(monkeypatch):
    # Record outbound messages instead of hitting Meta
    sent: list[str] = []

    async def _fake_send_text_message(user_id: str, text: str) -> None:  # user_id unused in test
        sent.append(text)

    async def _fake_send_voice_message(user_id: str, text: str) -> None:  # user_id unused in test
        sent.append("[voice] " + text)

    async def _fake_typing_indicator_message(message_id: str) -> None:  # message_id unused in test
        return None

    monkeypatch.setattr(messaging, "send_text_message", _fake_send_text_message)
    monkeypatch.setattr(messaging, "send_voice_message", _fake_send_voice_message)
    monkeypatch.setattr(messaging, "send_typing_indicator_message", _fake_typing_indicator_message)

    client = TestClient(api.app)

    body_obj = _meta_text_payload("What are the keywords in 3 John?")
    body = json.dumps(body_obj).encode("utf-8")

    app_secret = os.environ.get("META_APP_SECRET", "test")
    sig = _make_signature(app_secret, body)
    ua = os.environ.get("FACEBOOK_USER_AGENT", "test")

    # POST webhook
    resp = client.post(  # type: ignore[arg-type]
        "/meta-whatsapp",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": sig,
            "User-Agent": ua,
        },
    )
    assert resp.status_code == 200

    # Poll for side-effect (background task) to finish
    deadline = time.time() + 20
    while time.time() < deadline and not sent:
        time.sleep(0.25)

    assert sent, "No outbound messages captured from keywords flow"
    combined = "\n".join(sent)
    assert "Keywords in 3 John" in combined
