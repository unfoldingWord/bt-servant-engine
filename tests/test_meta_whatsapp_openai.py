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
from dotenv import load_dotenv
import pytest
from fastapi.testclient import TestClient

import bt_servant as api
from config import config as app_config


def _has_real_openai() -> bool:
    key = str(app_config.OPENAI_API_KEY)
    return bool(key and key != "test" and key.startswith("sk-"))


load_dotenv(override=True)

# Require both a real key and an explicit opt-in for API-level OpenAI test
_RUN_OPENAI_API = os.environ.get("RUN_OPENAI_API_TESTS", "")

pytestmark = [
    pytest.mark.openai,
    pytest.mark.skipif(not _has_real_openai() or _RUN_OPENAI_API != "1",
                       reason="OPENAI_API_KEY missing or RUN_OPENAI_API_TESTS!=1"),
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


@pytest.mark.skipif(not _has_real_openai(), reason="OPENAI_API_KEY not set for live OpenAI tests")
def test_meta_whatsapp_keywords_flow_with_openai(monkeypatch):
    # Ensure sandbox guard does not block the test sender
    monkeypatch.setattr(api.config, "IN_META_SANDBOX_MODE", False, raising=True)
    # Record outbound messages instead of hitting Meta
    sent: list[str] = []

    async def _fake_send_text_message(user_id: str, text: str) -> None:  # user_id unused in test
        sent.append(text)

    async def _fake_send_voice_message(user_id: str, text: str) -> None:  # user_id unused in test
        sent.append("[voice] " + text)

    async def _fake_typing_indicator_message(message_id: str) -> None:  # message_id unused in test
        return None

    # Patch the functions as imported into the API module to ensure the endpoint uses fakes
    monkeypatch.setattr(api, "send_text_message", _fake_send_text_message)
    monkeypatch.setattr(api, "send_voice_message", _fake_send_voice_message)
    monkeypatch.setattr(api, "send_typing_indicator_message", _fake_typing_indicator_message)

    client = TestClient(api.app)

    body_obj = _meta_text_payload("What are the keywords in 3 John?")
    body = json.dumps(body_obj).encode("utf-8")

    app_secret = os.environ.get("META_APP_SECRET", "test")
    sig = _make_signature(app_secret, body)
    ua = os.environ.get("FACEBOOK_USER_AGENT", "test")

    # POST webhook
    resp = client.post(
        "/meta-whatsapp",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": sig,
            "User-Agent": ua,
        },
    )
    assert resp.status_code == 200

    # Poll for side-effect (background task) to finish.
    # OpenAI-backed paths can occasionally exceed 20s; allow up to 60s.
    deadline = time.time() + 60
    while time.time() < deadline and not sent:
        time.sleep(0.25)

    assert sent, "No outbound messages captured from keywords flow"
    combined = "\n".join(sent)
    assert "Keywords in 3 John" in combined
