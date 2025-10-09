"""Tests for intent classification prompt consistency."""

from __future__ import annotations

import re

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.services.preprocessing import INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT


def test_intent_prompt_lists_all_intents() -> None:
    """Ensure the prompt enumerates every intent and only valid ones."""

    intent_names = set(re.findall(r'<intent name="([^"]+)"', INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT))
    enum_values = {intent.value for intent in IntentType}
    assert intent_names == enum_values

    example_intents = set(re.findall(r"<intent>([^<]+)</intent>", INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT))
    assert example_intents <= enum_values
