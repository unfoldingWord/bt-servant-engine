"""Unit tests for OpenAI utilities."""

from __future__ import annotations

from types import SimpleNamespace

from bt_servant_engine.services import openai_utils as utils

CACHED_INPUT_TOKEN_COUNT = 42
CACHED_PROMPT_TOKEN_COUNT = 7
INPUT_TOKENS = 10
OUTPUT_TOKENS = 5

# pylint: disable=missing-function-docstring,missing-class-docstring,too-few-public-methods


def test_extract_cached_input_tokens_handles_various_shapes() -> None:
    usage = SimpleNamespace(
        input_token_details={"cache_read_input_tokens": CACHED_INPUT_TOKEN_COUNT}
    )
    assert utils.extract_cached_input_tokens(usage) == CACHED_INPUT_TOKEN_COUNT

    usage_dict = SimpleNamespace(prompt_tokens_details={"cached_tokens": CACHED_PROMPT_TOKEN_COUNT})
    assert utils.extract_cached_input_tokens(usage_dict) == CACHED_PROMPT_TOKEN_COUNT

    assert utils.extract_cached_input_tokens(object()) is None

    class Exploding:
        @property
        def input_token_details(self):  # type: ignore[override]
            raise RuntimeError("boom")

    assert utils.extract_cached_input_tokens(Exploding()) is None


def test_track_openai_usage_invokes_token_counter() -> None:
    usage = SimpleNamespace(
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        total_tokens=None,
    )
    recorded: list = []

    def fake_add_tokens(increments):
        recorded.append(increments)

    utils.track_openai_usage(usage, "gpt-4o", add_tokens_fn=fake_add_tokens)
    assert recorded
    first = recorded[0]
    assert getattr(first, "input_tokens", None) == INPUT_TOKENS
    assert getattr(first, "output_tokens", None) == OUTPUT_TOKENS
