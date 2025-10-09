"""Unit tests for OpenAI utilities."""

from __future__ import annotations

from types import SimpleNamespace

from bt_servant_engine.services import openai_utils as utils

# pylint: disable=missing-function-docstring,missing-class-docstring,too-few-public-methods


def test_extract_cached_input_tokens_handles_various_shapes() -> None:
    usage = SimpleNamespace(input_token_details={"cache_read_input_tokens": 42})
    assert utils.extract_cached_input_tokens(usage) == 42

    usage_dict = SimpleNamespace(prompt_tokens_details={"cached_tokens": 7})
    assert utils.extract_cached_input_tokens(usage_dict) == 7

    assert utils.extract_cached_input_tokens(object()) is None

    class Exploding:
        @property
        def input_token_details(self):  # type: ignore[override]
            raise RuntimeError("boom")

    assert utils.extract_cached_input_tokens(Exploding()) is None


def test_track_openai_usage_invokes_token_counter() -> None:
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=None)
    recorded: list[tuple] = []

    def fake_add_tokens(*args, **kwargs):
        recorded.append((args, kwargs))

    utils.track_openai_usage(usage, "gpt-4o", add_tokens_fn=fake_add_tokens)
    assert recorded
