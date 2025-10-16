"""Unit tests for perf token accumulation on spans."""

from utils import perf

INITIAL_INPUT_TOKENS = 10
INITIAL_OUTPUT_TOKENS = 20
INITIAL_TOTAL_TOKENS = 30
ADDITIONAL_INPUT_TOKENS = 1
ADDITIONAL_OUTPUT_TOKENS = 2
ADDITIONAL_TOTAL_TOKENS = 3
EXPECTED_INPUT_TOKEN_SUM = 11
EXPECTED_OUTPUT_TOKEN_SUM = 22
EXPECTED_TOTAL_TOKEN_SUM = 33


def test_add_tokens_accumulates_into_current_span():
    """add_tokens sums input/output/total into the current span."""
    tid = "trace-1"
    perf.set_current_trace(tid)
    with perf.time_block("brain:test_node"):
        perf.add_tokens(
            perf.TokenIncrements(
                input_tokens=INITIAL_INPUT_TOKENS,
                output_tokens=INITIAL_OUTPUT_TOKENS,
                total_tokens=INITIAL_TOTAL_TOKENS,
            )
        )
        perf.add_tokens(
            perf.TokenIncrements(
                input_tokens=ADDITIONAL_INPUT_TOKENS,
                output_tokens=ADDITIONAL_OUTPUT_TOKENS,
                total_tokens=ADDITIONAL_TOTAL_TOKENS,
            )
        )
    report = perf.summarize_report(tid)
    spans = report["spans"]
    assert spans, "expected at least one span in report"
    s0 = spans[0]
    assert s0["name"] == "brain:test_node"
    assert s0["input_tokens_expended"] == EXPECTED_INPUT_TOKEN_SUM
    assert s0["output_tokens_expended"] == EXPECTED_OUTPUT_TOKEN_SUM
    assert s0["total_tokens_expended"] == EXPECTED_TOTAL_TOKEN_SUM
