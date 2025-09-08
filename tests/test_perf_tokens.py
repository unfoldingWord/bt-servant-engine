"""Unit tests for perf token accumulation on spans."""

from utils import perf


def test_add_tokens_accumulates_into_current_span():
    """add_tokens sums input/output/total into the current span."""
    tid = "trace-1"
    perf.set_current_trace(tid)
    with perf.time_block("brain:test_node"):
        perf.add_tokens(10, 20, 30)
        perf.add_tokens(1, 2, 3)
    report = perf.summarize_report(tid)
    spans = report["spans"]
    assert spans, "expected at least one span in report"
    s0 = spans[0]
    assert s0["name"] == "brain:test_node"
    assert s0["input_tokens_expended"] == 11
    assert s0["output_tokens_expended"] == 22
    assert s0["total_tokens_expended"] == 33
