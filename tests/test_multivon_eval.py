"""Tests for multivon-eval — deterministic evaluators only (no LLM calls)."""
import pytest
from multivon_eval import EvalCase, AgentStep, ToolCall
from multivon_eval.evaluators.deterministic import (
    NotEmpty, ExactMatch, Contains, RegexMatch,
    JSONSchemaEval, WordCount, Latency, BLEU, ROUGE, StartsWith,
)
from multivon_eval.evaluators.agent import ToolCallAccuracy


def case(input="test", expected=None, context=None):
    return EvalCase(input=input, expected_output=expected, context=context)


class TestNotEmpty:
    def test_non_empty(self):
        assert NotEmpty().evaluate(case(), "hello").passed

    def test_empty(self):
        assert not NotEmpty().evaluate(case(), "   ").passed


class TestExactMatch:
    def test_passes_on_match(self):
        r = ExactMatch().evaluate(case(expected="Paris"), "Paris")
        assert r.passed and r.score == 1.0

    def test_fails_on_mismatch(self):
        r = ExactMatch().evaluate(case(expected="Paris"), "London")
        assert not r.passed

    def test_case_insensitive_default(self):
        assert ExactMatch().evaluate(case(expected="paris"), "PARIS").passed

    def test_case_sensitive(self):
        assert not ExactMatch(case_sensitive=True).evaluate(case(expected="paris"), "PARIS").passed

    def test_no_expected_skips(self):
        r = ExactMatch().evaluate(case(), "anything")
        assert r.passed and r.metadata.get("skipped") and r.reason.startswith("[skipped]")


class TestContains:
    def test_all_present(self):
        r = Contains(["red", "blue"]).evaluate(case(), "I see red and blue")
        assert r.passed and r.score == 1.0

    def test_partial(self):
        r = Contains(["red", "blue", "green"]).evaluate(case(), "red and blue")
        assert not r.passed
        assert abs(r.score - 2/3) < 0.01

    def test_case_insensitive(self):
        assert Contains(["PARIS"]).evaluate(case(), "I visited paris").passed


class TestRegexMatch:
    def test_matches(self):
        assert RegexMatch(r"\d{4}").evaluate(case(), "year 2026").passed

    def test_no_match(self):
        assert not RegexMatch(r"\d{4}").evaluate(case(), "no numbers").passed


class TestJSONSchemaEval:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }

    def test_valid(self):
        assert JSONSchemaEval(self.schema).evaluate(case(), '{"name":"Alice","age":30}').passed

    def test_invalid_json(self):
        assert not JSONSchemaEval(self.schema).evaluate(case(), "not json").passed

    def test_schema_violation(self):
        assert not JSONSchemaEval(self.schema).evaluate(case(), '{"name":"Alice"}').passed


class TestWordCount:
    def test_in_range(self):
        assert WordCount(min_words=1, max_words=10).evaluate(case(), "hello world").passed

    def test_too_long(self):
        assert not WordCount(max_words=2).evaluate(case(), "one two three").passed

    def test_too_short(self):
        assert not WordCount(min_words=5).evaluate(case(), "one two").passed


class TestLatency:
    def test_under(self):
        assert Latency(max_ms=1000).evaluate(case(), "r", latency_ms=500).passed

    def test_over(self):
        assert not Latency(max_ms=500).evaluate(case(), "r", latency_ms=1000).passed


class TestBLEU:
    def test_perfect_match(self):
        r = BLEU().evaluate(case(expected="the cat sat"), "the cat sat")
        assert r.score > 0.99

    def test_no_overlap(self):
        r = BLEU().evaluate(case(expected="the cat sat"), "xyz abc def")
        assert r.score == 0.0

    def test_partial(self):
        r = BLEU(n=2).evaluate(case(expected="the quick brown fox"), "the quick fox")
        assert 0.0 < r.score < 1.0

    def test_no_expected_skips(self):
        r = BLEU().evaluate(case(), "anything")
        assert r.passed and r.metadata.get("skipped") and r.reason.startswith("[skipped]")


class TestROUGE:
    def test_perfect(self):
        r = ROUGE().evaluate(case(expected="hello world"), "hello world")
        assert r.score > 0.99

    def test_no_overlap(self):
        r = ROUGE().evaluate(case(expected="hello world"), "foo bar")
        assert r.score == 0.0

    def test_partial(self):
        r = ROUGE().evaluate(case(expected="the quick brown fox"), "the quick fox")
        assert 0.0 < r.score < 1.0


class TestStartsWith:
    def test_passes(self):
        assert StartsWith("Hello").evaluate(case(), "Hello world").passed

    def test_fails(self):
        assert not StartsWith("Hello").evaluate(case(), "world hello").passed

    def test_case_insensitive(self):
        assert StartsWith("hello").evaluate(case(), "HELLO world").passed


class TestToolCallAccuracy:
    def _make_trace(self, tool_names: list[str]) -> list[AgentStep]:
        return [
            AgentStep(tool_calls=[ToolCall(name=name)])
            for name in tool_names
        ]

    def test_all_correct(self):
        c = EvalCase(
            input="search and summarize",
            agent_trace=self._make_trace(["search", "summarize"]),
            expected_tool_calls=["search", "summarize"],
        )
        r = ToolCallAccuracy().evaluate(c, "done")
        assert r.score == 1.0

    def test_missing_tool(self):
        c = EvalCase(
            input="search and summarize",
            agent_trace=self._make_trace(["search"]),
            expected_tool_calls=["search", "summarize"],
        )
        r = ToolCallAccuracy().evaluate(c, "done")
        assert r.score == 0.5

    def test_no_trace_skips(self):
        c = EvalCase(input="test", expected_tool_calls=["search"])
        r = ToolCallAccuracy().evaluate(c, "done")
        assert r.passed and r.metadata.get("skipped") and r.reason.startswith("[skipped]")

    def test_ordered_correct(self):
        c = EvalCase(
            input="test",
            agent_trace=self._make_trace(["a", "b", "c"]),
            expected_tool_calls=["a", "b", "c"],
        )
        r = ToolCallAccuracy(require_order=True).evaluate(c, "done")
        assert r.score == 1.0

    def test_ordered_wrong_order(self):
        c = EvalCase(
            input="test",
            agent_trace=self._make_trace(["b", "a", "c"]),
            expected_tool_calls=["a", "b", "c"],
        )
        r = ToolCallAccuracy(require_order=True).evaluate(c, "done")
        assert r.score < 1.0
