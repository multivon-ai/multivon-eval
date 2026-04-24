import pytest
from llm_evals import EvalCase, ExactMatch, Contains, RegexMatch, JSONSchemaEval, NotEmpty, WordCount
from llm_evals.evaluators.deterministic import Latency


def case(input="test", expected=None, context=None):
    return EvalCase(input=input, expected_output=expected, context=context)


class TestExactMatch:
    def test_passes_on_match(self):
        r = ExactMatch().evaluate(case(expected="Paris"), "Paris")
        assert r.passed and r.score == 1.0

    def test_fails_on_mismatch(self):
        r = ExactMatch().evaluate(case(expected="Paris"), "London")
        assert not r.passed and r.score == 0.0

    def test_case_insensitive_by_default(self):
        r = ExactMatch().evaluate(case(expected="paris"), "PARIS")
        assert r.passed

    def test_case_sensitive(self):
        r = ExactMatch(case_sensitive=True).evaluate(case(expected="paris"), "PARIS")
        assert not r.passed

    def test_no_expected_fails(self):
        r = ExactMatch().evaluate(case(), "anything")
        assert not r.passed


class TestContains:
    def test_all_present(self):
        r = Contains(["red", "blue"]).evaluate(case(), "I see red and blue colors")
        assert r.passed and r.score == 1.0

    def test_partial_missing(self):
        r = Contains(["red", "blue", "green"]).evaluate(case(), "I see red and blue")
        assert not r.passed
        assert abs(r.score - 2/3) < 0.01

    def test_case_insensitive(self):
        r = Contains(["PARIS"]).evaluate(case(), "I visited paris last year")
        assert r.passed


class TestRegexMatch:
    def test_matches(self):
        r = RegexMatch(r"\d{4}").evaluate(case(), "The year was 2026")
        assert r.passed

    def test_no_match(self):
        r = RegexMatch(r"\d{4}").evaluate(case(), "No numbers here")
        assert not r.passed


class TestJSONSchemaEval:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }

    def test_valid_json(self):
        r = JSONSchemaEval(self.schema).evaluate(case(), '{"name": "Alice", "age": 30}')
        assert r.passed

    def test_invalid_json(self):
        r = JSONSchemaEval(self.schema).evaluate(case(), "not json")
        assert not r.passed

    def test_schema_violation(self):
        r = JSONSchemaEval(self.schema).evaluate(case(), '{"name": "Alice"}')
        assert not r.passed


class TestNotEmpty:
    def test_non_empty(self):
        r = NotEmpty().evaluate(case(), "hello")
        assert r.passed

    def test_empty(self):
        r = NotEmpty().evaluate(case(), "   ")
        assert not r.passed


class TestWordCount:
    def test_in_range(self):
        r = WordCount(min_words=1, max_words=10).evaluate(case(), "hello world")
        assert r.passed

    def test_too_long(self):
        r = WordCount(max_words=2).evaluate(case(), "one two three four")
        assert not r.passed

    def test_too_short(self):
        r = WordCount(min_words=5).evaluate(case(), "one two")
        assert not r.passed


class TestLatency:
    def test_under_limit(self):
        r = Latency(max_ms=1000).evaluate(case(), "response", latency_ms=500)
        assert r.passed

    def test_over_limit(self):
        r = Latency(max_ms=500).evaluate(case(), "response", latency_ms=1000)
        assert not r.passed
