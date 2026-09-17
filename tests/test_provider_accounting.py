"""Reconcile frozen live evidence and fail closed on incomplete measurements."""
import json
import tarfile
from pathlib import Path

import pytest

from multivon_eval import (Costs, EvalGateFailure, EvalReport, ModelPricing, ProviderUsage,
                           account_provider_events, capture_provider_events, provider_events)
from multivon_eval.case_manifest import digest
from multivon_eval.provider_accounting import native_tokens


def artifact(name):
    path = Path(__file__).parents[1] / 'benchmarks/industrial/results/provider-capture-2026-09-17/evidence.tar.gz'
    with tarfile.open(path) as archive:
        return json.load(archive.extractfile(name))


def fixed_price(request, response):
    return {'cost_usd': 0.01, 'provenance': {'source': 'test fixture, not real pricing'}}


def test_live_events_reconcile_all_roles_and_roundtrip_without_new_calls():
    events = artifact('events.json')
    costs = account_provider_events(events, price_estimator=fixed_price,
                                  coverage_declaration='Frozen protocol: all four native calls are observed')
    assert costs.complete and not costs.evidence_gaps
    assert costs.total_calls == 4 and costs.total_tokens == 68
    assert costs.total_input_tokens == 48 and costs.total_output_tokens == 20
    assert costs.total_cost_usd == pytest.approx(0.04)
    assert [row['role'] for row in costs.evidence['requests']].count('target') == 2
    report = EvalReport.from_dict(artifact('report.json'))
    report.costs = costs
    restored = EvalReport.from_dict(json.loads(report.to_json()))
    assert restored.costs.to_dict() == costs.to_dict()
    restored.assert_budget(max_total_tokens=68, max_total_cost_usd=0.04)
    with pytest.raises(EvalGateFailure):
        restored.assert_budget(max_total_tokens=67)


def test_observed_usage_is_not_automatically_complete():
    costs = account_provider_events(artifact('events.json'), price_estimator=fixed_price)
    assert costs.recorded_cost_usd == pytest.approx(0.04)
    assert costs.total_cost_usd is None and not costs.complete
    report = EvalReport('partial', [], costs=costs)
    with pytest.raises(EvalGateFailure, match='indeterminate'):
        report.assert_budget(max_total_cost_usd=100, max_total_tokens=1000)


def test_report_snapshots_preserve_open_scope_and_regrade_spend_exclusion():
    report = EvalReport.from_dict(artifact('report.json'))
    costs = account_provider_events(provider_events(report), coverage_declaration='Test declaration')
    assert costs.total_calls == 2 and not costs.complete
    assert any('no closure' in issue for issue in costs.evidence_gaps)
    regraded = EvalReport.from_dict(artifact('regraded.json'))
    assert account_provider_events(provider_events(regraded)).total_calls == 0


@pytest.mark.parametrize('remove', ['http_response', 'capture_started', 'capture_finished', 'operation_started', 'operation_finished'])
def test_incomplete_events_cannot_be_overridden_by_declaration(remove):
    events = artifact('events.json')
    index = next(i for i, e in enumerate(events) if e['kind'] == remove)
    del events[index]
    costs = account_provider_events(events, price_estimator=fixed_price, coverage_declaration='All calls')
    assert costs.evidence_gaps and not costs.complete and costs.total_cost_usd is None


@pytest.mark.parametrize('mutation', ['missing_usage', 'negative_tokens', 'failure', 'wrong_binding', 'stream'])
def test_unknown_attempts_never_disappear_or_become_zero(mutation):
    events = artifact('events.json')
    event = next(e for e in events if e['kind'] == 'http_response')
    if mutation == 'missing_usage':
        event['response']['usage'] = None
    elif mutation == 'negative_tokens':
        event['response']['usage']['input_tokens'] = -1
    elif mutation == 'failure':
        event['response']['status_code'] = 429
    elif mutation == 'wrong_binding':
        event['operation_id'] = 'different'
    else:
        event['response']['capture_gap'] = 'Streaming body/usage not captured'
    event.pop('digest')
    event['digest'] = digest(event)
    costs = account_provider_events(events, price_estimator=fixed_price, coverage_declaration='All calls')
    assert costs.total_calls == 4
    assert not costs.complete and costs.total_cost_usd is None


def test_deduplicate_copies_but_reject_corruption():
    events = artifact('events.json')
    assert account_provider_events(events + events).total_calls == 4
    events[0]['labels']['forged'] = True
    with pytest.raises(ValueError, match='digest'):
        account_provider_events(events)


def test_zero_is_known_only_after_closed_capture_and_coverage_declaration():
    with capture_provider_events() as capture:
        pass
    costs = account_provider_events(capture.snapshot()['events'], coverage_declaration='Entirely local fixed target')
    assert costs.complete and costs.total_cost_usd == 0
    assert not account_provider_events([], coverage_declaration='Empty').complete


@pytest.mark.parametrize('costs', [None, Costs(), Costs.from_dict({'by_model': []})])
def test_absent_or_legacy_costs_fail_budget_gates(costs):
    report = EvalReport('missing', [], costs=costs)
    for limit in [{'max_total_tokens': 1000}, {'max_total_cost_usd': 10}, {'max_avg_cost_per_case_usd': 10}]:
        with pytest.raises(EvalGateFailure, match='indeterminate'):
            report.assert_budget(**limit)
    report.assert_budget()


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, '10'])
def test_invalid_thresholds_and_prices_are_rejected(value):
    with pytest.raises(ValueError):
        EvalReport('invalid', []).assert_budget(max_total_tokens=value)
    with pytest.raises(ValueError):
        ModelPricing(value, 1)
    with pytest.raises(ValueError):
        ProviderUsage('test', 'test', cost_usd=value)


def test_submicrodollar_cost_is_not_rounded_to_free():
    costs = Costs([ProviderUsage('test', 'small', calls=1, cost_usd=1e-9)],
                  scope='run_provider_usage', coverage_declaration='Fixture')
    assert costs.total_cost_usd == 1e-9
    with pytest.raises(EvalGateFailure):
        EvalReport('small', [], costs=costs).assert_budget(max_total_cost_usd=0)


def test_native_token_dimensions_are_not_double_counted():
    assert native_tokens('anthropic', {'input_tokens': 11, 'output_tokens': 3,
        'cache_read_input_tokens': 7, 'cache_creation_input_tokens': 5}) == (23, 3)
    assert native_tokens('openai', {'prompt_tokens': 11, 'completion_tokens': 3, 'total_tokens': 14,
        'prompt_tokens_details': {'cached_tokens': 7}, 'completion_tokens_details': {'reasoning_tokens': 2}}) == (11, 3)
    assert native_tokens('google', {'promptTokenCount': 11, 'candidatesTokenCount': 3,
        'thoughtsTokenCount': 2, 'cachedContentTokenCount': 7, 'totalTokenCount': 16}) == (11, 5)
    with pytest.raises(ValueError, match='inconsistent'):
        native_tokens('openai', {'prompt_tokens': 11, 'completion_tokens': 3, 'total_tokens': 99})


def test_pricing_failure_does_not_destroy_token_evidence():
    def broken(*args):
        raise LookupError('No verified tariff')
    costs = account_provider_events(artifact('events.json'), price_estimator=broken, coverage_declaration='All calls')
    assert costs.complete and costs.total_tokens == 68 and costs.total_cost_usd is None
    assert all('LookupError' in r['pricing_error'] for r in costs.evidence['requests'])
    EvalReport('tokens', [], costs=costs).assert_budget(max_total_tokens=68)
    with pytest.raises(EvalGateFailure, match='pricing'):
        EvalReport('dollars', [], costs=costs).assert_budget(max_total_cost_usd=10)


@pytest.mark.parametrize('estimate', [
    {'cost_usd': -1, 'provenance': {'test': True}},
    {'cost_usd': float('nan'), 'provenance': {'test': True}},
    {'cost_usd': True, 'provenance': {'test': True}},
    {'cost_usd': 0, 'provenance': {}},
])
def test_invalid_estimate_cannot_become_known_cost(estimate):
    costs = account_provider_events(artifact('events.json'), price_estimator=lambda *_: estimate,
                                   coverage_declaration='All native calls')
    assert costs.total_cost_usd is None and costs.total_tokens == 68
    assert all(row['pricing_error'] for row in costs.evidence['requests'])


def test_empty_report_has_no_average_cost_denominator():
    costs = Costs(scope='run_provider_usage', coverage_declaration='No provider calls')
    with pytest.raises(EvalGateFailure, match='denominator'):
        EvalReport('empty', [], costs=costs).assert_budget(max_avg_cost_per_case_usd=1)
