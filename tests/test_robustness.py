from dataclasses import replace

import pytest

from multivon_eval import EvalCase
from multivon_eval.case_manifest import CaseManifest
from multivon_eval.robustness import OracleVerdict, validate_variant


def pair():
    return (EvalCase('amount=1.25', '1.25', case_id='base', source_id='invoice'),
            EvalCase('amount=2.25', case_id='variant', source_id='invoice'))


def validate(base=None, variant=None, **kwargs):
    b, v = pair()
    return validate_variant(base or b, variant or v, relation=kwargs.pop('relation', 'counterfactual'),
                            contract='amount/v1', validator=kwargs.pop('validator', lambda *_: OracleVerdict(
                                True, 'parsed both amounts', '1.25', '2.25', {'parser': 'decimal/v1'})), **kwargs)


def test_validated_manifest_roundtrip_and_detached_evidence():
    b, v = pair()
    result = validate(b, v)
    b.input = 'changed'
    result.data['candidate']['input'] = 'changed'
    manifest = result.manifest('test')
    restored = CaseManifest.from_dict(manifest.manifest)
    assert restored.digest == manifest.digest
    assert [c.expected_output for c in restored.cases] == ['1.25', '2.25']
    assert restored.cases[0].input == 'amount=1.25'
    assert restored.cases[1].input == 'amount=2.25'
    assert {c.source_id for c in restored.cases} == {'invoice'}
    with pytest.raises(ValueError, match='leakage'):
        CaseManifest('bad', restored.cases, splits={'development': ['base'], 'heldout': ['variant']})


@pytest.mark.parametrize('verdict,relation', [
    (OracleVerdict(True, 'wrong base', '9.25', '2.25', {'check': 1}), 'counterfactual'),
    (OracleVerdict(True, 'not invariant', '1.25', '2.25', {'check': 1}), 'invariant'),
    (OracleVerdict(True, 'not changed', '1.25', '1.25', {'check': 1}), 'counterfactual'),
    (OracleVerdict(True, 'no evidence', '1.25', '2.25'), 'counterfactual'),
    (OracleVerdict(True, 'no answers', evidence={'check': 1}), 'counterfactual'),
    (OracleVerdict(False, 'invalid transform', evidence={'check': 1}), 'counterfactual'),
])
def test_invalid_oracles_cannot_be_exported(verdict, relation):
    result = validate(validator=lambda *_: verdict, relation=relation)
    assert result.status == 'invalid'
    assert result.data['verdict']['reason'] == verdict.reason
    with pytest.raises(ValueError, match='Only valid'):
        result.manifest('bad')


def test_mislabelled_candidate_is_not_silently_corrected():
    _, v = pair()
    result = validate(variant=replace(v, expected_output='1.25'))
    assert result.status == 'invalid'
    assert 'Candidate label' in result.data['issues'][0]


@pytest.mark.parametrize('callback', [
    lambda *_: OracleVerdict(None, 'needs review'),
    lambda *_: OracleVerdict('yes', 'bad type'),
    lambda *_: OracleVerdict(True, '', '1.25', '2.25', {'a': 1}),
    lambda *_: OracleVerdict(True, 'nonportable', '1.25', '2.25', {'a': float('nan')}),
    lambda *_: {},
])
def test_unknown_or_malformed_verdicts_stay_unknown(callback):
    result = validate(validator=callback)
    assert result.status == 'unknown'
    with pytest.raises(ValueError, match='Only valid'):
        result.manifest('bad')


def test_callback_error_and_mutation_are_preserved():
    def crash(*_):
        raise RuntimeError('oracle offline')
    assert validate(validator=crash).data['error_type'] == 'RuntimeError'

    def mutate(b, _):
        b.input = 'overwritten'
        return OracleVerdict(True, 'edited', '1.25', '2.25', {'check': 1})
    result = validate(validator=mutate)
    assert result.status == 'unknown'
    assert result.data['base']['input'] == 'amount=1.25'


@pytest.mark.parametrize('change', [{'case_id': None}, {'case_id': 'base'}, {'source_id': 'other'}])
def test_missing_identity_and_changed_groups_rejected(change):
    _, v = pair()
    with pytest.raises(ValueError, match='distinct explicit'):
        validate(variant=replace(v, **change))


def test_valid_invariant_and_empty_answer():
    b, v = pair()
    b.expected_output = ''
    result = validate(b, v, relation='invariant', validator=lambda *_: OracleVerdict(
        True, 'explicit empty answer', '', '', {'check': 'task/v1'}))
    assert result.status == 'valid'
    assert all(c.expected_output == '' for c in result.manifest('empty').cases)
