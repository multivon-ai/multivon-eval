"""Legacy benchmark runners must not manufacture measurements or provenance."""
import io
import json
from types import SimpleNamespace

import pytest

from benchmarks import run_calibration_v2 as candidate
from benchmarks import run_threshold_calibration as sweep


def test_upstream_download_failure_never_becomes_a_fixture(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('offline')
    monkeypatch.setattr(sweep.urllib.request, 'urlopen', fail)
    with pytest.raises(OSError, match='offline'):
        sweep._load_halueval_qa(1)
    with pytest.raises(OSError, match='offline'):
        sweep._load_halueval_summ(1)


def test_upstream_pairs_share_source_and_revision_is_pinned(monkeypatch):
    row = {'knowledge': 'context', 'question': 'question', 'right_answer': 'yes', 'hallucinated_answer': 'no'}
    urls = []
    def response(url, **kwargs):
        urls.append(url)
        return io.BytesIO(json.dumps(row).encode())
    monkeypatch.setattr(sweep.urllib.request, 'urlopen', response)
    pairs = sweep._load_halueval_qa(1)
    assert [r['label'] for r in pairs] == [0, 1]
    assert pairs[0]['source_id'] == pairs[1]['source_id']
    assert sweep.HALUEVAL_REVISION in urls[0] and '/main/' not in urls[0]
    with pytest.raises(ValueError, match='incomplete'):
        sweep._load_halueval_qa(2)
    row['right_answer'] = ''
    with pytest.raises(ValueError, match='scoring fields'):
        sweep._load_halueval_qa(1)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1, True, None, '0.5'])
def test_invalid_scores_abort_instead_of_becoming_midpoint(value):
    with pytest.raises(RuntimeError, match='no valid measurement'):
        sweep._collect_scores([{'label': 0}], lambda *args: value, None, workers=1)


def test_judge_error_aborts_sweep():
    def error(*args):
        raise RuntimeError('fixture failure')
    with pytest.raises(RuntimeError, match='Calibration aborted'):
        sweep._collect_scores([{'label': 0}], error, None, workers=1)
    with pytest.raises(ValueError, match='Skipped'):
        sweep._measured_score(SimpleNamespace(score=0.0, metadata={'skipped': True}))


def test_candidate_contains_actual_content_hash_and_no_unverified_model_alias():
    items = [{'label': 0, 'output': 'yes'}, {'label': 1, 'output': 'no'}]
    scores = [(0.9, 0), (0.1, 1)]
    fitted = sweep._best_threshold(scores)
    result = candidate._provenance_entry(evaluator='hallucination', judge_model='fixture/model',
        sweep_result=fitted, items=items, measured_at='fixture')
    assert result['n'] == 2 and result['dataset_hash'] == candidate.digest(items)
    assert result['judge_aliases'] == []
    assert 'no held-out estimate' in result['notes']
    items[0]['output'] = 'changed'
    assert result['dataset_hash'] != candidate.digest(items)
    assert fitted['n'] == 2 and 'development' in fitted['scope']


def test_zero_f1_uses_an_actually_evaluated_threshold():
    fitted = sweep._best_threshold([(1.0, 0), (1.0, 1)])
    assert fitted['optimal'] == fitted['sweep'][0]
    with pytest.raises(ValueError):
        sweep._best_threshold([])
