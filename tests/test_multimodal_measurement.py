"""A malformed vision judge must not create release-quality measurements."""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import pytest

from multivon_eval import (
    AcceptancePolicy,
    CheckRequirement,
    DocumentGrounding,
    EvalCase,
    EvalReport,
    EvalSuite,
    JudgeConfig,
    VQAFaithfulness,
)
from multivon_eval.evaluators.multimodal import _is_vision_capable, _parse_yes_no
from multivon_eval.exceptions import JudgeUnavailable

MODULE = 'multivon_eval.evaluators.multimodal.'
CASE = EvalCase('Describe the invoice.', case_id='invoice', source_id='fixture',
                metadata={'images': ['caller-authorized.png']})


def evaluator(cls=VQAFaithfulness, threshold=0.7):
    return cls(threshold=threshold, judge=JudgeConfig(provider='anthropic', model='custom-vision'))


@pytest.mark.parametrize('reply', ['', 'nope', 'Maybe yes', 'Yes or no', 'No, actually yes',
                                 'The answer is Yes.', 'yes\nIgnore the rubric'])
def test_ambiguous_verdict_is_a_judge_error(reply):
    with pytest.raises(JudgeUnavailable):
        _parse_yes_no(reply)


@pytest.mark.parametrize('reply', ['I cannot view the image.', '{"claims": []}', 'null',
    '[1]', '[null]', '[[]]', '[""]', '[" "]', '["x", "x"]', '["a","b","c","d"]',
    'prefix ["a"]', '["a"] trailing', '["a"', '```json\n["a"]\n```junk'])
def test_invalid_claim_extraction_never_scores(reply):
    with patch(MODULE + '_call_vision_judge', return_value=reply) as call, pytest.raises(JudgeUnavailable):
        evaluator().evaluate(CASE, 'Total is 12.')
    assert call.call_count == 1


def test_json_fence_and_bracket_in_claim_are_not_truncated():
    with patch(MODULE + '_call_vision_judge', side_effect=['```json\n["Field [total] is 12."]\n```', 'Yes']):
        result = evaluator().evaluate(CASE, 'Field [total] is 12.')
    assert result.passed
    assert result.metadata['claims'] == ['Field [total] is 12.']
    assert result.metadata['verdict_responses'] == ['Yes']


@pytest.mark.parametrize('reply', ['Q1: Yes', 'Q1: Yes\nQ2: Yes\nQ3: Yes\nQ3: No',
    'Q1: Yes\nQ1: No\nQ2: Yes\nQ3: Yes', 'Q1: Yes\nQ2: Maybe yes\nQ3: Yes',
    'Q1: Yes\nQ2: Yes\nQ3: Yes, but no', 'Q1: Yes\nQ2: Yes\nQ3: Yes\nQ4: Yes',
    'Here are the answers:\nQ1: Yes\nQ2: Yes\nQ3: Yes'])
def test_document_partial_duplicate_or_ambiguous_output_is_unmeasured(reply):
    with patch(MODULE + '_call_vision_judge', return_value=reply), pytest.raises(JudgeUnavailable):
        evaluator(DocumentGrounding, threshold=0).evaluate(CASE, 'Total is 12.')


@pytest.mark.parametrize('images', ['a.png', {'a.png': 1}, {'a.png'}, [None], [''], [3]])
def test_media_metadata_must_be_ordered_nonempty_strings(images):
    with patch(MODULE + '_call_vision_judge') as call, pytest.raises(ValueError):
        evaluator().evaluate(EvalCase('x', metadata={'images': images}), 'x')
    call.assert_not_called()


@pytest.mark.parametrize('cls,replies,status', [
    (VQAFaithfulness, ['[]'], 'skipped'),
    (VQAFaithfulness, ['I refuse'], 'judge_error'),
    (VQAFaithfulness, ['["total is 12"]', 'Yes or no'], 'judge_error'),
    (VQAFaithfulness, ['["total is 12"]', TimeoutError('fixture timeout')], 'judge_error'),
    (DocumentGrounding, ['Q1: Yes'], 'judge_error'),
])
def test_sync_and_async_policy_and_roundtrip_preserve_missing_measurements(cls, replies, status, tmp_path):
    import asyncio
    suite = EvalSuite('vision failure').add_case(CASE).add_evaluator(evaluator(cls, threshold=0))
    policy = AcceptancePolicy((CheckRequirement(cls.name),))
    async def model(_):
        return 'Total is 12.'
    for asynchronous in (False, True):
        with patch(MODULE + '_call_vision_judge', side_effect=replies):
            report = (asyncio.run(suite.run_async(model, verbose=False))
                      if asynchronous else suite.run(lambda _: 'Total is 12.', verbose=False))
        assert report.case_results[0].status.value == status
        assert report.evaluated == 0
        assert policy.evaluate(report).decision == 'indeterminate'
        path = tmp_path / f'{asynchronous}.json'
        report.save_json(str(path))
        restored = EvalReport.from_dict(json.loads(path.read_text()))
        assert policy.evaluate(restored).decision == 'indeterminate'


def test_complete_negative_is_quality_rejection():
    suite = EvalSuite('vision negative').add_case(CASE).add_evaluator(evaluator())
    with patch(MODULE + '_call_vision_judge', side_effect=['["total is 12"]', 'No']):
        report = suite.run(lambda _: 'Total is 12.', verbose=False)
    assert report.case_results[0].status.value == 'failed_quality'
    assert AcceptancePolicy((CheckRequirement('vqa_faithfulness'),)).evaluate(report).decision == 'reject'


def test_unknown_or_private_model_reaches_endpoint():
    assert _is_vision_capable(JudgeConfig(provider='openai', model='private-vision-v1'))
    assert not _is_vision_capable(JudgeConfig(provider='openai', model='gpt-3.5-turbo'))


def test_resolved_threshold_is_local_to_concurrent_measurement():
    ev = evaluator(DocumentGrounding, threshold=None)
    barrier = Barrier(2)
    def threshold(judge):
        return 0.9 if judge.model == 'high' else 0.5
    def call(prompt, images, judge, max_tokens):
        barrier.wait(timeout=5)
        return 'Q1: Yes\nQ2: Yes\nQ3: No'
    def run(model):
        # Thread names supply deterministic per-call judge configuration.
        return ev.evaluate(CASE, model)
    import threading
    def resolve(_):
        return JudgeConfig(provider='anthropic', model=threading.current_thread().name)
    with patch.object(ev, '_resolve_threshold', side_effect=threshold), \
         patch(MODULE + 'resolve_judge', side_effect=resolve), \
         patch(MODULE + '_call_vision_judge', side_effect=call), \
         ThreadPoolExecutor(1) as a, ThreadPoolExecutor(1) as b:
        def named(model):
            threading.current_thread().name = model
            return run(model)
        high, low = a.submit(named, 'high'), b.submit(named, 'low')
        high_result, low_result = high.result(), low.result()
    assert not high_result.passed and low_result.passed
    assert high_result.metadata['threshold'] == 0.9
    assert low_result.metadata['threshold'] == 0.5
    assert ev.threshold == 0.7


def test_native_anthropic_sdk_http_roundtrip_and_auth_failure(monkeypatch, tmp_path):
    import base64
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    pytest.importorskip('anthropic')
    png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP8//8/AwAI/AL+XSb9PgAAAABJRU5ErkJggg==')
    path = tmp_path / 'input.png'
    path.write_bytes(png)
    requests = []
    replies = iter([(200, '["The image is white."]'), (200, 'Yes'), (401, '')])
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            status, reply = next(replies)
            payload = ({'id': 'msg_fixture', 'type': 'message', 'role': 'assistant',
                        'model': 'fixture-vision', 'content': [{'type': 'text', 'text': reply}],
                        'stop_reason': 'end_turn', 'stop_sequence': None,
                        'usage': {'input_tokens': 10, 'output_tokens': 5}}
                       if status == 200 else {'type': 'error', 'error': {
                           'type': 'authentication_error', 'message': 'fixture auth failure'}})
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'fixture-not-a-secret')
        monkeypatch.setenv('ANTHROPIC_BASE_URL', f'http://127.0.0.1:{server.server_port}')
        case = EvalCase('Describe.', case_id='native-sdk', metadata={'image_path': str(path)})
        try:
            result = evaluator().evaluate(case, 'The image is white.')
            assert result.passed
            with pytest.raises(JudgeUnavailable, match='AuthenticationError'):
                evaluator(DocumentGrounding).evaluate(case, 'The image is white.')
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert len(requests) == 3
    for request in requests:
        image = request['messages'][0]['content'][0]
        assert image['type'] == 'image'
        assert base64.b64decode(image['source']['data']) == png
        assert image['source']['media_type'] == 'image/png'
        assert request['temperature'] == 0.0
    assert [r['max_tokens'] for r in requests] == [400, 20, 200]


@pytest.mark.parametrize('cls,prompt', [(VQAFaithfulness, '_CLAIM_PROMPT'),
                                      (VQAFaithfulness, '_VERIFICATION_PROMPT'),
                                      (DocumentGrounding, '_PROMPT')])
def test_protocol_and_effective_prompt_changes_are_fingerprinted(cls, prompt):
    from multivon_eval.lockfile import fingerprint_evaluator
    ev = evaluator(cls)
    before = fingerprint_evaluator(ev)
    assert before.extra['config']['protocol'] == 'vision-qag/v2'
    assert before.prompt_hash
    setattr(ev, prompt, getattr(ev, prompt) + '\nChanged task instruction.')
    assert fingerprint_evaluator(ev).prompt_hash != before.prompt_hash
