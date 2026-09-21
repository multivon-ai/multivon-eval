"""Routing for self-hosted and open-weights judges.

The on-prem path (ollama, vLLM/TGI through an OpenAI-compatible base_url, and
litellm) carries the compliance story but had almost no routing coverage. These
tests assert where a request is addressed and which settings survive the
translation. No server is contacted.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from multivon_eval import JudgeConfig
from multivon_eval.evaluators.llm_judge import _is_reasoning_model, _with_max_tokens
from multivon_eval.exceptions import JudgeUnavailable
from multivon_eval.judge import _ollama_as_openai, make_judge_call


# --- ollama reaches its own /v1, not OpenAI --------------------------------

def test_ollama_defaults_to_the_local_openai_compatible_endpoint(monkeypatch):
    monkeypatch.delenv('OLLAMA_HOST', raising=False)
    translated = _ollama_as_openai(JudgeConfig(provider='ollama', model='qwen2.5:7b').resolve())
    assert translated.provider == 'openai'
    assert translated.base_url == 'http://localhost:11434/v1'
    assert translated.model == 'qwen2.5:7b'


def test_ollama_host_env_is_honoured(monkeypatch):
    monkeypatch.setenv('OLLAMA_HOST', 'http://gpu-box.internal:11434')
    translated = _ollama_as_openai(JudgeConfig(provider='ollama', model='llama3.3').resolve())
    assert translated.base_url == 'http://gpu-box.internal:11434/v1'


def test_ollama_host_without_a_scheme_is_still_addressable(monkeypatch):
    monkeypatch.setenv('OLLAMA_HOST', 'gpu-box.internal:11434')
    translated = _ollama_as_openai(JudgeConfig(provider='ollama', model='llama3.3').resolve())
    assert translated.base_url == 'http://gpu-box.internal:11434/v1'


def test_explicit_base_url_beats_the_ollama_default(monkeypatch):
    monkeypatch.setenv('OLLAMA_HOST', 'http://ignored:11434')
    config = JudgeConfig(provider='ollama', model='llama3.3',
                         base_url='http://vllm.internal/v1').resolve()
    assert _ollama_as_openai(config).base_url == 'http://vllm.internal/v1'


def test_ollama_judge_call_goes_through_the_openai_path():
    config = JudgeConfig(provider='ollama', model='qwen2.5:7b', cache=False).resolve()
    with patch('multivon_eval.judge._sync_openai_call', return_value='Yes') as call:
        assert make_judge_call('Is this grounded?', config) == 'Yes'
    routed = call.call_args.args[1]
    assert routed.provider == 'openai' and routed.base_url.endswith('/v1')


def test_ollama_needs_no_litellm_extra():
    """The ollama route must not import litellm: that regression made
    --judge-provider ollama fail unless the extra happened to be installed."""
    config = JudgeConfig(provider='ollama', model='qwen2.5:7b', cache=False).resolve()
    with patch('multivon_eval.judge._sync_litellm_call',
               side_effect=AssertionError('ollama must not route through litellm')):
        with patch('multivon_eval.judge._sync_openai_call', return_value='No'):
            assert make_judge_call('grounded?', config) == 'No'


# --- self-hosted OpenAI-compatible servers ---------------------------------

def test_openai_provider_keeps_a_self_hosted_base_url():
    config = JudgeConfig(provider='openai', model='llama-3.3-70b-instruct',
                         base_url='https://vllm.internal/v1', cache=False).resolve()
    with patch('multivon_eval.judge._sync_openai_call', return_value='Yes') as call:
        make_judge_call('grounded?', config)
    assert call.call_args.args[1].base_url == 'https://vllm.internal/v1'


def test_per_call_token_override_preserves_the_on_prem_endpoint():
    """_with_max_tokens rebuilds the config; base_url must survive it."""
    config = JudgeConfig(provider='openai', model='llama-3.3-70b-instruct',
                         base_url='https://tgi.internal/v1', cache=False).resolve()
    narrowed = _with_max_tokens(config, 100)
    assert narrowed.base_url == 'https://tgi.internal/v1'
    assert narrowed.max_tokens == 100


# --- litellm ---------------------------------------------------------------

def test_litellm_routes_base_url_as_api_base():
    litellm = pytest.importorskip('litellm')
    config = JudgeConfig(provider='litellm', model='ollama/llama3.3',
                         base_url='http://localhost:11434', cache=False).resolve()
    with patch.object(litellm, 'completion') as completion:
        completion.return_value.choices = [type('C', (), {'message': type('M', (), {'content': 'Yes'})()})()]
        completion.return_value.usage = {}
        make_judge_call('grounded?', config)
    assert completion.call_args.kwargs['api_base'] == 'http://localhost:11434'


def test_litellm_absent_names_the_extra():
    config = JudgeConfig(provider='litellm', model='bedrock/anthropic.claude-3', cache=False).resolve()
    with patch.dict('sys.modules', {'litellm': None}):
        with pytest.raises(JudgeUnavailable, match=r"multivon-eval\[litellm\]"):
            make_judge_call('grounded?', config)


# --- open-weights judges that reason before answering ----------------------

@pytest.mark.parametrize('model', [
    'gemma-4-31b-it', 'gemma-4-26b-a4b-it', 'gemma-3-27b-it',
    'qwen3-32b', 'qwq-32b', 'deepseek-r1', 'magistral-small',
    'gpt-5.5', 'o3',
])
def test_reasoning_judges_get_a_floored_token_budget(model):
    """A 100-token cap truncates these mid-thought and yields no verdict.

    Measured for gemma-4: at 100 tokens it returns its working, cut off and
    unparseable; at 2048 it returns "Yes".
    """
    assert _is_reasoning_model(model)
    judge = JudgeConfig(provider='google', model=model).resolve()
    assert _with_max_tokens(judge, 100).max_tokens == 2048


@pytest.mark.parametrize('model', [
    'claude-haiku-4-5-20251001', 'gpt-4o-mini', 'llama-3.3-70b-instruct', 'mistral-large',
])
def test_plain_judges_keep_the_small_cap(model):
    assert not _is_reasoning_model(model)
    judge = JudgeConfig(provider='openai', model=model).resolve()
    assert _with_max_tokens(judge, 100).max_tokens == 100


def test_an_explicitly_larger_budget_is_never_lowered():
    judge = JudgeConfig(provider='google', model='gemma-4-31b-it').resolve()
    assert _with_max_tokens(judge, 4096).max_tokens == 4096


def test_unparseable_verdict_error_names_the_token_cap():
    """A name list always lags, so the failure has to be self-diagnosing."""
    from multivon_eval.evaluators.llm_judge import _qag_eval

    judge = JudgeConfig(provider='openai', model='some-local-reasoner', cache=False).resolve()
    with patch('multivon_eval.evaluators.llm_judge._call', return_value='Let me think about this'):
        with pytest.raises(JudgeUnavailable, match='reasons before answering'):
            _qag_eval([('Is it grounded?', True)], 'Context', judge)


def test_extraction_budget_scales_above_the_flat_floor():
    """Claim extraction must reason AND emit an array; the yes/no call need not.

    Measured on gemma-4-31b-it, HaluEval Summarization item 33: at a 2048
    ceiling the reply is 7,772 characters of working with no JSON array and the
    evaluator aborts; at 4096 the same item returns a clean three-claim array.
    """
    judge = JudgeConfig(provider='google', model='gemma-4-31b-it').resolve()
    assert _with_max_tokens(judge, 100).max_tokens == 2048     # yes/no, unchanged
    assert _with_max_tokens(judge, 512).max_tokens == 4096     # claim extraction
    assert _with_max_tokens(judge, 8000).max_tokens == 8000    # never multiplied upward
