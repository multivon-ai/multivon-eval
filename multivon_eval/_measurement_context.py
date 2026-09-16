"""Request-local controls for fresh judge measurements."""
from contextlib import contextmanager
from contextvars import ContextVar

_cache_bypass: ContextVar[bool] = ContextVar('multivon_judge_cache_bypass', default=False)


def judge_cache_bypassed() -> bool:
    return _cache_bypass.get()


@contextmanager
def fresh_judge_measurement():
    """Do not reuse or replace cached verdicts while measuring judge variability."""
    token = _cache_bypass.set(True)
    try:
        yield
    finally:
        _cache_bypass.reset(token)
