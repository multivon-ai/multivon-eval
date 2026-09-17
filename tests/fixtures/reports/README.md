# Report migration fixtures

`legacy-v1.json` is a handwritten, schema-less legacy envelope with pass,
model-error and skipped cases. It has no identities, trials or lock. Loading it
must not invent those missing measurements.

`published-0.18.0.json` is frozen output from the installed PyPI 0.18.0 package
on Python 3.12.13, generated outside the checkout on September 17, 2026. The
environment's `multivon_eval.__file__` resolved to site-packages. It contains
three saved-output cases: pass, quality failure and unavailable reference.
It retains the released suite lock and raw trial digests unchanged.
No target or judge requests were made.

Generation code (run against the published package, not development source):

```python
from multivon_eval import EvalCase, EvalSuite, ExactMatch, MaxLatency

suite = EvalSuite("published-0.18.0-saved-fixture").add_evaluators(
    ExactMatch(), MaxLatency(100),
)
report = suite.run_on_cases([
    (EvalCase("pass", "ok", case_id="pass"), "ok"),
    (EvalCase("fail", "ok", case_id="fail"), "wrong"),
    (EvalCase("unknown", case_id="unknown"), ""),
], verbose=False)
report.save_json("published-0.18.0.json")
```

Do not regenerate historical fixtures when changing the reader. Add a new
fixture for a new format, plus a test for supported migration or explicit
rejection. Timestamps differ if the example is rerun.
