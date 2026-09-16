# multivon-eval examples

Six numbered case studies plus the integration demonstrations below. Each script is self-contained — no shared
utilities, no relative imports. Run with `python <name>.py` after setting the
required environment variables.

```bash
pip install multivon-eval pdfhell

# Anthropic key needed for the LLM-judge examples (1, 3, and 6)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI key needed for the contract / vision example (2)
export OPENAI_API_KEY=sk-proj-...

# 4 and 5 need no key — they run entirely offline
```

| # | Script                                          | Evaluators                                            | API needed                | Cost     |
|---|-------------------------------------------------|-------------------------------------------------------|---------------------------|----------|
| 1 | `01_rag_insurance_faithfulness.py`              | `Faithfulness`, `Relevance`                           | Anthropic claude-haiku-4-5 | <$0.05  |
| 2 | `02_contract_pdfhell_trap.py`                   | `pdfhell.score_case`                                  | OpenAI gpt-4o (vision)    | <$0.30  |
| 3 | `03_support_qa_multi_evaluator.py`              | `Faithfulness`, `Relevance`, `CheckEvaluator`         | Anthropic claude-haiku-4-5 | <$0.15  |
| 4 | `04_pii_medical_records.py`                     | `PIIEvaluator`                                        | none — regex only         | $0      |
| 5 | `05_staleness_drift.py`                         | `staleness` (baseline → stamp → CHANGED → CI gate)    | none — static analysis    | $0      |
| 6 | `06_simulate_personas.py`                       | `simulate`, conversation evaluators, goal judge       | Anthropic claude-haiku-4-5 | <$0.05  |

Examples 1–4 are reproducible scored runs. Each one:

- Exits 0 on overall pass, exits 1 if any case fails its threshold (mirror real CI gates).
- Saves a full results JSON next to the script as `<name>_output.json`.
- Prints clean, terminal-friendly output you can paste into a PR.

Their captured outputs are saved as `0X_output.txt` next to each script — these
are exactly what the multivon.ai `/examples` page renders.

Example 5 demonstrates the complete staleness lifecycle and intentionally exits
0 after showing the nested CI gate's exit code. Example 6 runs an adaptive
persona simulation and gates on goal completion; it prints its report but does
not create a results JSON or a captured output file.

## Development integration demonstrations

These examples require a checkout of `main`; the review and OTel previews are
not in PyPI 0.18.0. They use synthetic fixtures and make no model API calls.

| Script | What it checks | Installation |
|---|---|---|
| `review_saved_trials.py` | Native Label Studio task/review exchange and missing coverage | `pip install -e .` |
| `calibrate_reviewed_scores.py` | Frozen development fitting and source-disjoint held-out analysis | `pip install -e '.[review]'` |
| `otel_evidence.py` | Official SDK/OTLP trace and evaluation-event round trip; optional real MCP stdio | `pip install -e '.[otel]'` |

Run each with `--output-dir` pointing to a new directory. The OTel example's
optional `--mcp-python` selects a Python executable with multivon-mcp installed;
its calling environment also needs the official `mcp` client package.

For actual environment state and partial-failure checks, run from the repository
root with `pip install -e '.[gymnasium]'`:

```bash
python -m benchmarks.industrial.gymnasium_ledger --output-dir ledger-outcomes
python -m benchmarks.industrial.analyze_environment ledger-outcomes
```

This reuses the document study's SQLite handler, retains the databases and
observations, and compares answer-only checks with the explicit outcome contract.
See the [validation and critique](../benchmarks/industrial/ENVIRONMENT_VALIDATION.md).
