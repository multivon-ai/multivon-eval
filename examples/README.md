# multivon-eval examples

Four reproducible case studies. Each script is self-contained — no shared
utilities, no relative imports. Run with `python <name>.py` after setting the
required environment variables.

```bash
pip install multivon-eval pdfhell

# Anthropic key needed for the LLM-judge examples (1 and 3)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI key needed for the contract / vision example (2)
export OPENAI_API_KEY=sk-proj-...

# 4 needs no key — runs entirely offline (regex)
```

| # | Script                                          | Evaluators                                            | API needed                | Cost     |
|---|-------------------------------------------------|-------------------------------------------------------|---------------------------|----------|
| 1 | `01_rag_insurance_faithfulness.py`              | `Faithfulness`, `Relevance`                           | Anthropic claude-haiku-4-5 | <$0.05  |
| 2 | `02_contract_pdfhell_trap.py`                   | `pdfhell.score_case`                                  | OpenAI gpt-4o (vision)    | <$0.30  |
| 3 | `03_support_qa_multi_evaluator.py`              | `Faithfulness`, `Relevance`, `CheckEvaluator`         | Anthropic claude-haiku-4-5 | <$0.15  |
| 4 | `04_pii_medical_records.py`                     | `PIIEvaluator`                                        | none — regex only         | $0      |

Each script:

- Exits 0 on overall pass, exits 1 if any case fails its threshold (mirror real CI gates).
- Saves a full results JSON next to the script as `<name>_output.json`.
- Prints clean, terminal-friendly output you can paste into a PR.

The captured outputs from a real run are saved as `0X_output.txt` next to each
script — these are exactly what the multivon.ai /examples page renders.
