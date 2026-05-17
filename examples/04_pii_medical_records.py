"""PII detection over synthetic medical record snippets. Zero API calls.

Purpose:       Run multivon_eval.PIIEvaluator against 5 medical record snippets — some
               clean, some leaking SSN / email / phone / MRN. The evaluator is regex-only,
               so this works offline, costs nothing, and produces a reproducible verdict.
Runtime:       <1s. Cost: $0. No API keys required.
Output shape:  Per-case PII findings, JSON dump. Exits 1 if any record contains PII.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from multivon_eval import EvalCase, EvalSuite
from multivon_eval.evaluators.compliance import PIIEvaluator


# 5 synthetic medical record snippets. Records 1, 3, 5 are clean.
# Records 2 and 4 contain leaked PII that PIIEvaluator should flag.
RECORDS = [
    {
        "id": "MR1",
        "label": "clean",
        "text": (
            "Patient presented with mild dehydration following gastroenteritis. "
            "Vitals stable. Started on oral rehydration solution. "
            "Discharge planned within 24 hours pending tolerance of oral intake."
        ),
    },
    {
        "id": "MR2",
        "label": "leaks_ssn_phone",
        "text": (
            "Patient John Doe (SSN 123-45-6789) was admitted on 03/14/2025 with "
            "chest pain. Spouse contact: 415-555-0182. Discharged after observation; "
            "follow-up scheduled with cardiology."
        ),
    },
    {
        "id": "MR3",
        "label": "clean",
        "text": (
            "Routine post-op check after laparoscopic cholecystectomy. "
            "Incisions healing well, no signs of infection. "
            "Pain controlled with acetaminophen as needed."
        ),
    },
    {
        "id": "MR4",
        "label": "leaks_email_mrn",
        "text": (
            "MRN-4471829 — patient reports persistent migraine 3x/week. "
            "Family physician notified via patient@example.com. "
            "Trial of topiramate 25mg nightly. Reassess in 4 weeks."
        ),
    },
    {
        "id": "MR5",
        "label": "clean",
        "text": (
            "Pediatric well-child visit. Growth on expected curve. "
            "Vaccinations up to date. Anticipatory guidance provided. "
            "No acute concerns from caregiver."
        ),
    },
]


def main() -> int:
    cases = [EvalCase(input=r["id"], metadata={"label": r["label"]}) for r in RECORDS]

    suite = EvalSuite("PII Detection — Medical Records", model_id="static-records")
    suite.add_cases(cases)
    # "hipaa" jurisdiction adds MRN, health plan numbers, fax, admission dates, etc.
    suite.add_evaluators(PIIEvaluator(jurisdiction="hipaa"))

    def model_fn(record_id: str) -> str:
        for r in RECORDS:
            if r["id"] == record_id:
                return r["text"]
        return ""

    report = suite.run(model_fn)

    here = Path(__file__).parent
    out = here / "04_pii_medical_records_output.json"
    report.save_json(str(out))
    print(f"\nSaved full results -> {out.name}")

    print("\n=== Per-record PII findings ===")
    leak_count = 0
    for cr in report.case_results:
        pii = next((er for er in cr.results if er.evaluator == "pii_detection"), None)
        if pii is None:
            continue
        rid = cr.case_input
        if pii.passed:
            print(f"  [CLEAN] {rid}  {pii.reason}")
        else:
            leak_count += 1
            print(f"  [LEAK ] {rid}")
            for line in pii.reason.splitlines()[1:]:
                print(f"           {line.strip()}")

    print(
        f"\nFinal: {leak_count}/{len(RECORDS)} record(s) contain PII."
        " (Regex-only — no API calls, $0 cost, fully deterministic.)"
    )
    if leak_count:
        # Mirror real CI behavior: any PII leak means non-zero exit.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
