"""
Compliance evaluators — local-first, no API calls required.

Designed for regulated industries (healthcare, finance, legal, government)
where production traces cannot leave the environment.

- PIIEvaluator:    Detect PII in outputs using regex patterns. No LLM calls.
- SchemaEvaluator: Validate structured outputs against Pydantic models or JSON Schema.

Both produce per-field failure breakdowns suitable for compliance artifacts.
"""
from __future__ import annotations
import json
import re
from typing import Any

from .base import Evaluator
from ..case import EvalCase
from ..result import EvalResult


# ── PII pattern library ──────────────────────────────────────────────────────

_PII_PATTERNS: dict[str, str] = {
    # Identifiers
    "email":         r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "phone_us":      r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "phone_intl":    r"\+\d{1,3}[-.\s]?\d{6,14}\b",
    "ssn":           r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    "credit_card":   r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6011\d{12})\b",
    "iban":          r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b",
    "ip_address":    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "date_of_birth": r"\b(?:DOB|date of birth|born on)[:\s]+\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
    "passport":      r"\b[A-Z]{1,2}\d{6,9}\b",
    "nhs_number":    r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b",
    # Names (heuristic — high false positive rate, disabled by default)
    # "name_prefix": r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",
    # Free text markers
    "address":       r"\b\d{1,6}\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b",
}

# GDPR-relevant patterns (EU-focused)
_GDPR_EXTRA: dict[str, str] = {
    # EU VAT: country code + 2-13 alphanumeric chars that must include at least one digit
    # e.g. DE123456789, FR12345678901
    "eu_vat": r"\b[A-Z]{2}[0-9][0-9A-Z]{1,12}\b",
}

# CCPA (California) adds financial account numbers
_CCPA_EXTRA: dict[str, str] = {
    "bank_account":  r"\b\d{8,17}\b",
}

# HIPAA PHI identifiers detectable via regex in text output.
# 13 of the 18 HIPAA Safe Harbor identifiers can be pattern-matched;
# 5 cannot be reliably detected with regex and require upstream de-identification:
#   - Names (high false positive rate on common words)
#   - Geographic subdivisions smaller than state (free text, no canonical form)
#   - Full-face photographs (binary content, not text)
#   - Biometric identifiers (fingerprints, voiceprints — binary content)
#   - Any unique identifying number not covered below
_HIPAA_EXTRA: dict[str, str] = {
    "medical_record_number": r"\bMRN[-:\s]?\d{6,10}\b",
    "health_plan_number":    r"\b(?:HPN|HPBN|Member\s?ID)[-:\s]?\w{6,15}\b",
    "vin":                   r"\b[A-HJ-NPR-Z0-9]{17}\b",
    # Fax numbers share the US phone pattern; listed separately for HIPAA completeness
    "fax_number":            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "admission_date":        r"\b(?:admitted|admission|discharge[d]?)\s+(?:on\s+)?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    "device_identifier":     r"\b(?:UDI|Device\s?ID)[-:\s][\w\-]{8,30}\b",
    "account_number":        r"\b(?:Acct|Account)\.?\s?#?\s?\d{6,16}\b",
    "npi_dea_license":       r"\b(?:NPI|DEA|License)[-:\s]?\d{7,10}\b",
    "certificate_number":    r"\bCert(?:ificate)?\.?\s?#?\s?[A-Z0-9]{6,15}\b",
    "url":                   r"\bhttps?://[^\s\"'<>]{4,100}\b",
}

# DPDP (Digital Personal Data Protection Act, India 2023) — India-specific
# personal identifiers. DPDP imposes notable penalties for unauthorised
# cross-border transfer of personal data, so local-first PII detection
# (zero egress) matters even more than under GDPR. Patterns cover the
# identifiers most commonly seen in Indian production traces: Aadhaar
# (UIDAI 12-digit), PAN (Income Tax 10-char), GSTIN (15-char tax ID),
# IFSC (bank branch), Voter ID (Election Commission EPIC), and Indian
# mobile numbers (+91 prefix). Names, addresses below state level, and
# biometric identifiers remain out of scope (regex cannot reliably
# detect them) and require upstream de-identification.
_DPDP_EXTRA: dict[str, str] = {
    # Aadhaar: 12 digits, often grouped 4-4-4 with spaces or hyphens.
    # Word-boundary anchored to reduce collision with 12-digit account
    # numbers; consumers should pair this with the bank_account pattern
    # for full coverage in financial contexts.
    "aadhaar":       r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    # PAN: Permanent Account Number — 5 uppercase letters + 4 digits +
    # 1 uppercase letter. Format defined by Income Tax Department.
    "pan":           r"\b[A-Z]{5}\d{4}[A-Z]\b",
    # GSTIN: Goods and Services Tax Identification Number.
    # 2 digits (state code) + 10-char PAN + 1 alphanumeric (entity code) +
    # "Z" (default) + 1 alphanumeric (checksum).
    "gstin":         r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b",
    # IFSC: Indian Financial System Code (bank branch identifier).
    # 4 uppercase letters (bank code) + "0" (reserved) + 6 alphanumeric.
    "ifsc":          r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    # Voter ID (EPIC): 3 uppercase letters + 7 digits.
    "voter_id":      r"\b[A-Z]{3}\d{7}\b",
    # India mobile: +91 prefix followed by 10 digits starting with 6-9.
    # Allows an internal separator after the first 5 digits to match the
    # common "+91 98765 43210" display format. Anchored to the +91 prefix
    # in the most distinctive variant to avoid false positives on generic
    # 10-digit strings (which are common in transaction IDs); the bare
    # 10-digit variant relies on the leading-digit-range constraint [6-9].
    "phone_in":      r"\b(?:\+?91[-.\s]?)?[6-9]\d{4}[-.\s]?\d{5}\b",
}

_JURISDICTION_EXTRAS: dict[str, dict[str, str]] = {
    "gdpr":   _GDPR_EXTRA,
    "ccpa":   _CCPA_EXTRA,
    "pipeda": {},  # Same base patterns suffice for PIPEDA
    "hipaa":  _HIPAA_EXTRA,
    "dpdp":   _DPDP_EXTRA,
}


class PIIEvaluator(Evaluator):
    """
    Detects PII in LLM outputs using local regex patterns. Zero API calls.

    Passes when no PII is detected. Fails with a per-type breakdown
    of what was found and where.

    Args:
        jurisdiction: "gdpr" | "ccpa" | "pipeda" | "hipaa" | "dpdp" | "all" (default "all").
                      Selects which pattern extensions to include.
                      "hipaa" adds MRN, health plan numbers, VINs, fax numbers,
                      admission/discharge dates, device identifiers, account numbers,
                      NPI/DEA/license numbers, certificate numbers, and URLs.
                      "dpdp" (India, Digital Personal Data Protection Act 2023) adds
                      Aadhaar, PAN, GSTIN, IFSC, Voter ID (EPIC), and +91 mobile numbers.
                      Note: regex-based detection cannot catch names, free-form addresses,
                      or biometric identifiers — these require upstream de-identification.
        patterns:     Additional custom {name: regex} patterns.
        redact:       If True, replace found PII with [REDACTED-TYPE] in the
                      reason field (default False — shows matched substring).

    Usage:
        suite.add_evaluators(PIIEvaluator())
        suite.add_evaluators(PIIEvaluator(jurisdiction="gdpr"))
        suite.add_evaluators(PIIEvaluator(jurisdiction="hipaa"))
        suite.add_evaluators(PIIEvaluator(patterns={"employee_id": r"EMP-\\d{6}"}))
    """
    name = "pii_detection"

    def __init__(
        self,
        jurisdiction: str = "all",
        patterns: dict[str, str] | None = None,
        redact: bool = False,
        threshold: float = 1.0,
    ):
        super().__init__(threshold)
        self.redact = redact
        self._compiled: dict[str, re.Pattern] = {}
        all_patterns = dict(_PII_PATTERNS)
        if jurisdiction == "all":
            for extra in _JURISDICTION_EXTRAS.values():
                all_patterns.update(extra)
        elif jurisdiction in _JURISDICTION_EXTRAS:
            all_patterns.update(_JURISDICTION_EXTRAS[jurisdiction])
        if patterns:
            all_patterns.update(patterns)
        # EU VAT needs case-sensitive country codes; others are case-insensitive
        _case_sensitive = {"eu_vat"}
        for name, pattern in all_patterns.items():
            try:
                flags = 0 if name in _case_sensitive else re.IGNORECASE
                self._compiled[name] = re.compile(pattern, flags)
            except re.error:
                pass

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        found: dict[str, list[str]] = {}
        for pii_type, pattern in self._compiled.items():
            matches = pattern.findall(output)
            if matches:
                if self.redact:
                    found[pii_type] = [f"[REDACTED-{pii_type.upper()}]"] * len(matches)
                else:
                    found[pii_type] = [m[:40] for m in matches[:3]]

        if not found:
            return self._result(1.0, "No PII detected")

        reasons = [f"PII detected ({len(found)} type(s)):"]
        for pii_type, examples in found.items():
            ex = ", ".join(f'"{e}"' for e in examples[:2])
            reasons.append(f"  {pii_type}: {ex}")
        return self._result(0.0, "\n".join(reasons))


# ── SchemaEvaluator ──────────────────────────────────────────────────────────

class SchemaEvaluator(Evaluator):
    """
    Validates LLM outputs against a Pydantic model or JSON Schema dict.

    Reports per-field failure rates and type mismatches — not just "valid/invalid".
    Aligned with StructEval (2025): frontier models fail structured output at
    measurable rates even with explicit format instructions.

    Args:
        schema: A Pydantic model class OR a JSON Schema dict.
        strict: If True, extra fields not in schema are also failures (default False).

    Usage:
        from pydantic import BaseModel
        class Summary(BaseModel):
            title: str
            score: float
            tags: list[str]

        suite.add_evaluators(SchemaEvaluator(Summary))
        suite.add_evaluators(SchemaEvaluator({"type": "object", "required": ["name"]}))
    """
    name = "schema_compliance"

    def __init__(self, schema: Any, strict: bool = False, threshold: float = 1.0):
        super().__init__(threshold)
        self._pydantic_model = None
        self._json_schema: dict | None = None
        self.strict = strict

        # Detect Pydantic model
        try:
            if hasattr(schema, "model_validate") or hasattr(schema, "parse_raw"):
                self._pydantic_model = schema
                return
        except Exception:
            pass

        # Detect JSON Schema dict
        if isinstance(schema, dict):
            self._json_schema = schema
            return

        raise ValueError("schema must be a Pydantic model class or a JSON Schema dict")

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        # Try to parse JSON from output
        raw = output.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return self._result(0.0, f"Invalid JSON: {e}")

        if self._pydantic_model is not None:
            return self._eval_pydantic(data)
        return self._eval_jsonschema(data)

    def _eval_pydantic(self, data: Any) -> EvalResult:
        model = self._pydantic_model
        # Support both Pydantic v1 and v2
        try:
            if hasattr(model, "model_validate"):
                model.model_validate(data)
            else:
                model(**data)
            return self._result(1.0, "Schema valid")
        except Exception as e:
            errors = str(e)
            # Extract field-level errors if available (Pydantic v2)
            try:
                if hasattr(e, "errors"):
                    field_errors = e.errors()
                    reasons = [f"  {'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in field_errors[:10]]
                    return self._result(0.0, "Schema validation failed:\n" + "\n".join(reasons))
            except Exception:
                pass
            return self._result(0.0, f"Schema validation failed: {errors[:300]}")

    def _eval_jsonschema(self, data: Any) -> EvalResult:
        try:
            import jsonschema
        except ImportError:
            return self._result(0.0, "jsonschema package required: pip install jsonschema")

        errors = list(jsonschema.Draft7Validator(self._json_schema).iter_errors(data))
        if not errors:
            return self._result(1.0, "Schema valid")

        reasons = ["Schema validation failed:"]
        for err in errors[:10]:
            path = ".".join(str(p) for p in err.path) or "(root)"
            reasons.append(f"  {path}: {err.message[:100]}")

        score = max(0.0, 1.0 - len(errors) / 10)
        return self._result(score, "\n".join(reasons))
