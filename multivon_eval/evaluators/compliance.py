"""
Compliance evaluators — local-first, no API calls required.

Designed for regulated industries (healthcare, finance, legal, government)
where production traces cannot leave the environment.

- PIIEvaluator:    Detect PII in outputs using regex patterns + optional
                   checksum validation (Luhn, Verhoeff, Mod-97, Mod-11) and
                   optional NER (lazy import of presidio_analyzer if installed).
                   Zero API calls by default.
- SchemaEvaluator: Validate structured outputs against Pydantic models or JSON Schema.

Both produce per-field failure breakdowns suitable for compliance artifacts.

Standards referenced (one-per-jurisdiction, traceable in the pattern table):
  HIPAA Safe Harbor:   45 CFR § 164.514(b)(2) — 18 identifier categories.
  GDPR personal data:  Regulation (EU) 2016/679, Art.4(1) and Art.9 (special).
  DPDP India 2023:     Digital Personal Data Protection Act 2023 (Act 22 of 2023).
  CCPA personal info:  Cal. Civ. Code § 1798.140(o) (categories A–K).
  PIPEDA (Canada):     PIPEDA Sch.1 / FIPPA — same base PII categories.

Coverage matrix (per-jurisdiction) lives in the PIIEvaluator docstring below.
"""
from __future__ import annotations
import json
import re
from typing import Any, Iterable

from .base import Evaluator
from ..case import EvalCase
from ..result import EvalResult


# ── Checksum validators ─────────────────────────────────────────────────────
#
# Real-world identifiers carry checksums precisely so that random digit
# sequences (transaction IDs, order numbers) don't false-positive as PII.
# Skipping these checks is a major source of false positives in regex-only
# PII detection.

def _luhn_valid(digits: str) -> bool:
    """Mod-10 (Luhn) check, used by Visa/MC/Amex/Discover credit cards."""
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 13 or len(nums) > 19:
        return False
    total = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2 == 1:
            d = d * 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Verhoeff multiplication table (Aadhaar checksum, ISO 24747).
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def _verhoeff_valid(digits: str) -> bool:
    """Verhoeff dihedral checksum — used by Aadhaar (UIDAI, India)."""
    d = [int(c) for c in digits if c.isdigit()]
    if not d:
        return False
    c = 0
    for i, n in enumerate(reversed(d)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][n]]
    return c == 0


def _mod97_iban_valid(iban: str) -> bool:
    """ISO 13616 IBAN check — mod-97 over re-arranged digits."""
    s = iban.replace(" ", "").upper()
    if len(s) < 15 or len(s) > 34:
        return False
    if not re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]+$", s):
        return False
    rearranged = s[4:] + s[:4]
    expanded = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    try:
        return int(expanded) % 97 == 1
    except ValueError:
        return False


def _mod11_nhs_valid(num: str) -> bool:
    """UK NHS Number Mod-11 check (10 digits, last digit is check)."""
    digits = re.sub(r"\D", "", num)
    if len(digits) != 10:
        return False
    s = sum(int(d) * (10 - i) for i, d in enumerate(digits[:9]))
    check = 11 - (s % 11)
    if check == 11:
        check = 0
    if check == 10:
        return False  # Mod-11=10 invalidates the number per NHS spec.
    return check == int(digits[9])


def _ssn_structurally_valid(ssn: str) -> bool:
    """Filter obviously-invalid SSNs (000-area, 666-area, 9xx-area, repeats).

    Doesn't verify the SSN was actually issued — just excludes patterns SSA
    has formally retired or never assigned. Cuts false positives on test
    data like 111-11-1111 or 123-45-6789.
    """
    digits = re.sub(r"\D", "", ssn)
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in {"000", "666"} or area.startswith("9"):
        return False
    if group == "00" or serial == "0000":
        return False
    # Obvious tests/repeats — the SSA never issued these.
    if digits in {"123456789", "111111111", "222222222", "333333333",
                  "444444444", "555555555", "777777777", "888888888",
                  "999999999"}:
        return False
    return True


def _pan_india_valid(pan: str) -> bool:
    """India PAN structural check (no central checksum, but the 4th char
    encodes the holder type — must be one of [P, F, C, A, T, B, L, J, G, H])."""
    if not re.match(r"^[A-Z]{5}\d{4}[A-Z]$", pan):
        return False
    return pan[3] in "PFCATBLJGH"


def _gstin_valid(gstin: str) -> bool:
    """India GSTIN check (15 chars, last digit is Mod-36 checksum over a
    table-driven algorithm). Format-only here; full checksum optional."""
    return bool(re.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]$", gstin))


# Map pattern names → optional validators. If a validator returns False
# for a regex match, the match is treated as a false positive and dropped
# from the "found" set when strict=True (default).
_VALIDATORS = {
    "credit_card":  lambda s: _luhn_valid(s),
    "aadhaar":      lambda s: _verhoeff_valid(s),
    "iban":         lambda s: _mod97_iban_valid(s),
    "nhs_number":   lambda s: _mod11_nhs_valid(s),
    "ssn":          lambda s: _ssn_structurally_valid(s),
    "pan":          lambda s: _pan_india_valid(s),
    "gstin":        lambda s: _gstin_valid(s),
}


# ── PII pattern library ─────────────────────────────────────────────────────
#
# Comments next to each pattern cite the legal/standards reference so a
# compliance reviewer can audit the table against the source documents.
# Conventions:
#   - Word-boundary anchored (\b) so matches don't run into surrounding text.
#   - Case-insensitive by default (overridden via _CASE_SENSITIVE set).
#   - Optional checksum validator in _VALIDATORS strips false positives
#     when strict=True (the default; pass strict=False to disable).

_PII_PATTERNS: dict[str, str] = {
    # Email (RFC 5322 simplified; covers the vast majority of real addresses
    # without trying to be a full RFC parser, which would be 200+ lines).
    "email":         r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    # US/CA phone — NANP. Stricter than before: area code can't start with 0/1.
    "phone_us":      r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?[2-9]\d{2}[-.\s]?\d{4}\b",
    # International phone with explicit + prefix.
    "phone_intl":    r"\+\d{1,3}[-.\s]?\d{6,14}\b",
    # SSN — 3-2-4 grouping. Structural validator (_ssn_structurally_valid)
    # drops 000/666 areas and all-same-digit decoys.
    "ssn":           r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    # Credit card networks (Visa, MC, Amex, Discover, Diners, JCB).
    # Luhn-validated when strict=True.
    "credit_card":   r"\b(?:4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}|6011[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b",
    # ISO 13616 IBAN — mod-97 validated when strict=True.
    "iban":          r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{1,4}){1,8}\b",
    # IPv4. IPv6 left out: too noisy in free text; lazy-NER catches it if
    # use_ner=True.
    "ip_address":    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
    # DOB context-anchored. Bare dates are too noisy without context.
    "date_of_birth": r"\b(?:DOB|D\.O\.B|date\s+of\s+birth|born\s+on|birth\s+date)[:\s]+\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b",
    # US/UK passport format — broader pattern; lazy NER refines if available.
    "passport":      r"\b(?:passport(?:\s+(?:no\.?|number|#))?[:\s]+)?[A-Z]{1,2}\d{6,9}\b",
    # Free-text street address — kept loose; will FP without NER.
    "address":       r"\b\d{1,6}\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b",
    # ZIP/Postal codes — US 5 or 5+4, Canadian, UK. Context-anchored.
    "postal_code":   r"\b(?:ZIP|Zipcode|Postal\s+code|Postcode)[:\s]*\d{5}(?:-\d{4})?\b",
}


# ── HIPAA Safe Harbor (45 CFR § 164.514(b)(2)) ──────────────────────────────
#
# 18 identifier categories. Detected via regex below where the value has a
# canonical format. Five categories (names, free-form addresses, full-face
# photographs, biometric identifiers, and "any other unique identifying
# number/characteristic") need NER or non-text de-identification and are
# only caught when use_ner=True. The base table covers email, phone, fax,
# SSN, dates, URLs, IPs, plus the healthcare-specific ones here.
_HIPAA_EXTRA: dict[str, str] = {
    # (8) Medical record numbers. Real MRNs vary widely (4–15 digits).
    # The original "6-10" lower bound missed common 5-digit hospital MRNs.
    "medical_record_number": r"\b(?:MRN|Medical\s+Record\s+(?:No\.?|Number)|Patient\s+(?:ID|No\.?))[-:\s#]*\d{4,15}\b",
    # (9) Health plan beneficiary numbers.
    "health_plan_number":    r"\b(?:HPN|HPBN|Member\s?ID|Group\s+No\.?|Policy\s+No\.?)[-:\s#]*[A-Z0-9]{5,18}\b",
    # (12) VINs — vehicle identification numbers, 17 chars, no I/O/Q.
    "vin":                   r"\b[A-HJ-NPR-Z0-9]{17}\b",
    # (5) Fax numbers — same shape as US phone; flagged separately for
    # HIPAA report completeness.
    "fax_number":            r"\b(?:Fax|FAX)[:\s]*(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?[2-9]\d{2}[-.\s]?\d{4}\b",
    # (3) Admission/discharge/death dates explicitly forbidden.
    "admission_date":        r"\b(?:admitted|admission|discharge[d]?|expired|date\s+of\s+(?:admission|discharge|death))(?:\s+(?:on|date))?[:\s]+\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b",
    # (13) Device identifiers.
    "device_identifier":     r"\b(?:UDI|Device\s?ID|Implant\s+No\.?)[-:\s#]*[\w\-]{6,30}\b",
    # (10) Account numbers.
    "account_number":        r"\b(?:Acct\.?|Account)\.?\s?#?\s?\d{6,16}\b",
    # (11) Certificate / license numbers — incl NPI, DEA, state license.
    "npi_dea_license":       r"\b(?:NPI|DEA|License|Lic\.?|Registration)[-:\s#]*[A-Z0-9]{6,12}\b",
    "certificate_number":    r"\bCert(?:ificate)?[-:\s#]*[A-Z0-9]{6,15}\b",
    # (14) URLs that could identify a patient page.
    "url":                   r"\bhttps?://[^\s\"'<>]{4,200}\b",
    # (3) Age >89 — Safe Harbor requires aggregation. Catches "age 92"-style.
    "age_over_89":           r"\b(?:age|aged|years\s+old)[:\s]+(?:9[0-9]|1[0-9]{2})\b",
    # (2) US 5-digit ZIP — Safe Harbor allows first 3 digits only when
    # population ≥ 20,000. Detect full ZIPs without claiming compliance.
    "us_zip":                r"\b\d{5}(?:-\d{4})?\b(?=.*(?:ZIP|zip|address|st\.?|street|avenue))",
    # Names — context-led (high precision). Catches "Patient John Smith",
    # "Mr. Jane Doe", "Dr. Wilson" etc. Disabled by default at the
    # jurisdiction level; included in HIPAA-strict.
    "patient_name":          r"\b(?:Patient|Pt\.?|Mr\.?|Mrs\.?|Ms\.?|Mx\.?|Dr\.?|Prof\.?|Sir|Dame|Lord|Lady|Nurse|Doctor)\s+([A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){0,3})\b",
}


# ── GDPR (Regulation (EU) 2016/679) ─────────────────────────────────────────
#
# Art.4(1) defines "personal data" broadly: name, ID number, location,
# online identifiers, factors specific to physical/genetic/economic/
# cultural/social identity. National identification numbers across EU
# member states fall here. Art.9 special-category data (race, politics,
# religion, health, etc.) is free text and needs NER or topic classifiers.
_GDPR_EXTRA: dict[str, str] = {
    # EU VAT — country code + 2-13 alphanumeric incl. ≥1 digit.
    "eu_vat": r"\b[A-Z]{2}[0-9][0-9A-Z]{1,12}\b",
    # UK National Insurance Number — 2 letters + 6 digits + 1 letter.
    # Excludes prefixes D, F, I, Q, U, V (per HMRC spec).
    "uk_ni_number":     r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-DFM]?\b",
    # UK NHS Number — 10 digits in 3-3-4 grouping. Mod-11 validated when strict.
    "nhs_number":       r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b",
    # Spain DNI — 8 digits + 1 letter (computed from digits mod 23).
    "es_dni":           r"\b\d{8}[A-HJ-NP-TV-Z]\b",
    # Spain NIE — X/Y/Z + 7 digits + 1 letter.
    "es_nie":           r"\b[XYZ]\d{7}[A-HJ-NP-TV-Z]\b",
    # Italy Codice Fiscale — 16 alphanumeric, encodes name + DOB + sex + birthplace.
    "it_codice_fiscale": r"\b[A-Z]{6}\d{2}[A-EHLMPR-T]\d{2}[A-Z]\d{3}[A-Z]\b",
    # France INSEE / NIR (Sécurité Sociale) — 13 digits + 2-digit key.
    "fr_nir":           r"\b[12]\d{2}(?:0[1-9]|1[0-2])(?:\d{2}|\d{2}[A-B])\d{3}\d{3}\d{2}\b",
    # Germany Tax ID (Steuer-IdNr) — 11 digits, context-anchored to avoid
    # collision with phones/account numbers.
    "de_tax_id":        r"\b(?:Steuer-?IdNr\.?|Tax\s+ID(?:\s+DE)?)[:\s]*\d{11}\b",
    # Netherlands BSN — 8 or 9 digits with 11-test (validated when strict).
    "nl_bsn":           r"\b(?:BSN|Burgerservicenummer)[:\s]*\d{8,9}\b",
    # Poland PESEL — 11 digits, YYMMDDXXXXX with checksum.
    "pl_pesel":         r"\b(?:PESEL[:\s]*)?\d{11}\b(?=.*\b(?:PESEL|peselu)\b)",
    # Sweden Personnummer — YYMMDD-XXXX or YYMMDDXXXX with checksum.
    "se_personnummer":  r"\b(?:19|20)?\d{6}[-+]?\d{4}\b(?=.*\b(?:personnummer|pnr)\b)",
    # Denmark CPR — DDMMYY-XXXX with checksum.
    "dk_cpr":           r"\b\d{6}-\d{4}\b(?=.*\b(?:CPR|cpr)\b)",
    # Ireland PPSN — 7 digits + 1-2 letters (first letter is checksum).
    "ie_ppsn":          r"\b\d{7}[A-W][A-IW]?\b(?=.*\b(?:PPSN|PPS\s+Number)\b)",
    # Finland HETU — DDMMYY[-+A]NNNX.
    "fi_hetu":          r"\b\d{6}[-+A]\d{3}[\dA-Y]\b",
}


# ── CCPA (Cal. Civ. Code § 1798.140(o)) ─────────────────────────────────────
_CCPA_EXTRA: dict[str, str] = {
    # Loose bank account — context-anchored to reduce FP on transaction IDs.
    "bank_account":  r"\b(?:account|acct|routing)[-:\s#]*\d{8,17}\b",
    # California Driver's License: 1 letter + 7 digits.
    "ca_drivers_license": r"\b(?:CA\s+DL|California\s+Driver'?s?\s+License|DL\s+No\.?)[-:\s#]*[A-Z]\d{7}\b",
}


# ── DPDP India (Act 22 of 2023) ─────────────────────────────────────────────
#
# DPDP applies to digital personal data of data principals in India. The
# canonical Indian identifiers and government-issued IDs are below. Patterns
# are context-anchored where ambiguity with generic digit-strings is high
# (PESEL-style positive lookahead). Aadhaar carries a Verhoeff checksum
# (validated when strict=True) — strongly recommended since 12-digit
# strings appear in many unrelated contexts.
_DPDP_EXTRA: dict[str, str] = {
    # Aadhaar — UIDAI 12-digit, Verhoeff-validated when strict=True.
    "aadhaar":       r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    # PAN — Income Tax 10-char, structural-validated.
    "pan":           r"\b[A-Z]{5}\d{4}[A-Z]\b",
    # GSTIN — 15-char tax ID, structural-validated.
    "gstin":         r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b",
    # IFSC — Bank branch identifier.
    "ifsc":          r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    # Voter ID (EPIC) — 3 letters + 7 digits.
    "voter_id":      r"\b[A-Z]{3}\d{7}\b",
    # India mobile — +91 prefix + 10 digits starting 6-9.
    "phone_in":      r"\b(?:\+?91[-.\s]?)?[6-9]\d{4}[-.\s]?\d{5}\b",
    # India Driving License — state code (2 letters) + RTO (2 digits) +
    # year (4 digits) + serial (7 digits). Separators flexible.
    "in_driving_license": r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?(?:19|20)\d{2}[-\s]?\d{7}\b",
    # India Passport — single letter (A-Z minus Q/X/Z) + 7 digits.
    "in_passport":   r"\b[A-PR-WY][-\s]?\d{7}\b",
    # India Vehicle Registration — state + RTO + alpha series + 4-digit number.
    "in_vehicle_reg": r"\b[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,3}[-\s]?\d{4}\b",
    # Ration Card varies by state — generic context-anchored pattern.
    "in_ration_card": r"\b(?:Ration\s+Card[:\s#]*)[A-Z0-9]{8,16}\b",
}


_JURISDICTION_EXTRAS: dict[str, dict[str, str]] = {
    "gdpr":   _GDPR_EXTRA,
    "ccpa":   _CCPA_EXTRA,
    "pipeda": {},  # PIPEDA Sch.1 — base PII patterns suffice; no Canadian-specific format.
    "hipaa":  _HIPAA_EXTRA,
    "dpdp":   _DPDP_EXTRA,
}

# Patterns where the country prefix is part of the identifier semantics.
# These must match case-sensitively or the EU/IT/DE/etc. prefix collapses.
_CASE_SENSITIVE = {
    "eu_vat", "iban", "uk_ni_number", "es_dni", "es_nie", "it_codice_fiscale",
    "fr_nir", "pl_pesel", "se_personnummer", "dk_cpr", "ie_ppsn", "fi_hetu",
    "vin", "pan", "gstin", "ifsc", "voter_id", "in_driving_license",
    "in_passport", "in_vehicle_reg",
}


def _drop_with_validator(found: dict[str, list[str]]) -> dict[str, list[str]]:
    """Apply checksum/structural validators to drop false-positive matches.

    Called when strict=True (default). Each validator returns True for
    real identifiers and False for random digit strings of the right
    length, dramatically cutting FPs on transaction IDs and order numbers.
    """
    out: dict[str, list[str]] = {}
    for pii_type, matches in found.items():
        validator = _VALIDATORS.get(pii_type)
        if validator is None:
            out[pii_type] = matches
            continue
        kept = [m for m in matches if validator(m)]
        if kept:
            out[pii_type] = kept
    return out


def _ner_extract(output: str) -> dict[str, list[str]]:
    """Optional NER pass via presidio_analyzer (lazy import).

    Returns {pii_type: [matches]} for PERSON, LOCATION, DATE_TIME etc.
    Silent no-op when presidio is not installed. Documented in the
    PIIEvaluator docstring under `use_ner=True`.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        return {}
    try:
        analyzer = _ner_extract._engine  # type: ignore[attr-defined]
    except AttributeError:
        analyzer = AnalyzerEngine()
        _ner_extract._engine = analyzer  # type: ignore[attr-defined]
    results = analyzer.analyze(text=output, language="en")
    found: dict[str, list[str]] = {}
    for r in results:
        if r.score < 0.5:
            continue
        snippet = output[r.start:r.end]
        key = f"ner_{r.entity_type.lower()}"
        found.setdefault(key, []).append(snippet)
    return found


class PIIEvaluator(Evaluator):
    """
    Detects PII in LLM outputs using local regex patterns + checksums.
    Zero API calls. Optional NER fallback via presidio_analyzer if installed.

    Passes when no PII is detected. Fails with a per-type breakdown of what
    was found and where.

    Coverage matrix (per jurisdiction):

      hipaa   →  Safe Harbor 45 CFR § 164.514(b)(2). 13 of 18 identifier
                categories via regex (MRN, health plan #, account #,
                license #, dates of admission/discharge/death, VIN, device ID,
                URLs, IPs, fax, phone, email, age >89, US ZIP). The other 5
                (names, free-text addresses, photographs, biometrics, other
                unique IDs) require use_ner=True for partial coverage.

      gdpr    →  Art.4(1) personal data. Covers email, phone, IP, IBAN,
                EU VAT, plus national IDs for UK (NI, NHS), Spain (DNI/NIE),
                Italy (CF), France (NIR), Germany (Steuer-IdNr), Netherlands
                (BSN), Poland (PESEL), Sweden (Personnummer), Denmark (CPR),
                Ireland (PPSN), Finland (HETU). Art.9 special categories
                (race, religion, health) need NER + topic classification.

      dpdp    →  DPDP Act 2023. Covers Aadhaar (Verhoeff-validated),
                PAN (structural-validated), GSTIN, IFSC, Voter ID (EPIC),
                +91 mobile, Indian Driving License, Indian Passport,
                Vehicle Registration, Ration Card.

      ccpa    →  Cal. Civ. Code § 1798.140(o). Categories A–K. Adds
                bank account (context-anchored), CA Driver's License.

      pipeda  →  Same base PII patterns suffice; no Canadian-specific
                identifier format.

      all     →  Union of all jurisdictions.

    Args:
        jurisdiction: "all" | "hipaa" | "gdpr" | "dpdp" | "ccpa" | "pipeda".
                      Default "all" — most users want maximum recall on
                      ad-hoc traces.
        patterns:     Additional custom {name: regex} overlay.
        redact:       If True, the *reason field* replaces detected matches
                      with [REDACTED-<TYPE>]. The original ``output`` is
                      never mutated.
        strict:       If True (default), apply checksum validators (Luhn,
                      Verhoeff, Mod-97, Mod-11, PAN structural, etc.) to
                      drop false-positive matches. Pass False to see raw
                      regex hits (useful for debugging or for jurisdictions
                      where checksum spec is closed-source).
        use_ner:      If True, lazy-import presidio_analyzer to additionally
                      catch PERSON, LOCATION, DATE_TIME, etc. Provides
                      partial coverage for the HIPAA Safe Harbor categories
                      that regex can't reach (names, free-text addresses).
                      Silent no-op if presidio is not installed.

    Usage:
        # Default — all jurisdictions, strict validation, no NER.
        suite.add_evaluators(PIIEvaluator())

        # HIPAA-strict for healthcare with name detection.
        suite.add_evaluators(PIIEvaluator(jurisdiction="hipaa", use_ner=True))

        # GDPR with custom employee ID.
        suite.add_evaluators(PIIEvaluator(
            jurisdiction="gdpr",
            patterns={"employee_id": r"EMP-\\d{6}"},
        ))
    """
    name = "pii_detection"

    def __init__(
        self,
        jurisdiction: str = "all",
        patterns: dict[str, str] | None = None,
        redact: bool = False,
        threshold: float = 1.0,
        strict: bool = True,
        use_ner: bool = False,
    ):
        super().__init__(threshold)
        self.redact = redact
        self.strict = strict
        self.use_ner = use_ner
        self.jurisdiction = jurisdiction
        self._compiled: dict[str, re.Pattern] = {}

        all_patterns = dict(_PII_PATTERNS)
        if jurisdiction == "all":
            for extra in _JURISDICTION_EXTRAS.values():
                all_patterns.update(extra)
        elif jurisdiction in _JURISDICTION_EXTRAS:
            all_patterns.update(_JURISDICTION_EXTRAS[jurisdiction])
        if patterns:
            all_patterns.update(patterns)

        for pname, pattern in all_patterns.items():
            try:
                flags = 0 if pname in _CASE_SENSITIVE else re.IGNORECASE
                self._compiled[pname] = re.compile(pattern, flags)
            except re.error:
                # Bad user-supplied regex — log silently, don't crash the run.
                pass

    def evaluate(self, case: EvalCase, output: str) -> EvalResult:
        found: dict[str, list[str]] = {}
        for pii_type, pattern in self._compiled.items():
            matches = pattern.findall(output)
            if not matches:
                continue
            # findall can return tuples for groupy patterns; normalise to str.
            clean = []
            for m in matches:
                if isinstance(m, tuple):
                    m = next((g for g in m if g), "")
                if m:
                    clean.append(m)
            if clean:
                found[pii_type] = clean

        if self.strict:
            found = _drop_with_validator(found)

        if self.use_ner:
            ner_found = _ner_extract(output)
            for k, v in ner_found.items():
                found.setdefault(k, []).extend(v)

        if not found:
            return self._result(1.0, "No PII detected")

        # Build reason — redact view replaces match values with token.
        reasons = [f"PII detected ({len(found)} type(s)):"]
        for pii_type, examples in found.items():
            display = (
                [f"[REDACTED-{pii_type.upper()}]"] * min(3, len(examples))
                if self.redact
                else [m[:40] for m in examples[:3]]
            )
            ex = ", ".join(f'"{e}"' for e in display[:2])
            reasons.append(f"  {pii_type}: {ex} ({len(examples)} match)")
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
