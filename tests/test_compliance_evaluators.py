from pydantic import BaseModel

from multivon_eval import EvalCase
from multivon_eval.evaluators.compliance import PIIEvaluator, SchemaEvaluator


class Invoice(BaseModel):
    invoice_id: str
    amount: float


def case():
    return EvalCase(input="Validate output")


class TestPIIEvaluator:
    def test_pii_content_fails_for_each_jurisdiction(self):
        outputs = {
            "hipaa": "Patient MRN 123456 and admitted on 01/02/2024.",
            "gdpr": "Contact me at jane@example.com and VAT DE123456789.",
            "ccpa": "Email jane@example.com and bank account 123456789012.",
            "pipeda": "Call me at 415-555-1212 or email jane@example.com.",
            "dpdp": "Aadhaar 2341 2341 2346 and PAN BNZPM2501F linked to GSTIN 27BNZPM2501F1Z5.",
            "all": "Patient MRN 123456, VAT DE123456789, and email jane@example.com.",
        }
        for jurisdiction, output in outputs.items():
            result = PIIEvaluator(jurisdiction=jurisdiction).evaluate(case(), output)
            assert not result.passed
            assert result.score == 0.0
            assert "PII detected" in result.reason

    def test_non_pii_content_passes_for_each_jurisdiction(self):
        for jurisdiction in ["hipaa", "gdpr", "ccpa", "pipeda", "dpdp", "all"]:
            result = PIIEvaluator(jurisdiction=jurisdiction).evaluate(
                case(),
                "The quarterly summary contains aggregate metrics only.",
            )
            assert result.passed
            assert result.score == 1.0

    def test_redaction_masks_matches(self):
        result = PIIEvaluator(jurisdiction="all", redact=True).evaluate(case(), "Email jane@example.com")
        assert "[REDACTED-EMAIL]" in result.reason


class TestPIIEvaluatorHIPAASpecific:
    """Test HIPAA-specific PII patterns that extend the base set."""

    def test_mrn_pattern_detected(self):
        result = PIIEvaluator(jurisdiction="hipaa").evaluate(
            case(), "Patient record: MRN-123456 was admitted yesterday."
        )
        assert not result.passed
        assert result.score == 0.0

    def test_mrn_colon_variant_detected(self):
        result = PIIEvaluator(jurisdiction="hipaa").evaluate(
            case(), "Medical Record Number: MRN:9876543 for the patient."
        )
        assert not result.passed

    def test_url_detected_under_hipaa(self):
        # HIPAA treats URLs as potential PHI (patient portal links, etc.)
        result = PIIEvaluator(jurisdiction="hipaa").evaluate(
            case(), "See the patient portal at https://portal.hospital.com/patient/123"
        )
        assert not result.passed

    def test_npi_pattern_detected(self):
        result = PIIEvaluator(jurisdiction="hipaa").evaluate(
            case(), "Prescribing physician NPI:1234567890"
        )
        assert not result.passed

    def test_device_id_detected(self):
        # The device_identifier pattern requires a single separator char after UDI/Device ID.
        # Use no space between "UDI:" and the identifier string.
        result = PIIEvaluator(jurisdiction="hipaa").evaluate(
            case(), "Implanted device UDI:ABCD1234EFGH"
        )
        assert not result.passed

    def test_clean_clinical_summary_passes(self):
        result = PIIEvaluator(jurisdiction="hipaa").evaluate(
            case(),
            "The patient was prescribed standard analgesics for post-operative pain management.",
        )
        assert result.passed
        assert result.score == 1.0


class TestPIIEvaluatorGDPRSpecific:
    """Test GDPR-specific PII patterns."""

    def test_eu_vat_detected(self):
        result = PIIEvaluator(jurisdiction="gdpr").evaluate(
            case(), "Invoice issued to DE123456789 (German VAT registrant)."
        )
        assert not result.passed

    def test_email_detected_under_gdpr(self):
        result = PIIEvaluator(jurisdiction="gdpr").evaluate(
            case(), "Data subject email: hans.mueller@example.de"
        )
        assert not result.passed

    def test_clean_aggregate_data_passes_gdpr(self):
        result = PIIEvaluator(jurisdiction="gdpr").evaluate(
            case(), "Total EU users: 42,000. Retention period: 24 months."
        )
        assert result.passed


class TestPIIEvaluatorDPDPSpecific:
    """Test DPDP (India, Digital Personal Data Protection Act 2023) patterns."""

    def test_aadhaar_detected(self):
        result = PIIEvaluator(jurisdiction="dpdp").evaluate(
            case(), "Customer KYC linked Aadhaar 2341 2341 2346 on file."
        )
        assert not result.passed
        assert "aadhaar" in result.reason.lower()

    def test_pan_detected(self):
        result = PIIEvaluator(jurisdiction="dpdp").evaluate(
            case(), "Quoted PAN BNZPM2501F on invoice."
        )
        assert not result.passed
        assert "pan" in result.reason.lower()

    def test_gstin_detected(self):
        result = PIIEvaluator(jurisdiction="dpdp").evaluate(
            case(), "Vendor GSTIN 27BNZPM2501F1Z5 registered in Maharashtra."
        )
        assert not result.passed
        assert "gstin" in result.reason.lower()

    def test_ifsc_detected(self):
        result = PIIEvaluator(jurisdiction="dpdp").evaluate(
            case(), "Refund routed via IFSC SBIN0001234 to the listed account."
        )
        assert not result.passed
        assert "ifsc" in result.reason.lower()

    def test_voter_id_detected(self):
        result = PIIEvaluator(jurisdiction="dpdp").evaluate(
            case(), "Voter ID ABC1234567 verified for the registration."
        )
        assert not result.passed
        assert "voter_id" in result.reason.lower()

    def test_india_mobile_with_prefix_detected(self):
        result = PIIEvaluator(jurisdiction="dpdp").evaluate(
            case(), "Call back at +91 98765 43210 within 24 hours."
        )
        assert not result.passed
        assert "phone_in" in result.reason.lower()

    def test_clean_aggregate_data_passes_dpdp(self):
        result = PIIEvaluator(jurisdiction="dpdp").evaluate(
            case(), "Total Indian users: 1.2M. Avg session: 14 minutes."
        )
        assert result.passed


class TestSchemaEvaluator:
    def test_valid_pydantic_model_passes(self):
        result = SchemaEvaluator(Invoice).evaluate(case(), '{"invoice_id":"INV-1","amount":42.5}')
        assert result.passed
        assert result.score == 1.0

    def test_invalid_pydantic_model_fails(self):
        result = SchemaEvaluator(Invoice).evaluate(case(), '{"invoice_id":"INV-1","amount":"oops"}')
        assert not result.passed
        assert result.score == 0.0
        assert "Schema validation failed" in result.reason

    def test_pydantic_missing_required_field_fails(self):
        # amount is required — omitting it must fail
        result = SchemaEvaluator(Invoice).evaluate(case(), '{"invoice_id":"INV-1"}')
        assert not result.passed
        assert result.score == 0.0
        assert "Schema validation failed" in result.reason

    def test_valid_json_schema_passes(self):
        schema = {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["label", "score"],
        }
        result = SchemaEvaluator(schema).evaluate(case(), '{"label":"spam","score":0.9}')
        assert result.passed

    def test_invalid_json_schema_fails(self):
        schema = {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["label", "score"],
        }
        result = SchemaEvaluator(schema).evaluate(case(), '{"label":"spam"}')
        assert not result.passed
        assert "Schema validation failed" in result.reason

    def test_json_schema_missing_required_field_fails(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        result = SchemaEvaluator(schema).evaluate(case(), '{"name":"Alice"}')
        assert not result.passed
        assert "Schema validation failed" in result.reason

    def test_invalid_json_string_fails(self):
        result = SchemaEvaluator(Invoice).evaluate(case(), "this is not json at all")
        assert not result.passed
        assert "Invalid JSON" in result.reason

    def test_markdown_fenced_json_is_accepted(self):
        result = SchemaEvaluator(Invoice).evaluate(
            case(),
            '```json\n{"invoice_id":"INV-2","amount":12.0}\n```',
        )
        assert result.passed


class TestPhoneIntlSpacedGroups:
    """Spaced international numbers must be detected (re-verification
    finding 2026-06-13: '+44 7911 123456' scored 'No PII detected')."""

    def test_spaced_and_contiguous_intl_numbers_flagged(self):
        ev = PIIEvaluator()
        for number in ("+44 7911 123456", "+447911123456",
                       "+91 98765 43210", "+1-212-555-0198",
                       "+33 6 12 34 56 78"):
            res = ev.evaluate(EvalCase(input="contact"), f"Call me at {number}.")
            assert res.score == 0.0, f"not flagged: {number}"

    def test_arithmetic_and_versions_not_flagged_as_intl_phone(self):
        ev = PIIEvaluator()
        for text in ("the answer is 2+2", "x = 1+23 here", "upgrade to +1.2.3"):
            res = ev.evaluate(EvalCase(input="q"), text)
            assert res.score == 1.0, f"false positive on: {text!r}"
