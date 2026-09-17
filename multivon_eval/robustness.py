"""Bind task-specific transformation validation to immutable case evidence.

This is an oracle-validation boundary, not a transformation/search engine.
Validators are trusted caller code; a digest does not prove oracle correctness.
The initial profile supports explicit string answers and two relations.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from .case import EvalCase
from .case_manifest import CaseManifest, canonical_json, case_from_dict, case_to_dict, digest


@dataclass(frozen=True)
class OracleVerdict:
    """Validator assertion with independently derived answers and evidence.

    ``valid=None`` means unavailable/ambiguous. ``False`` means the transformation
    violates the declared task contract. Never use the target model's answer to
    establish its own expected answer. Record review identity or code provenance
    in evidence; the library cannot establish independence from a callback.
    """

    valid: bool | None
    reason: str
    base_expected: str | None = None
    variant_expected: str | None = None
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VariantValidation:
    _json: str

    def __post_init__(self):
        data = self.data
        claimed = data.pop('digest', None)
        if (data.get('schema') != 'multivon.variant-validation/v1'
                or data.get('status') not in {'valid', 'invalid', 'unknown'}
                or digest(data) != claimed):
            raise ValueError('Invalid variant validation schema or digest')

    @property
    def data(self) -> dict:
        return json.loads(self._json)

    @property
    def status(self) -> str:
        return self.data['status']

    def manifest(self, name: str, *, split: str = 'development') -> CaseManifest:
        """Export both cases only after validation; preserve their source group.

        Split selection is caller supplied and cannot attest that sources were
        untouched. Keep all variants of a source together in the final manifest.
        """
        data = self.data
        if data['status'] != 'valid':
            raise ValueError('Only valid variants can become scored cases')
        base, variant = [case_from_dict(data[key]) for key in ('base', 'candidate')]
        variant.expected_output = data['verdict']['variant_expected']
        variant.reference_output = None
        variant.metadata['multivon_robustness'] = {
            'validation_digest': data['digest'], 'relation': data['relation'],
            'parent_case_id': base.case_id, 'contract': data['contract'],
        }
        generation = variant.metadata.get('generation')
        if isinstance(generation, dict) and generation.get('kind') == 'mutation':
            generation['oracle_status'] = 'validated'
        return CaseManifest(name, [base, variant],
                            splits={split: [base.case_id, variant.case_id]},
                            provenance={'variant_validation': data})


def validate_variant(
    base: EvalCase, candidate: EvalCase, *, relation: str, contract: str,
    validator: Callable[[EvalCase, EvalCase], OracleVerdict],
) -> VariantValidation:
    """Validate a candidate and retain valid, invalid and unknown outcomes.

    The callback receives detached cases. Input/label/provenance mutation by the
    callback is an error. A valid invariant requires equal independently derived
    answers; a counterfactual requires different answers. Directional confidence
    relations and state/tool oracles are outside this explicit-answer profile.
    """
    if relation not in {'invariant', 'counterfactual'}:
        raise ValueError('relation must be invariant or counterfactual')
    if not isinstance(contract, str) or not contract.strip():
        raise ValueError('Supply a versioned task/validator contract')
    if (not base.case_id or not candidate.case_id or base.case_id == candidate.case_id
            or not base.source_id or base.source_id != candidate.source_id):
        raise ValueError('Use distinct explicit case IDs and the same explicit source_id')
    if not isinstance(base.expected_output, str):
        raise TypeError('The base requires an explicit expected_output')
    original, proposed = case_to_dict(base), case_to_dict(candidate)
    body = {'schema': 'multivon.variant-validation/v1', 'relation': relation,
            'contract': contract, 'base': original, 'candidate': proposed}
    b, c = case_from_dict(original), case_from_dict(proposed)
    try:
        result = validator(b, c)
        if case_to_dict(b) != original or case_to_dict(c) != proposed:
            raise ValueError('Validator mutated its inputs')
        if not isinstance(result, OracleVerdict):
            raise TypeError('Validator must return OracleVerdict')
        if result.valid is not None and type(result.valid) is not bool:
            raise ValueError('valid must be bool or None')
        if not isinstance(result.reason, str) or not result.reason.strip():
            raise ValueError('Validator must explain its decision')
        if not isinstance(result.evidence, dict):
            raise TypeError('Validator evidence must be a portable JSON object')
        verdict = json.loads(canonical_json(asdict(result)))
        body['verdict'] = verdict
        status = {True: 'valid', False: 'invalid', None: 'unknown'}[result.valid]
        issues = []
        if result.valid:
            if not result.evidence:
                issues.append('No supporting validation evidence')
            if (not isinstance(result.base_expected, str)
                    or not isinstance(result.variant_expected, str)):
                issues.append('Both independently derived string answers are required')
            elif result.base_expected != base.expected_output:
                issues.append('Base oracle disagrees with validator')
            elif candidate.expected_output not in {None, result.variant_expected}:
                issues.append('Candidate label disagrees with validator')
            elif (result.base_expected == result.variant_expected) != (relation == 'invariant'):
                issues.append('Derived answers contradict the declared relation')
            if issues:
                status = 'invalid'
        body.update(status=status, issues=issues)
    except Exception as exc:  # noqa: BLE001 — retain failures from caller-supplied code
        body.update(status='unknown', issues=[str(exc)],
                    error_type=type(exc).__name__, verdict=None)
    return VariantValidation(canonical_json({**body, 'digest': digest(body)}))
