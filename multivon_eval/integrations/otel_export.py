"""Emit standard GenAI evaluation log events through a caller-owned OTel logger."""
from __future__ import annotations

import math

from ..result import EvalReport
from ..trials import trial_integrity_issues
from .otel import GENAI_CONVENTIONS_REVISION, GENAI_PROFILE, OtelTrace


def emit_evaluation_events(report: EvalReport, logger, *, include_explanations: bool = False) -> int:
    """Emit trial status and saved grader events without replacing missing scores with zero.

    The application configures OTel providers/processors/exporters and their
    limits. The return count means submitted to the logger, not delivered to a
    backend. Flush/shutdown acknowledgement remains the application's job.
    Native target context is attached when imported OTLP evidence is available.
    Otherwise events are unparented rather than attached to an unrelated ambient
    request. Content/explanations are opt-in; trial digests identify local evidence.
    """
    try:
        from opentelemetry._logs import LogRecord, SeverityNumber
        from opentelemetry.context import Context
    except ImportError as exc:
        raise ImportError('Install multivon-eval[otel] for evaluation event export') from exc
    if type(include_explanations) is not bool:
        raise ValueError('include_explanations must be boolean')
    pending = []
    for case in report.case_results:
        if not case.trials or case.evidence_error or trial_integrity_issues(case):
            raise ValueError('Event export requires intact saved trial evidence')
        for trial in case.trials:
            data = trial.data
            trace_id, span_id = None, None
            upstream = data.get('upstream', {})
            if upstream.get('format') in {'otlp/protobuf', 'otlp/json'}:
                native = OtelTrace.from_dict(upstream['evidence'])
                trace_id = int(native.trace_id, 16)
                selected, _ = native.selected()
                output_id = upstream['output_span_id']
                if output_id not in {r['span'].span_id.hex() for r in selected}:
                    raise ValueError('Evaluation parent is outside its retained native trace')
                span_id = int(output_id, 16)
            common = {'multivon.case.id': data['case_id'], 'multivon.trial.digest': trial.digest,
                      'multivon.trial.status': data['status'], 'multivon.trial.recorded_at': data['recorded_at'],
                      'multivon.report.has_evidence_issues': bool(report.evidence_issues),
                      'multivon.report.evidence_issue_count': len(report.evidence_issues)}
            pending.append(LogRecord(context=Context(), trace_id=trace_id, span_id=span_id,
                event_name='multivon.evaluation.trial', attributes=common, severity_number=SeverityNumber.INFO))
            for result in data['evaluators']:
                metadata = result.get('metadata', {})
                measured = (data['status'] in {'passed', 'failed_quality'} and
                            not metadata.get('skipped') and not metadata.get('error_kind'))
                attributes = {'gen_ai.evaluation.name': result['name'],
                    'multivon.case.id': data['case_id'], 'multivon.case.digest': data['case_digest'],
                    'multivon.trial.digest': trial.digest, 'multivon.trial.run_index': data['run_index'],
                    'multivon.trial.attempt': data['attempt'], 'multivon.evaluation.measured': measured,
                    'multivon.trial.status': data['status'],
                    'multivon.report.has_evidence_issues': bool(report.evidence_issues),
                    'multivon.otel.profile': GENAI_PROFILE,
                    'multivon.otel.conventions_revision': GENAI_CONVENTIONS_REVISION}
                if measured:
                    score = result['score']
                    if type(score) not in (int, float) or not math.isfinite(score) or type(result['passed']) is not bool:
                        raise ValueError('Measured events require a finite score and boolean verdict')
                    attributes['gen_ai.evaluation.score.value'] = float(score)
                    attributes['gen_ai.evaluation.score.label'] = 'pass' if result['passed'] else 'fail'
                else:
                    attributes['gen_ai.evaluation.score.label'] = 'unmeasured'
                    if metadata.get('error_kind'):
                        attributes['error.type'] = str(metadata['error_kind'])
                if include_explanations:
                    attributes['gen_ai.evaluation.explanation'] = result['reason']
                pending.append(LogRecord(context=Context(), trace_id=trace_id, span_id=span_id,
                    event_name='gen_ai.evaluation.result', attributes=attributes,
                    severity_number=SeverityNumber.INFO))
    for record in pending:
        logger.emit(record)
    return len(pending)
