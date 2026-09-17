"""Bound media -> Inspect -> saved grounded verdicts, with an optional six-call document probe.

Install development core[media,inspect] and pdfhell[pixels]==0.6.2.
Default execution is mock-only. --document-model anthropic/claude-haiku-4-5-20251001
makes six target requests on two explicitly synthetic development documents.
Credentials come from the environment. This is not an accuracy benchmark.
"""
from __future__ import annotations

import argparse
import io
import json
import wave
from importlib.metadata import version
from pathlib import Path

import av
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelCost, ModelInfo, ModelOutput, set_model_info
from inspect_ai.solver import generate
from pdfhell.generators.hidden_ocr_mismatch import generate as generate_pdf
from pypdf import PdfReader

from multivon_eval import CaseManifest, EvalCase, ExactMatch
from multivon_eval.evaluators.base import Evaluator
from multivon_eval.grounding import annotation, grounded_result
from multivon_eval.integrations.inspect import (
    as_inspect_scorer,
    from_inspect_log,
    to_inspect_dataset,
)
from multivon_eval.media import MediaArtifact, case_media, with_media
from multivon_eval.media_pdf import render_pdf_page


class VisibleAmount(Evaluator):
    name = 'visible_amount_exact'

    def __init__(self):
        super().__init__(1.0)
        self.contract = 'pdfhell-visible-amount-exact/v1'

    def evaluate(self, case, output):
        artifacts = case_media(case)
        exact = ExactMatch().evaluate(case, output)
        if not artifacts:
            return self._result(exact.score, exact.reason, treatment='extracted_text')
        return grounded_result(self.name, exact.passed, 'Exact comparison to the generator-authored visible amount.',
            annotations=[annotation(artifacts[0], f'Expected visible amount: {case.expected_output}')], artifacts=artifacts)


def temporal_fixtures():
    audio = io.BytesIO()
    with wave.open(audio, 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(bytes(16000))
    video = io.BytesIO()
    with av.open(video, 'w', format='mp4') as container:
        stream = container.add_stream('libx264', rate=4)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for index in range(4):
            frame = av.VideoFrame(64, 48, 'rgb24')
            frame.planes[0].update(bytes([40 + index * 50]) * frame.planes[0].buffer_size)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return [(audio.getvalue(), 'audio/wav', 'silence.wav'), (video.getvalue(), 'video/mp4', 'ramp.mp4')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--document-model', choices=['mockllm/model', 'anthropic/claude-haiku-4-5-20251001'],
                        default='mockllm/model')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'assets').mkdir()
    registry, paths, documents, transport, references = {}, {}, [], [], []
    def retain(artifact, data, filename):
        registry[artifact.id] = data
        paths[artifact.id] = {'path': 'assets/' + filename, 'artifact': artifact.data}
        (root / 'assets' / filename).write_bytes(data)
        return artifact
    for seed in (501, 502):
        pdf, source = generate_pdf(seed)
        parent = retain(MediaArtifact.capture(pdf, 'application/pdf', provenance={
            'generator': 'pdfhell.hidden_ocr_mismatch', 'version': version('pdfhell'),
            'seed': seed, 'split': 'development', 'source_kind': 'synthetic'}), pdf, f'{seed}.pdf')
        page, pixels = render_pdf_page(parent, pdf, page=1)
        retain(page, pixels, f'{seed}.png')
        extracted = '\n'.join(p.extract_text() or '' for p in PdfReader(io.BytesIO(pdf)).pages)
        assert source.expected_answer in extracted and source.forbidden_answers[0] in extracted
        (root / 'assets' / f'{seed}.txt').write_text(extracted)
        prompt = 'What is the visibly printed TOTAL DUE? Return only the amount with currency symbol.'
        for treatment, artifact in [('pdf', parent), ('pixels', page), ('text', None)]:
            case = EvalCase(prompt + ('\n\nExtracted text:\n' + extracted if artifact is None else ''),
                source.expected_answer, case_id=f'{seed}-{treatment}', source_id=f'pdfhell-{seed}',
                tags=[treatment], metadata={'treatment': treatment, 'seed': seed,
                    'forbidden_hidden_amount': source.forbidden_answers[0], 'split': 'development'})
            documents.append(with_media(case, artifact) if artifact else case)
        if seed == 501:
            transport.extend([with_media(EvalCase('Retain native PDF.', 'fixture', case_id='native-pdf'), parent),
                              with_media(EvalCase('Retain page pixels.', 'fixture', case_id='page-image'), page)])
            references.append(annotation(page, 'The visible total appears on this page.',
                region=(0, 0, page.data['properties']['width'], page.data['properties']['height'])))
    for data, mime, filename in temporal_fixtures():
        artifact = retain(MediaArtifact.capture(data, mime, provenance={'source_kind': 'synthetic transport fixture'}), data, filename)
        transport.append(with_media(EvalCase('Retain fixture media.', 'fixture', case_id=filename), artifact))
        references.append(annotation(artifact, 'Synthetic fixture interval.', time=(0, 0.5)))
    manifest = CaseManifest('visible amount development probe', documents,
        splits={'development': [c.case_id for c in documents]},
        provenance={'source_groups': 2, 'treatments': ['pdf', 'pixels', 'text'], 'max_target_calls': 6,
                    'max_output_tokens_per_call': 128, 'retry_count': 0,
                    'scope': 'synthetic integration probe, not independent task validity or accuracy'})
    manifest.save(root / 'manifest.json')
    (root / 'assets.json').write_text(json.dumps(paths, indent=2))
    (root / 'annotations.json').write_text(json.dumps(references, indent=2))
    set_model_info('mockllm/model', ModelInfo())
    results = {}
    for name, selected, model, grader in [
        ('transport', CaseManifest('native media transport', transport), 'mockllm/model', ExactMatch()),
        ('documents', manifest, args.document_model, VisibleAmount()),
    ]:
        task = Task(name=name, dataset=to_inspect_dataset(selected, media_resolver=lambda a: registry[a.id]),
                    solver=generate(), scorer=as_inspect_scorer(grader))
        model_args = ({'custom_outputs': [ModelOutput.from_content('fixture', c.expected_output) for c in selected.cases]}
                      if model == 'mockllm/model' else {})
        logs = inspect_eval(task, model=model, model_args=model_args, max_tokens=128,
            max_retries=0, retry_on_error=0, max_samples=1, max_connections=1, time_limit=120,
            cost_limit=0.05 if model != 'mockllm/model' else None, model_cost_config={'anthropic/claude-haiku-4-5-20251001': ModelCost(input=1, output=5, input_cache_write=1.25, input_cache_read=0.1)},
            log_dir=str(root / 'logs'), log_model_api=True, log_images=True, display='none')
        log = read_eval_log(logs[0].location, resolve_attachments=True)
        report = from_inspect_log(log)
        report.save_json(str(root / f'{name}-report.json'))
        results[name] = {'status': log.status, 'model': model, 'cases': report.total, 'passed': report.passed,
                         'errors': report.errors, 'usage': {k: v.model_dump(mode='json') for k,v in log.stats.model_usage.items()},
                         'outputs': [{ 'case_id': r.case_id, 'output': r.actual_output, 'passed': r.passed} for r in report.case_results]}
        if log.status != 'success':
            raise RuntimeError(f'{name} execution failed; inspect saved log before continuing')
    results['versions'] = {name: version(name) for name in ['inspect-ai','pdfhell','av','pypdfium2','Pillow','anthropic']}
    (root / 'summary.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
