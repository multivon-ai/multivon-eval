"""Promote a bound Label Studio review into a new development case manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multivon_eval import EvalReport
from multivon_eval.integrations.label_studio import import_review_annotations
from multivon_eval.regressions import promote_reviewed_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'tasks', 'export', 'trial-digest', 'evaluator', 'case-id',
                 'expected-output-file', 'rationale-file', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--reviewer-kind', required=True, choices=['human', 'model', 'synthetic'])
    parser.add_argument('--min-reviewers', type=int, default=2)
    parser.add_argument('--source-id')
    args = parser.parse_args()
    def read(path):
        return json.loads(Path(path).read_text())
    report = EvalReport.from_dict(read(args.report))
    tasks = read(args.tasks)
    reviews = import_review_annotations(read(args.export), tasks, reviewer_kind=args.reviewer_kind)
    manifest = promote_reviewed_trial(report, args.trial_digest, tasks, reviews, evaluator=args.evaluator,
        case_id=args.case_id, expected_output=Path(args.expected_output_file).read_text(),
        rationale=Path(args.rationale_file).read_text(), reviewer_kind=args.reviewer_kind,
        min_reviewers=args.min_reviewers, source_id=args.source_id)
    with Path(args.output).open('x') as handle:
        json.dump(manifest.manifest, handle, indent=2)
    print(f'Wrote development regression manifest to {args.output}')


if __name__ == '__main__':
    main()
