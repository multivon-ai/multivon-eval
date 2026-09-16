"""Exchange saved trials with Label Studio; no target or judge API calls.

Development checkout only until the next release. Run with --help for commands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multivon_eval import EvalReport
from multivon_eval.integrations.label_studio import (
    LABEL_CONFIG, export_review_tasks, import_review_annotations, reconcile_reviews,
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('report')
    export.add_argument('--evaluator', required=True)
    export.add_argument('--rubric-file', required=True)
    export.add_argument('--output-dir', required=True)
    ingest = commands.add_parser('import')
    ingest.add_argument('original_tasks')
    ingest.add_argument('label_studio_export')
    ingest.add_argument('--reviewer-kind', required=True, choices=['human', 'model', 'synthetic'])
    ingest.add_argument('--min-reviewers', type=int, default=2)
    ingest.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if args.command == 'export':
        tasks = export_review_tasks(EvalReport.from_dict(read(args.report)), args.evaluator,
                                    rubric=Path(args.rubric_file).read_text(encoding='utf-8'))
        write(output / 'tasks.json', tasks)
        (output / 'label-config.xml').write_text(LABEL_CONFIG, encoding='utf-8')
    else:
        tasks = read(args.original_tasks)
        annotations = import_review_annotations(read(args.label_studio_export), tasks,
                                               reviewer_kind=args.reviewer_kind)
        consensus = reconcile_reviews(tasks, annotations, min_reviewers=args.min_reviewers,
                                      reviewer_kind=args.reviewer_kind)
        write(output / 'annotations.json', [a.to_dict() for a in annotations])
        write(output / 'consensus.json', consensus)
    print(f'Wrote review artifacts to {output}')


if __name__ == '__main__':
    main()
