"""Offline exploratory ablation of the synthetic ledger validation evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.industrial.gymnasium_ledger import CHECKS, POLICY, posting
from multivon_eval import ExactMatch
from multivon_eval.case_manifest import case_from_dict
from multivon_eval.episode import EpisodeEvidence, evaluate_episode


def analyze(root: Path) -> dict:
    rows = []
    for path in sorted(root.glob('*-episode.json')):
        episode = EpisodeEvidence.from_dict(json.loads(path.read_text()))
        data = episode.data
        decision = POLICY.evaluate(evaluate_episode(episode, CHECKS)).decision
        completed = not episode.coverage_issues
        text = ExactMatch().evaluate(case_from_dict(data['case']), data['output']).passed if completed else None
        final_post = posting(episode).passed if completed else None
        rows.append({'episode': path.name, 'digest': episode.digest, 'complete': completed,
                     'final_text_passed': text, 'final_posting_passed': final_post,
                     'full_outcome_decision': decision})
    measured = [row for row in rows if row['complete']]
    baselines = {}
    for field in ('final_text_passed', 'final_posting_passed'):
        baselines[field] = {'measured': len(measured), 'accepted': sum(r[field] for r in measured),
            'accepted_despite_failed_full_contract': sum(r[field] and r['full_outcome_decision'] == 'reject'
                                                        for r in measured)}
    return {'purpose': 'Exploratory controlled-fixture ablation, not an independent accuracy benchmark',
            'episodes': len(rows), 'complete_episodes': len(measured),
            'incomplete_episodes': len(rows) - len(measured), 'baselines': baselines, 'rows': rows,
            'critique': 'A bespoke checker of the same database and audit history can reproduce the full verdicts. '
                        'This validates evidence handling and policy integration, not a new oracle or research moat.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory), indent=2))
