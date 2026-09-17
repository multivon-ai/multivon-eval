"""Render the saved world-model diagnostics; no target or simulator execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(root, output):
    summary = json.loads((root / 'summary.json').read_text())
    names = ['action_conditioned', 'action_blind', 'persistence']
    labels = ['Action-conditioned', 'Action-blind', 'Persistence']
    colors = ['#087F8C', '#C56B26', '#697386']
    horizons = [1, 4, 8, 16, 32]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout='constrained')
    fig.suptitle('CartPole: prediction quality needs a decision test', fontsize=18, fontweight='bold')
    ax = axes[0, 0]
    for name, label, color in zip(names, labels, colors):
        values = [summary['models'][name]['by_horizon'][str(h)]['coordinates']['pole_angle']['common_cohort_rmse']
                  for h in horizons]
        ax.plot(horizons, values, marker='o', label=label, color=color, linewidth=2)
    ax.set(yscale='log', xlabel='Prediction horizon (steps)', ylabel='Pole-angle RMSE (rad)',
           title='A  Same six sources at every horizon')
    ax.set_xticks(horizons)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.18)
    ax = axes[0, 1]
    values = [summary['models'][name]['action_effect']['1']['cart_velocity']['effect_rmse'] for name in names]
    ax.bar(labels, values, color=colors, width=0.6)
    ax.set(ylabel='Action-effect RMSE (m/s)', title='B  Action 1 versus action 0 · n = 40 sources', ylim=(0, 0.47))
    for i, value in enumerate(values):
        ax.text(i, value + 0.013, f'{value:.6f}', ha='center', fontsize=10)
    ax = axes[1, 0]
    for i, (name, color) in enumerate(zip(names, colors)):
        rows = summary['models'][name]['planning']['rows']
        values = [r['steps'] for r in rows]
        ax.bar(i, np.mean(values), color=color, alpha=0.5, width=0.6)
        ax.scatter(i + np.linspace(-0.12, 0.12, len(values)), values, s=22, color=color, zorder=3)
        ax.text(i, np.mean(values) + 9, f'{np.mean(values):.1f}', ha='center')
    ax.set(xticks=range(3), xticklabels=labels, ylabel='Real environment steps executed', ylim=(0, 235),
           title='C  Same 64-plan search · n = 10 seeds each')
    ax.axhline(200, color='#999999', linestyle='--', linewidth=1)
    ax.text(2.4, 204, '200-step cap', ha='right', fontsize=9, color='#666666')
    ax = axes[1, 1]
    for name, label, color in zip(names[:2], labels[:2], colors[:2]):
        point = summary['models'][name]['by_horizon']['8']['coordinates']['pole_angle']
        ax.scatter(point['mean_width'], point['coverage_90'], s=90, color=color)
        ax.annotate(label, (point['mean_width'], point['coverage_90']), xytext=(0, -20),
                    textcoords='offset points', ha='center', color=color, fontsize=10)
    ax.axhline(0.9, color='#777777', linestyle='--', linewidth=1)
    ax.text(0.0007, 0.905, 'Nominal 90%', fontsize=9, color='#666666')
    ax.set(xscale='log', xlim=(0.0006, 0.8), ylim=(0.86, 1.025),
           xlabel='Mean marginal interval width (rad)', ylabel='Observed coverage',
           title='D  Pole angle at step 8 · n = 40 sources')
    ax.text(0.5, 0.04, 'Persistence supplies no uncertainty estimate.', ha='center',
            transform=ax.transAxes, fontsize=9, color='#666666')
    fig.supxlabel('Frozen Gymnasium study · fully observed states · no video or real-robot claim\n'
                  'A uses only the six trajectories observed through step 32; B–D use the stated source counts.', fontsize=10)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'world-model-diagnostics.png', dpi=180)
    fig.savefig(output / 'world-model-diagnostics.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(args.root, args.output)
