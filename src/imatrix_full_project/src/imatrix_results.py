"""Generate P1 layer-sensitivity measurements and final report from real scores.

CSV input columns: model,gsm8k,ifeval,mmlu and optionally kl,size_mib.
No benchmark scores are synthesized. Scores must have matched evaluation setups.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results' / 'imatrix_analysis'
TASKS = ('gsm8k', 'ifeval', 'mmlu')


def load_scores(path: Path):
    frame = pd.read_csv(path)
    needed = {'model', *TASKS}
    if not needed.issubset(frame.columns):
        raise ValueError(f'{path} missing fields {sorted(needed - set(frame.columns))}')
    if frame.model.duplicated().any():
        raise ValueError('Model names must be unique in the evaluation CSV')
    for task in TASKS:
        frame[task] = pd.to_numeric(frame[task], errors='raise')
        if not np.isfinite(frame[task]).all():
            raise ValueError(f'Invalid {task} scores')
    return frame.set_index('model')


def p1_sensitivity(scores: pd.DataFrame, plan: pd.DataFrame, out: Path,
                   control: str = 'p1-all-high'):
    if control not in scores.index:
        raise ValueError(f'Missing control model score: {control}')
    matched = plan[plan.method == 'layer-ablation']
    rows = []
    for record in matched.itertuples():
        if record.name not in scores.index:
            continue
        row = {'layer': int(record.ablated_layer), 'model': record.name}
        for task in TASKS:
            row[f'{task}_drop'] = float(scores.loc[control, task] - scores.loc[record.name, task])
        # Positive drop means the lower precision layer hurt observed quality.
        row['sensitivity'] = float(np.mean([row[f'{task}_drop'] for task in TASKS]))
        row['sensitivity_abs'] = float(np.mean([abs(row[f'{task}_drop']) for task in TASKS]))
        rows.append(row)
    if not rows:
        raise ValueError('No evaluated ablation models match the experiment plan')
    df = pd.DataFrame(rows).sort_values('layer')
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'p1_layer_sensitivity.csv', index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    for task in TASKS:
        ax.plot(df.layer, df[f'{task}_drop'], marker='.', label=task)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set(xlabel='Perturbed layer', ylabel='Accuracy drop vs all-high control',
           title='P1: measured one-layer precision perturbations')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / 'p1_layer_ablation.png', dpi=200)
    plt.close(fig)
    return df


def p2_summary(scores: pd.DataFrame, out: Path):
    rows = []
    for name, row in scores.iterrows():
        category = 'mixed' if name.startswith('mixed-') else ('uniform' if name.startswith('uniform-') else 'other')
        if category == 'other':
            continue
        if 'size_mib' not in scores.columns or pd.isna(row.get('size_mib')):
            continue
        rows.append(dict(model=name, method=category, size_mib=float(row.size_mib),
                         **{task: float(row[task]) for task in TASKS},
                         mean_score=float(np.mean([row[task] for task in TASKS]))))
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df['pareto'] = [not bool(((df.size_mib <= r.size_mib) & (df.mean_score >= r.mean_score) &
                              ((df.size_mib < r.size_mib) | (df.mean_score > r.mean_score))).any())
                    for r in df.itertuples()]
    df.to_csv(out / 'p2_model_comparison.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, group in df.groupby('method'):
        ax.scatter(group.size_mib, group.mean_score, label=method, s=60)
        for r in group.itertuples():
            ax.annotate(r.model, (r.size_mib, r.mean_score), xytext=(4,4),
                        textcoords='offset points', fontsize=7)
    frontier = df[df.pareto].sort_values('size_mib')
    if len(frontier) > 1:
        ax.plot(frontier.size_mib, frontier.mean_score, '--', alpha=.5)
    ax.set(xlabel='GGUF model size (MiB)', ylabel='Mean task score (matched metrics)',
           title='P2: measured quality vs model size')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'p2_pareto.png', dpi=200)
    plt.close(fig)
    return df


def combined_report(out: Path, p1, p2, measured: pd.DataFrame):
    chunks = [
        '# Quantization stability: measured results\n',
        f'Models with benchmark metrics: **{len(measured)}**.\n',
        'All figures and rankings in this report are based on supplied experiment scores, not simulated data.\n',
    ]
    if p1 is None:
        chunks.append('## P1\nNo matched layer-ablation evaluations yet. Run the generated P1 variants using identical task settings.\n')
    else:
        chunks += [f'## P1\nCompared **{len(p1)}** individually perturbed layers with the all-high precision control.\n',
                   'The layer sensitivity metric is the mean benchmark accuracy drop; negative values mean the ablation improved results.\n',
                   f'Top five measured layers: {", ".join(str(int(x)) for x in p1.nlargest(5,"sensitivity").layer)}.\n']
    if p2 is None:
        chunks.append('## P2\nNo uniform/mixed comparisons with model sizes available yet.\n')
    else:
        chunks += [f'## P2\nCompared **{len(p2)}** uniform/mixed models using measured file sizes and benchmark results.\n',
                   f'Non-dominated models on size and mean score: {", ".join(p2.loc[p2.pareto,"model"])}.\n']
    chunks.append('## Limitations\nLayer perturbations quantify dependence on the chosen precision pair and task samples; they do not directly identify intrinsic layer importance. Ensure identical prompts, splits, number of examples and generation settings. Compare each task individually, since averaging different benchmark scales can conceal trade-offs. Imatrix rankings remain predictive hypotheses until evaluated against these measurements.\n')
    (out / 'EXPERIMENT_REPORT.md').write_text('\n'.join(chunks), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores-csv', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--control', default='p1-all-high')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = load_scores(args.scores_csv)
    manifest_path = args.output_dir / 'experiment_plan.csv'
    p1 = None
    if manifest_path.is_file():
        plan = pd.read_csv(manifest_path)
        if args.control in scores.index and any(plan.method == 'layer-ablation'):
            p1 = p1_sensitivity(scores, plan, args.output_dir, args.control)
    p2 = p2_summary(scores, args.output_dir)
    combined_report(args.output_dir, p1, p2, scores)
    print(f'Wrote reports and available charts to {args.output_dir}')

if __name__ == '__main__':
    main()
