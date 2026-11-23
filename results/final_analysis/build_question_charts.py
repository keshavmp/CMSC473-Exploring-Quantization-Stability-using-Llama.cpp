"""Build the audience-facing, multi-precision analysis. PNG only; no inference."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

import build_analysis as base

OUT = Path(__file__).resolve().parent
CHARTS = OUT / 'questions'
PRECISIONS = ['Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8']
BLOCK_PRECISIONS = ['Q2', 'Q4', 'Q6']
BLUE, ORANGE = '#387aa5', '#d98932'


def save(fig, name):
    fig.savefig(CHARTS / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def top3(g, column):
    return set(g.sort_values([column, 'block'], ascending=[False, True]).head(3).block)


def annotate_bars(ax, bars, labels):
    ax.bar_label(bars, labels=labels, padding=5, fontsize=12)


def finish(ax, title, ylabel, footer):
    ax.set_title(title, fontsize=18, fontweight='bold', pad=16)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    ax.figure.supxlabel(footer, fontsize=10)


def prediction_chart(prediction):
    rows = []
    for (precision, domain), g in prediction.groupby(['precision', 'domain']):
        for predictor in ['input_energy', 'local_increment_pct']:
            measured = top3(g, 'sensitivity_nats')
            predicted = top3(g, predictor)
            rows.append(dict(precision=precision, domain=domain, predictor=predictor,
                             hits=len(measured & predicted), possible_hits=3,
                             predicted_blocks=','.join(map(str, sorted(predicted))),
                             measured_blocks=','.join(map(str, sorted(measured)))))
    detail = pd.DataFrame(rows)
    pooled = detail.groupby(['precision', 'predictor'], as_index=False)[['hits', 'possible_hits']].sum()
    pooled['identified_pct'] = 100 * pooled.hits / pooled.possible_hits
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    x = np.arange(3)
    for offset, predictor, color, label in [(-.18, 'input_energy', '#a7bbc9', 'Activation energy'),
                                          (.18, 'local_increment_pct', BLUE, 'Activation-weighted error')]:
        g = pooled[pooled.predictor.eq(predictor)].set_index('precision').loc[BLOCK_PRECISIONS]
        bars = ax.bar(x + offset, g.identified_pct, width=.34, color=color, label=label)
        annotate_bars(ax, bars, [f'{n}/9' for n in g.hits])
    ax.set_xticks(x, BLOCK_PRECISIONS)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(frameon=False, loc='upper right')
    finish(ax, 'Can we find the most sensitive blocks?', 'Top-sensitive blocks identified (%)',
           'Top 3 per domain, pooled across code / math / chat: 9 selections per precision.\nLlama 1B | Higher is better')
    save(fig, '01_identify_sensitive_blocks')
    return detail, pooled


def preservation_chart(prediction, preservation):
    rows = []
    for (precision, domain), g in prediction.groupby(['precision', 'domain']):
        selected = top3(g, 'sensitivity_nats')
        p = preservation[preservation.precision.eq(precision) & preservation.domain.eq(domain)]
        for sensitive, label in [(True, 'Most sensitive'), (False, 'Other blocks')]:
            group = p[p.block.isin(selected).eq(sensitive)]
            rows.append(dict(precision=precision, domain=domain, group=label,
                             control_error=group.control_error.sum(), error_removed=group.error_removed.sum()))
    detail = pd.DataFrame(rows)
    pooled = detail.groupby(['precision', 'group'], as_index=False)[['control_error', 'error_removed']].sum()
    pooled['recovery_pct'] = 100 * pooled.error_removed / pooled.control_error
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    x = np.arange(3)
    for offset, group, color in [(-.18, 'Most sensitive', ORANGE), (.18, 'Other blocks', BLUE)]:
        g = pooled[pooled.group.eq(group)].set_index('precision').loc[BLOCK_PRECISIONS]
        bars = ax.bar(x + offset, g.recovery_pct, width=.34, color=color, label=group)
        annotate_bars(ax, bars, [f'{n:.1f}%' for n in g.recovery_pct])
    ax.set_xticks(x, BLOCK_PRECISIONS)
    ax.set_ylim(0, max(45, np.ceil(pooled.recovery_pct.max() / 10) * 10 + 10))
    ax.legend(frameon=False, loc='upper right')
    finish(ax, 'Do the sensitive blocks benefit more?', 'Original reconstruction error removed (%)',
           'Top 3 sensitive blocks per domain vs. the other 13; domains and calibrations pooled.\nLlama 1B | Taller orange bars would mean more proportional recovery in sensitive blocks')
    save(fig, '02_preserve_sensitive_blocks')
    return detail, pooled


def output_chart(kl):
    pooled = kl[kl.variant.ne('control')].groupby('precision')[['kl_nats', 'control_kl_nats']].sum().loc[PRECISIONS]
    pooled['reduction_pct'] = 100 * (1 - pooled.kl_nats / pooled.control_kl_nats)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    bars = ax.bar(PRECISIONS, pooled.reduction_pct,
                  color=[BLUE if v >= 0 else ORANGE for v in pooled.reduction_pct], width=.6)
    annotate_bars(ax, bars, [f'{v:+.1f}%' if abs(v) > .0001 else '0%' for v in pooled.reduction_pct])
    ax.axhline(0, color='gray', lw=1)
    ax.set_ylim(min(-15, pooled.reduction_pct.min() - 7), max(45, pooled.reduction_pct.max() + 8))
    finish(ax, 'Does imatrix keep outputs closer to BF16?', 'Output KL reduction (%)',
           'Matched no-imatrix control = 0%; calibration choices and text domains pooled.\nLlama 1B | Above zero = better; below zero = worse')
    save(fig, '03_output_fidelity')
    return pooled.reset_index()


def benchmark_charts(gains):
    # All tasks share the same color scale in percentage points. No score ratios.
    bound = max(1, np.ceil(gains.gain_pp.abs().max()))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    labels = {'gsm8k': ('Does imatrix improve math reasoning?', 'GSM8K'),
              'ifeval': ('Does imatrix improve instruction following?', 'IFEval'),
              'hellaswag': ('Does imatrix improve commonsense answers?', 'HellaSwag')}
    for index, (task, (title, name)) in enumerate(labels.items(), start=4):
        g = gains[gains.task.eq(task)]
        matrix = g.pivot(index='variant', columns='precision', values='gain_pp').loc[base.DOMAINS, PRECISIONS]
        stars = g.pivot(index='variant', columns='precision', values='passes_holm_0_05').loc[base.DOMAINS, PRECISIONS]
        fig, ax = plt.subplots(figsize=(8.5, 3.6), layout='constrained')
        im = ax.imshow(matrix.to_numpy(), cmap='RdBu', norm=norm, aspect='auto')
        for r in range(3):
            for c in range(6):
                value = matrix.iloc[r, c]
                text = f'{value:+.1f}' if abs(value) >= .05 else '0.0'
                if stars.iloc[r, c]:
                    text += '*'
                ax.text(c, r, text, ha='center', va='center', fontsize=13,
                        color='white' if abs(value) > bound * .6 else '#172a39')
        ax.set_xticks(range(6), PRECISIONS)
        ax.set_yticks(range(3), ['Code calibration', 'Math calibration', 'Chat calibration'])
        ax.set_title(title, fontsize=17, fontweight='bold', pad=16)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(im, ax=ax, shrink=.8, label='Score gain (percentage points)')
        fig.supxlabel(f'{name} | Blue = better; red = worse; 0 = no change.\n* passes the existing paired test after correction; intervals are in the CSV.', fontsize=10)
        save(fig, f'{index:02d}_{task}_performance')


def main():
    CHARTS.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    predictions, preservations, all_gains, all_kl = [], [], [], []
    for precision in BLOCK_PRECISIONS:
        prediction = base.load_prediction(precision)
        preservation, _ = base.load_preservation(prediction, precision)
        prediction['precision'] = precision
        preservation['precision'] = precision
        predictions.append(prediction)
        preservations.append(preservation)
    for precision in PRECISIONS:
        manifest = base.read_json(f'results/Imatrix_Recovery_Sweep_full_kl4/quantization_manifest_{precision.lower()}.json')
        # Verify matching quantization inputs/layouts also for levels without block interventions.
        for variant in base.DOMAINS:
            base.require(manifest[variant]['tensor_layout'] == manifest['control']['tensor_layout'], 'Layout mismatch')
            for key in ['source_sha256', 'tool_sha256', 'quantization_type', 'pure']:
                base.require(manifest[variant]['inputs'][key] == manifest['control']['inputs'][key], f'Quantization mismatch: {key}')
        gains, kl = base.load_outcomes(manifest, precision)
        kl['precision'] = precision
        all_gains.append(gains)
        all_kl.append(kl)
    prediction = pd.concat(predictions, ignore_index=True)
    preservation = pd.concat(preservations, ignore_index=True)
    gains = pd.concat(all_gains, ignore_index=True)
    kl = pd.concat(all_kl, ignore_index=True)
    detail, pooled = prediction_chart(prediction)
    group_detail, group_pooled = preservation_chart(prediction, preservation)
    fidelity = output_chart(kl)
    benchmark_charts(gains)
    tables = {'prediction_blocks': prediction, 'preservation_blocks': preservation,
              'benchmark_gains': gains, 'output_kl': kl, 'prediction_top3_by_domain': detail,
              'prediction_summary': pooled, 'sensitive_group_by_domain': group_detail,
              'sensitive_group_summary': group_pooled, 'output_fidelity_summary': fidelity}
    for name, data in tables.items():
        base.require(not data.isna().any().any(), f'Missing data: {name}')
        data.to_csv(CHARTS / f'{name}.csv', index=False, lineterminator='\n')
    from build_statistic_comparison import build as compare_statistics
    compare_statistics(prediction)
    write_readme(pooled, group_pooled, fidelity)
    refs = {}
    for ref in ['HEAD', 'layers_sneha', 'origin/layers_sneha']:
        result = subprocess.run(['git', 'rev-parse', ref], cwd=base.ROOT, capture_output=True, text=True)
        refs[ref] = result.stdout.strip() if result.returncode == 0 else None
    checks = {'scope': 'Llama only; block analysis Q2/Q4/Q6; outcomes Q2/Q3/Q4/Q5/Q6/Q8',
              'input_sha256': base.INPUTS, 'local_git_refs': refs,
              'report_sha256': hashlib.sha256((OUT / 'README.md').read_bytes()).hexdigest(),
              'generator_sha256': {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                                   for name in ['build_question_charts.py', 'build_analysis.py', 'build_statistic_comparison.py']},
              'rebuild': 'python results/final_analysis/build_question_charts.py',
              'output_sha256': {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in CHARTS.rglob('*') if p.is_file() and p != CHARTS / 'manifest.json'},
              'checks': ['48 domain/block measurements per intervention precision',
                         'seven projections per block; unique and complete joins',
                         'quantized/calibrated models have matched inputs and stored layouts',
                         'all 24 raw evaluation inputs and paired gain table match publication hashes',
                         'gains recomputed from raw metrics; KL direction and token counts match'],
              'limits': ['cross-machine checkpoint/text/build hashes not established for interventions',
                         'no new inference, repeated executions, or weight payload audit',
                         'pooled domains/calibrations are related observations, not independent trials']}
    (CHARTS / 'manifest.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8', newline='\n')
    old_manifest = OUT / 'manifest.json'
    old_manifest.write_text(json.dumps({
        'scope': checks['scope'], 'current_results': 'questions/',
        'current_audit_manifest': 'questions/manifest.json',
        'current_audit_sha256': hashlib.sha256((CHARTS / 'manifest.json').read_bytes()).hexdigest(),
        'title': 'Llama Quantization: Sensitivity Predictions and Imatrix Benefits'
    }, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('Built six question charts (PNG only) and nine multi-precision CSVs.')
    print(pooled.to_string(index=False))
    print(group_pooled.to_string(index=False))
    print(fidelity[['precision', 'reduction_pct']].to_string(index=False))


def write_readme(prediction, groups, fidelity):
    previous = OUT / 'README.md'
    readme = '''# Llama Quantization: Sensitivity Predictions and Imatrix Benefits

Each chart answers one question. All results use existing Llama-3.2-1B measurements.

## 1. Can we identify sensitive blocks?

![Find sensitive blocks](questions/01_identify_sensitive_blocks.png)

Compare the predicted top 3 with Sneha's measured top 3 in each domain.
Nine selections = three blocks in each of code, math and chat; the same block can
appear in more than one domain. Higher is better. No method identifies every block.

### Can another statistic do better?

![Other imatrix statistics](questions/statistics/01_find_critical_blocks.png)

**Energy concentration (lower entropy) finds more top-sensitive blocks:** 18/27
selections versus 13/27 for weighted error. But weighted error still best matches
the full ordering of all 16 blocks. These are different goals.

See the [two-chart statistic comparison](questions/statistics/README.md) for full
rank agreement, definitions and checks. This is exploratory, not proof of a winner.

## 2. Do sensitive blocks benefit more?

![Sensitive-block recovery](questions/02_preserve_sensitive_blocks.png)

Orange = the three most sensitive blocks in each domain; blue = the other thirteen.
Bars show the percentage of each group's original error removed. Taller orange
bars would support preferential recovery of sensitive blocks.

## 3. Does the model stay closer to BF16?

![Output fidelity](questions/03_output_fidelity.png)

Above zero = closer to BF16. Below zero = farther away. This measures output
distributions, not task accuracy.

## 4. Does task performance improve?

Each number is the score change against the same-precision no-imatrix model,
in percentage points. Blue = better; red = worse. All three charts share one color
scale. A star marks a result passing the existing corrected paired test.

![Math reasoning](questions/04_gsm8k_performance.png)

![Instruction following](questions/05_ifeval_performance.png)

![Commonsense](questions/06_hellaswag_performance.png)

## What can we conclude?

Imatrix's predictions are imperfect. It improves local reconstruction broadly,
but does not consistently favor the most sensitive blocks. Output fidelity
improves most at low precision; task gains depend on precision and calibration.
Q6 can have better local reconstruction yet worse output KL. Q8 has unchanged
stored weights in the earlier audit, so its small score changes are not evidence
of an imatrix weight improvement.

## Scope and normalization

- **Block comparisons: Q2, Q4, Q6.** No block interventions are available at
  Q3/Q5/Q8. These comparisons cannot establish sensitivity there.
- **Whole-model results: Q2, Q3, Q4, Q5, Q6, Q8.** Controls use the same standard
  recipe and stored tensor layouts as the calibrated model at each precision.
- **Chart 1:** correct top-three identifications / nine domain-specific selections.
  Ties at the top-three boundary favor the lower block index. This is not full-rank
  agreement or a statistical significance test.
- **Chart 2:** 100 x summed error removed / summed control error. Pool three
  calibration choices and three measurement domains before division; select
  sensitive blocks separately for each precision/domain. Unequal group sizes
  are normalized to each group's own error, not to parameter count or file size.
- **Chart 3:** 100 x (1 - summed imatrix KL / summed control KL), pooling three
  calibrations and three domains. Each KL measurement scores 1,020 tokens.
- **Task charts:** raw percentage-point gains, not relative percent improvements.
  Paired confidence intervals and corrected p-values remain in
  [benchmark_gains.csv](questions/benchmark_gains.csv). Colors alone do not imply
  significance; the original correction covers 54 comparisons. Single executions
  do not establish run-to-run reliability.
- **Protocol mismatch:** sensitivity lowers one block in an otherwise-Q8 model;
  reconstruction and task gains come from whole-model standard recipes. The
  connection is exploratory. Intervention checkpoint/text/build hashes are not
  fully matched across machines. Calibration budgets differ: code/math/chat
  49/97/699 chunks of 512 tokens. Results cannot isolate domain from budget.

## Files

Use the six PNGs in [questions/](questions/) for presentation. Their source tables,
exact plotted values, paired uncertainty and provenance are in the same folder.
The additional feature comparison is in [questions/statistics/](questions/statistics/).

Redraw all eight published PNGs from the included tables (no model files needed):

`python results/final_analysis/build_question_charts.py --render-only`

Requires Python with numpy, pandas and matplotlib (included in the project requirements).

Recompute from the complete local experiment outputs:

`python results/final_analysis/build_question_charts.py`

The full recomputation requires the original raw sweep, imatrix exports and audit
inputs named in `questions/manifest.json`. Those large intermediate folders are
not bundled with this summary. The published CSVs preserve the measured values
used in the charts; hashes alone cannot reconstruct omitted source data.

No new model runs. PNG only.
'''
    previous.write_text(readme, encoding='utf-8', newline='\n')


def render_saved():
    """Redraw the published figures from bundled CSVs only."""
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    prediction = pd.read_csv(CHARTS / 'prediction_blocks.csv')
    preservation = pd.read_csv(CHARTS / 'preservation_blocks.csv')
    gains = pd.read_csv(CHARTS / 'benchmark_gains.csv')
    kl = pd.read_csv(CHARTS / 'output_kl.csv')
    prediction_chart(prediction)
    preservation_chart(prediction, preservation)
    output_chart(kl)
    benchmark_charts(gains)
    from build_statistic_comparison import charts
    stats = pd.read_csv(CHARTS / 'statistics/comparisons_by_precision.csv')
    summary = pd.read_csv(CHARTS / 'statistics/summary.csv')
    charts(stats, summary.sort_values('mean_rank_agreement', ascending=False).feature.tolist())
    print('Redrew eight PNG charts from bundled tables; no original experiment files or model runs required.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only', action='store_true', help='Redraw from bundled CSVs without raw experiment inputs')
    args = parser.parse_args()
    render_saved() if args.render_only else main()
