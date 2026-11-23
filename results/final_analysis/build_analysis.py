"""Validated measurement loaders and legacy Q4 chart helpers; no inference.

Run from the repository root: python results/final_analysis/build_analysis.py
Requires the project's numpy, pandas and matplotlib dependencies.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DOMAINS = ['code', 'math', 'chat']
TYPES = {'attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down'}
COLORS = {'code': '#3973ac', 'math': '#c86628', 'chat': '#32835c'}
INPUTS = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source(relative):
    path = ROOT / relative
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(relative):
    return json.loads(source(relative).read_text(encoding='utf-8'))


def csv(relative):
    return pd.read_csv(source(relative))


def unique(frame, keys):
    require(not frame.duplicated(keys).any(), f'Duplicate keys: {keys}')


def coverage(frame, keys):
    unique(frame, keys + ['tensor'])
    kinds = frame.groupby(keys).tensor_type.agg(list)
    require(kinds.map(lambda values: len(values) == 7 and set(values) == TYPES).all(),
            'Every block must contain exactly the seven expected projections')


def rho(frame, x, y='sensitivity_nats'):
    return float(frame[x].rank(method='average').corr(frame[y].rank(method='average')))


def table(frame):
    def fmt(v):
        return f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)
    lines = ['| ' + ' | '.join(frame.columns) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    return '\n'.join(lines + ['| ' + ' | '.join(map(fmt, row)) + ' |'
                              for row in frame.itertuples(index=False, name=None)])


def save_plot(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def load_prediction(precision='Q4'):
    folder = 'results/Block_Sensitivity_Differences'
    control = read_json(f'{folder}/pure-q8/kl.json')
    rows = []
    for block in range(16):
        run = read_json(f'{folder}/blk{block:02d}-{precision.lower()}_k/kl.json')
        require(run['baseline'] == control['baseline'], 'Block baseline mismatch')
        require(run['direction'] == control['direction'] == 'BF16 || evaluated model', 'KL direction mismatch')
        require(run['units'] == control['units'] == 'nats', 'KL unit mismatch')
        for domain in DOMAINS:
            a, b = run['domains'][domain], control['domains'][domain]
            require(a['tokens'] == b['tokens'] == 1020, 'KL coverage mismatch')
            require(a['text_file'] == b['text_file'], 'Block/control KL text path mismatch')
            rows.append(dict(block=block, domain=domain, intervention_kl_nats=a['mean'],
                             q8_control_kl_nats=b['mean'], sensitivity_nats=a['mean'] - b['mean'],
                             intervention_tool_uncertainty_nats=a['uncertainty'],
                             q8_tool_uncertainty_nats=b['uncertainty'], scored_tokens=a['tokens']))
    observed = pd.DataFrame(rows)
    stats = csv('results/imatrix_full_pure_sweep/tensor_statistics.csv')
    stats = stats[stats.model.eq('Llama-3.2-1B-Instruct-BF16')].copy()
    stats['tensor_type'] = stats.tensor.str.extract(r'blk\.\d+\.(.*)\.weight')[0]
    coverage(stats, ['task', 'layer'])
    require(len(stats) == 336, 'Expected 112 BF16 projections in each domain')
    energy = stats.groupby(['task', 'layer'], as_index=False).agg(input_energy=('energy', 'sum'))
    errors = csv('results/important plots/activation_weighted_errors.csv')
    refs = csv('results/important plots/by precision/bf16_diagonal_reference_energy.csv')
    for name, path, manifest in [
        ('errors', 'results/important plots/activation_weighted_errors.csv', 'results/important plots/weight_error_provenance.json'),
        ('references', 'results/important plots/by precision/bf16_diagonal_reference_energy.csv', 'results/important plots/by precision/reference_energy_provenance.json')]:
        require(read_json(manifest)['csv_sha256'] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                f'Stale {name} table')
    errors = errors[errors.precision.isin([precision, 'Q8'])]
    coverage(errors, ['precision', 'task', 'layer'])
    unique(refs, ['task', 'tensor'])
    errors = errors.merge(refs, on=['task', 'tensor'], how='left', validate='many_to_one')
    require(errors.reference_energy.notna().all(), 'Missing reference energy')
    sums = errors.groupby(['precision', 'task', 'layer'])[['weighted_error', 'reference_energy']].sum()
    q4, q8 = sums.loc[precision].copy(), sums.loc['Q8']
    require(np.allclose(q4.reference_energy, q8.reference_energy), 'Q4/Q8 reference mismatch')
    q4['q8_weighted_error'] = q8.weighted_error
    q4['local_increment_pct'] = 100 * (q4.weighted_error - q4.q8_weighted_error) / q4.reference_energy
    predictors = q4.reset_index().merge(energy, on=['task', 'layer'], validate='one_to_one')
    predictors = predictors.rename(columns={'task': 'domain', 'layer': 'block'})
    joined = observed.merge(predictors, on=['domain', 'block'], how='left', validate='one_to_one')
    require(len(joined) == 48 and joined.local_increment_pct.notna().all(), 'Incomplete prediction join')
    for column in ['input_energy', 'local_increment_pct', 'sensitivity_nats']:
        joined[column + '_rank'] = joined.groupby('domain')[column].rank(ascending=False, method='average')
    return joined


def load_preservation(prediction, precision='Q4'):
    folder = 'results/Imatrix_Recovery_Sweep_full_kl4'
    manifest = read_json(f'{folder}/quantization_manifest_{precision.lower()}.json')
    base = manifest['control']
    for variant in DOMAINS:
        item = manifest[variant]
        require(item['tensor_layout'] == base['tensor_layout'], 'Calibrated/control layouts differ')
        for key in ['source_sha256', 'tool_sha256', 'quantization_type', 'pure']:
            require(item['inputs'][key] == base['inputs'][key], f'Quantization mismatch: {key}')
    recipe = {'Q2': 'Q2_K', 'Q4': 'Q4_K_M', 'Q6': 'Q6_K'}[precision]
    require(base['inputs']['quantization_type'] == recipe and not base['inputs']['pure'], 'Unexpected recipe')
    local = csv(f'{folder}/local_errors/{precision.lower()}/local_errors.csv')
    coverage(local, ['variant', 'domain', 'layer'])
    require(len(local) == 1344, 'Expected four variants x three domains x 112 projections')
    for row in local.itertuples():
        require(row.stored_type == manifest[row.variant]['tensor_layout'][row.tensor], 'Stored type mismatch')
    controls = local[local.variant.eq('control')][['domain', 'tensor', 'weighted_error', 'reference_energy']]
    check = local.merge(controls, on=['domain', 'tensor'], suffixes=('', '_actual_control'), validate='many_to_one')
    require(np.allclose(check.control_error, check.weighted_error_actual_control), 'Local control error mismatch')
    require(np.allclose(check.reference_energy, check.reference_energy_actual_control), 'Local reference mismatch')
    pooled = local.groupby(['variant', 'domain', 'layer'], as_index=False)[['weighted_error', 'control_error', 'reference_energy']].sum()
    pooled = pooled[pooled.variant.ne('control')].rename(columns={'layer': 'block'})
    require((pooled.control_error > 0).all() and (pooled.reference_energy > 0).all(), 'Invalid denominator')
    pooled['error_removed'] = pooled.control_error - pooled.weighted_error
    pooled['recovery_pct'] = 100 * pooled.error_removed / pooled.control_error
    pooled['removed_pct_reference'] = 100 * pooled.error_removed / pooled.reference_energy
    pooled['control_relative_error_pct'] = 100 * pooled.control_error / pooled.reference_energy
    joined = pooled.merge(prediction[['domain', 'block', 'sensitivity_nats', 'sensitivity_nats_rank']],
                          on=['domain', 'block'], how='left', validate='many_to_one')
    require(len(joined) == 144 and joined.sensitivity_nats.notna().all(), 'Incomplete preservation join')
    return joined, manifest


def load_outcomes(manifest, precision='Q4'):
    folder = 'results/Imatrix_Recovery_Sweep_full_kl4'
    pub = 'results/published/imatrix_recovery'
    audit = read_json(f'{pub}/experiment_manifest.json')
    gain_path = f'{pub}/benchmark_gains.csv'
    gains = csv(gain_path)
    require(INPUTS[gain_path] == audit['tables']['benchmark_gains']['sha256'], 'Published paired table hash mismatch')
    # Check Q4 source identities against the saved publication audit.
    for variant in ['control'] + DOMAINS:
        for filename in ['metrics.json', 'metadata.json', 'samples.json', 'kl.json']:
            rel = f'{folder}/mixed-{precision.lower()}-{variant}/{filename}'
            actual = hashlib.sha256(source(rel).read_bytes()).hexdigest()
            require(actual == audit['input_hashes'][rel.replace('/', '\\')], 'Published source hash mismatch: ' + rel)
    scores, kl = [], []
    fields = {'gsm8k': 'exact_match,flexible-extract', 'ifeval': 'prompt_level_strict_acc,none',
              'hellaswag': 'acc_norm,none'}
    counts = {'gsm8k': 1000, 'ifeval': 541, 'hellaswag': 2000}
    for variant in ['control'] + DOMAINS:
        metrics = read_json(f'{folder}/mixed-{precision.lower()}-{variant}/metrics.json')
        config = read_json(f'{folder}/mixed-{precision.lower()}-{variant}/config.json')
        require(config['pure_quantization'] is False and config['quantization_type'] == manifest[variant]['inputs']['quantization_type'], 'Evaluation recipe mismatch')
        for task, field in fields.items():
            require(metrics[task]['sample_len'] == counts[task], 'Benchmark sample count mismatch')
            scores.append(dict(variant=variant, task=task, score_pct=100 * metrics[task][field], n_examples=counts[task]))
        result = read_json(f'{folder}/mixed-{precision.lower()}-{variant}/kl.json')
        require(result['direction'] == 'BF16 || evaluated model' and result['units'] == 'nats', 'Outcome KL mismatch')
        for domain in DOMAINS:
            d = result['domains'][domain]
            require(d['tokens'] == 1020, 'Outcome KL token mismatch')
            kl.append(dict(variant=variant, domain=domain, kl_nats=d['mean'], tokens=d['tokens']))
        # Raw metric JSONs do not store model hashes; the audited quantization manifest does.
        require(manifest[variant]['inputs']['source_sha256'] == audit['source_sha256'], 'Source identity mismatch')
    scores = pd.DataFrame(scores)
    control = scores[scores.variant.eq('control')][['task', 'score_pct']].rename(columns={'score_pct': 'control_score_pct'})
    scores = scores.merge(control, on='task', validate='many_to_one')
    gains = gains[gains.precision.eq(precision)].merge(scores[scores.variant.ne('control')],
                    on=['variant', 'task', 'n_examples'], how='left', validate='one_to_one')
    require(len(gains) == 9 and gains.score_pct.notna().all(), 'Incomplete benchmark join')
    require(np.allclose(gains.gain_pp, gains.score_pct - gains.control_score_pct), 'Benchmark gain mismatch')
    kl = pd.DataFrame(kl)
    control = kl[kl.variant.eq('control')][['domain', 'kl_nats']].rename(columns={'kl_nats': 'control_kl_nats'})
    kl = kl.merge(control, on='domain', validate='many_to_one')
    kl['reduction_pct'] = 100 * (1 - kl.kl_nats / kl.control_kl_nats)
    return gains, kl


def plots(prediction, preservation, gains, kl):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for col, domain in enumerate(DOMAINS):
        g = prediction[prediction.domain.eq(domain)]
        for row, x in enumerate(['input_energy', 'local_increment_pct']):
            ax = axes[row, col]
            ax.scatter(g[x], g.sensitivity_nats, color=COLORS[domain])
            for d in g.itertuples():
                if d.block in {0, 1, 2, 8, 15}:
                    ax.annotate(str(d.block), (getattr(d, x), d.sensitivity_nats), xytext=(3, 3), textcoords='offset points', fontsize=8)
            ax.set_title(f'{domain.capitalize()} | Spearman rho = {rho(g, x):.2f}')
            ax.set_xlabel('Summed BF16 input energy' if row == 0 else 'Q4 minus Q8 local error / BF16 energy (%)')
            ax.set_ylabel('Added output KL (nats)')
            ax.grid(alpha=.18)
    fig.suptitle('1. Do imatrix-derived scores predict sensitive blocks?\nOne block at Q4_K; rest Q8_0. Selected block IDs labeled; all 16 blocks included.')
    save_plot(fig, '01_prediction_vs_sensitivity')

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), layout='constrained')
    norm = Normalize(vmin=min(0, preservation.recovery_pct.min()), vmax=max(1, preservation.recovery_pct.max()))
    for r, domain in enumerate(DOMAINS):
        for c, variant in enumerate(DOMAINS):
            g = preservation[preservation.domain.eq(domain) & preservation.variant.eq(variant)]
            ax = axes[r, c]
            points = ax.scatter(g.sensitivity_nats, g.removed_pct_reference, c=g.recovery_pct, norm=norm, cmap='viridis')
            for d in g.itertuples():
                if d.block in {0, 1, 2, 8, 15}:
                    ax.annotate(str(d.block), (d.sensitivity_nats, d.removed_pct_reference), xytext=(3, 3), textcoords='offset points', fontsize=8)
            ax.axhline(0, color='gray', lw=.6)
            ax.set_title(f'{variant} calibration / {domain} measurement\nrho = {rho(g, "removed_pct_reference"):.2f}')
            ax.set_xlabel('Measured block sensitivity (added KL, nats)')
            ax.set_ylabel('Error removed / BF16 energy (%)')
            ax.grid(alpha=.18)
    fig.colorbar(points, ax=axes, label='Control local error removed (%)', shrink=.6)
    fig.suptitle('2. Does calibration preserve the blocks measured as sensitive?\nQ4_K sensitivity on Q8 background versus reconstruction recovery in standard Q4_K_M models')
    save_plot(fig, '02_sensitive_block_preservation')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    tasks = ['gsm8k', 'ifeval', 'hellaswag']
    for i, variant in enumerate(DOMAINS):
        g = gains[gains.variant.eq(variant)].set_index('task').loc[tasks]
        x = np.arange(3) + (i - 1) * .23
        axes[0].errorbar(x, g.gain_pp, yerr=np.array([g.gain_pp - g.paired_ci95_low_pp, g.paired_ci95_high_pp - g.gain_pp]),
                         fmt='o', capsize=4, color=COLORS[variant], label=variant + ' calibration')
        for xx, (_, d) in zip(x, g.iterrows()):
            if d.passes_holm_0_05:
                axes[0].annotate('*', (xx, d.paired_ci95_high_pp), xytext=(0, 3), textcoords='offset points', ha='center')
        k = kl[kl.variant.eq(variant)].set_index('domain').loc[DOMAINS]
        axes[1].bar(x, k.reduction_pct, width=.22, color=COLORS[variant], label=variant + ' calibration')
    for ax in axes:
        ax.axhline(0, color='gray', lw=.8)
        ax.grid(axis='y', alpha=.18)
        ax.set_xticks(range(3))
        ax.legend(fontsize=8)
    axes[0].set_xticklabels(['GSM8K', 'IFEval', 'HellaSwag'])
    axes[0].set_ylabel('Benchmark gain (percentage points)')
    axes[0].set_title('Task performance: paired 95% bootstrap intervals\n* passes existing Holm correction over 54 comparisons')
    axes[1].set_xticklabels(['Code text', 'Math text', 'Chat text'])
    axes[1].set_ylabel('Output KL reduction (%)')
    axes[1].set_title('Held-out output fidelity: 1,020 tokens per domain\nDescriptive reductions; no paired KL intervals available')
    fig.suptitle('3. Does imatrix improve whole-model quality?\nStandard Q4_K_M, compared with the matched no-imatrix control; positive is better')
    save_plot(fig, '03_whole_model_benefit')


def simple_plots(prediction, preservation):
    """Use shared rank and percent scales; retain the original technical figures."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), layout='constrained')
    columns = ['input_energy', 'local_increment_pct', 'sensitivity_nats']
    labels = ['Activation\nenergy', 'Weighted-error\nprediction', 'Measured\nsensitivity']
    rank_rows = []
    for ax, domain in zip(axes, DOMAINS):
        g = prediction[prediction.domain.eq(domain)].copy()
        positions = np.arange(16)
        offsets = [-.25, 0, .25]
        colors = ['#a5bed6', '#536f9b', '#e29038']
        for col, offset, color, label in zip(columns, offsets, colors, labels):
            # Average ranks preserve exact ties. 100 = rank 1; 0 = rank 16.
            ranks = g.set_index('block')[col].rank(ascending=False, method='average').reindex(range(16))
            score = 100 * (16 - ranks) / 15
            ax.barh(positions + offset, score, height=.24, color=color, label=label.replace('\n', ' '))
            for block in range(16):
                rank_rows.append(dict(domain=domain, block=block, measure=col,
                                      rank=float(ranks.loc[block]), normalized_rank_score=float(score.loc[block])))
        ax.set_yticks(positions, [f'Block {b}' for b in range(16)])
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('Position in the ranking (0-100)')
        ax.set_title(domain.capitalize() + ' text')
        ax.grid(axis='x', alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3)
    fig.suptitle('Which blocks look most sensitive?\n100 = highest ranked; 0 = lowest ranked. Similar bar lengths mean the rankings agree.\nAll methods use the same rank scale; bar lengths do not show the size of the damage.', fontsize=13)
    save_plot(fig, 'simple_01_block_rankings')
    pd.DataFrame(rank_rows).to_csv(OUT / 'normalized_block_rankings.csv', index=False)

    # Identical axes across all nine panels; denominators are each block's control error.
    upper = max(5, np.ceil(preservation.recovery_pct.max() / 5) * 5)
    lower = min(0, np.floor(preservation.recovery_pct.min() / 5) * 5)
    for domain in DOMAINS:
        order = prediction[prediction.domain.eq(domain)].sort_values(
            ['sensitivity_nats', 'block'], ascending=[False, True]).block.tolist()
        top = set(order[:3])
        fig, axes = plt.subplots(1, 3, figsize=(14, 7), layout='constrained')
        for ax, variant in zip(axes, DOMAINS):
            g = preservation[preservation.domain.eq(domain) & preservation.variant.eq(variant)].set_index('block').loc[order]
            bars = ax.barh(range(16), g.recovery_pct,
                           color=['#e29038' if b in top else '#7197b8' for b in order])
            ax.bar_label(bars, labels=[f'{v:.1f}%' for v in g.recovery_pct], padding=3, fontsize=8)
            ax.set_yticks(range(16), [f'Block {b}' for b in order])
            ax.invert_yaxis()
            ax.set_xlim(lower, upper + 3)
            ax.set_xlabel('Original block error removed (%)')
            ax.set_title(variant.capitalize() + ' calibration')
            ax.axvline(0, color='gray', lw=.7)
            ax.grid(axis='x', alpha=.18)
        fig.suptitle(f'How much does imatrix help each block? | Measured on {domain} text\nBlocks ordered from most to least sensitive; orange = three most sensitive.\n20% means one fifth of that block\'s original error was removed. All charts use the same scale.', fontsize=13)
        save_plot(fig, f'simple_02_{domain}_block_recovery')


def audience_overview(prediction, preservation, gains, kl):
    """Compact presentation figure; aggregate quantities before taking ratios."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout='constrained')
    blue, orange = '#527fa2', '#df8b35'
    x = np.arange(3)
    rows = []
    ax = axes[0, 0]
    for offset, measure, color, label in [(-.18, 'input_energy', blue, 'Activation energy'),
                                          (.18, 'local_increment_pct', orange, 'Weighted-error prediction')]:
        values = []
        for domain in DOMAINS:
            g = prediction[prediction.domain.eq(domain)]
            top = lambda column: set(g.sort_values([column, 'block'], ascending=[False, True]).head(3).block)
            hits = len(top(measure) & top('sensitivity_nats'))
            values.append(100 * hits / 3)
            rows.append(dict(question='prediction', domain_or_task=domain, series=measure,
                             value=100 * hits / 3, unit='percent of measured top three recovered',
                             numerator=hits, denominator=3))
        bars = ax.bar(x + offset, values, width=.34, color=color, label=label)
        ax.bar_label(bars, labels=[f'{int(round(v * 3 / 100))}/3' for v in values], padding=3)
    ax.set_title('1. Can we identify the sensitive blocks?', fontweight='bold')
    ax.set_ylabel('Measured top 3 correctly identified (%)')
    ax.set_ylim(0, 100)
    ax.set_xticks(x, ['Code', 'Math', 'Chat'])
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[0, 1]
    for offset, sensitive, color, label in [(-.18, True, orange, '3 most sensitive blocks'),
                                           (.18, False, blue, 'Other 13 blocks')]:
        values = []
        for domain in DOMAINS:
            g = preservation[preservation.domain.eq(domain)]
            measured = prediction[prediction.domain.eq(domain)]
            top = set(measured.sort_values(['sensitivity_nats', 'block'], ascending=[False, True]).head(3).block)
            selected = g[g.block.isin(top).eq(sensitive)]
            numerator, denominator = selected.error_removed.sum(), selected.control_error.sum()
            values.append(100 * numerator / denominator)
            rows.append(dict(question='preservation', domain_or_task=domain, series=label,
                             value=values[-1], unit='percent of pooled original local error removed',
                             numerator=numerator, denominator=denominator))
        bars = ax.bar(x + offset, values, width=.34, color=color, label=label)
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
    ax.set_title('2. Do sensitive blocks benefit more?', fontweight='bold')
    ax.set_ylabel('Original local error removed (%)')
    ax.set_ylim(0, 25)
    ax.set_xticks(x, ['Code', 'Math', 'Chat'])
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1, 0]
    for offset, kind, color, label in [(-.18, 'local', blue, 'Local reconstruction error'),
                                      (.18, 'output', orange, 'Output KL')]:
        values = []
        for variant in DOMAINS:
            if kind == 'local':
                g = preservation[preservation.variant.eq(variant)]
                numerator, denominator = g.error_removed.sum(), g.control_error.sum()
            else:
                g = kl[kl.variant.eq(variant)]
                numerator, denominator = (g.control_kl_nats - g.kl_nats).sum(), g.control_kl_nats.sum()
            values.append(100 * numerator / denominator)
            rows.append(dict(question='overall_error', domain_or_task=variant, series=kind,
                             value=values[-1], unit='percent of matched control error removed',
                             numerator=numerator, denominator=denominator))
        bars = ax.bar(x + offset, values, width=.34, color=color, label=label)
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
    ax.set_title('3. Does the model retain more fidelity?', fontweight='bold')
    ax.set_ylabel('Reduction versus no-imatrix model (%)')
    ax.set_ylim(0, 25)
    ax.set_xticks(x, ['Code calibration', 'Math calibration', 'Chat calibration'])
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1, 1]
    task_names = ['gsm8k', 'ifeval', 'hellaswag']
    mean_gains = []
    for task in task_names:
        g = gains[gains.task.eq(task)]
        mean_gains.append(g.gain_pp.mean())
        rows.append(dict(question='benchmark', domain_or_task=task, series='mean across calibrations',
                         value=g.gain_pp.mean(), unit='percentage points',
                         numerator=g.gain_pp.sum(), denominator=len(g)))
    bars = ax.bar(x, mean_gains, color=blue, width=.5, alpha=.8)
    for i, task in enumerate(task_names):
        g = gains[gains.task.eq(task)].set_index('variant').loc[DOMAINS]
        for j, variant in enumerate(DOMAINS):
            ax.scatter(i + (j - 1) * .1, g.loc[variant, 'gain_pp'], color=COLORS[variant],
                       edgecolors='white', linewidth=.5, s=45, zorder=4,
                       label=variant.capitalize() + ' calibration' if i == 0 else None)
        ax.annotate(f'Mean {mean_gains[i]:+.1f} pp', (i, max(0, mean_gains[i], g.gain_pp.max())),
                    xytext=(0, 7), textcoords='offset points', ha='center', fontsize=9)
    ax.axhline(0, color='gray', lw=.8)
    ax.set_ylim(-2.5, 9.5)
    ax.set_title('4. Does task performance improve?', fontweight='bold')
    ax.set_ylabel('Score gain (percentage points)')
    ax.set_xticks(x, ['GSM8K\nMath reasoning', 'IFEval\nInstruction following', 'HellaSwag\nCommonsense'])
    ax.legend(loc='upper right', fontsize=8)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Does imatrix identify and preserve the blocks that matter?\nLlama-3.2-1B-Instruct | Q4 analysis | Higher bars indicate more agreement or benefit', fontsize=15, fontweight='bold')
    fig.supxlabel('Panel 2 pools calibration choices; panel 3 pools measurement domains.\nPanel 4: bars = mean of three calibrations; dots = individual results, not confidence intervals. See report for uncertainty and protocol differences.', fontsize=9)
    save_plot(fig, '00_audience_overview')
    pd.DataFrame(rows).to_csv(OUT / 'audience_overview.csv', index=False)


def main():
    prediction = load_prediction()
    preservation, quantization = load_preservation(prediction)
    gains, kl = load_outcomes(quantization)
    correlations = []
    for domain, g in prediction.groupby('domain', sort=False):
        for predictor in ['input_energy', 'local_increment_pct']:
            top = lambda col: set(g.sort_values([col, 'block'], ascending=[False, True]).head(3).block)
            correlations.append(dict(domain=domain, predictor=predictor, n_blocks=16,
                spearman=rho(g, predictor), spearman_without_block1=rho(g[g.block.ne(1)], predictor),
                top3_overlap=len(top(predictor) & top('sensitivity_nats'))))
    correlations = pd.DataFrame(correlations)
    recovery_corr = []
    top_rows = []
    for (domain, variant), g in preservation.groupby(['domain', 'variant']):
        recovery_corr.append(dict(domain=domain, calibration=variant, n_blocks=16,
            rho_absolute_removed=rho(g, 'error_removed'), rho_fraction_removed=rho(g, 'recovery_pct'),
            rho_normalized_removed=rho(g, 'removed_pct_reference'),
            rho_normalized_without_block1=rho(g[g.block.ne(1)], 'removed_pct_reference'),
            blocks_improved=int(g.error_removed.gt(0).sum())))
        top = g.sort_values(['sensitivity_nats', 'block'], ascending=[False, True]).head(3)
        rest = g[~g.block.isin(top.block)]
        for label, group in [('top3_sensitive', top), ('other13', rest)]:
            top_rows.append(dict(domain=domain, calibration=variant, group=label,
                blocks=','.join(map(str, sorted(group.block))), control_error=group.control_error.sum(),
                error_removed=group.error_removed.sum(),
                recovery_pct=100 * group.error_removed.sum() / group.control_error.sum()))
    recovery_corr = pd.DataFrame(recovery_corr)
    stability = []
    for a, b in [('code', 'math'), ('code', 'chat'), ('math', 'chat')]:
        for column in ['input_energy', 'local_increment_pct', 'sensitivity_nats']:
            pivot = prediction.pivot(index='block', columns='domain', values=column)
            stability.append(dict(domain_a=a, domain_b=b, measure=column,
                spearman=pivot[a].rank().corr(pivot[b].rank())))
    outputs = {'prediction_blocks.csv': prediction, 'prediction_agreement.csv': correlations,
               'preservation_blocks.csv': preservation, 'preservation_agreement.csv': recovery_corr,
               'sensitive_group_recovery.csv': pd.DataFrame(top_rows),
               'domain_rank_stability.csv': pd.DataFrame(stability),
               'benchmark_gains.csv': gains, 'output_kl.csv': kl}
    for name, frame in outputs.items():
        require(not frame.isna().any().any(), f'Unexpected missing values in {name}')
        frame.to_csv(OUT / name, index=False)
    plots(prediction, preservation, gains, kl)
    simple_plots(prediction, preservation)
    audience_overview(prediction, preservation, gains, kl)
    write_report(prediction, preservation, correlations, recovery_corr, gains, kl)
    tracked = []
    for ref in ['HEAD', 'layers_sneha', 'origin/layers_sneha']:
        result = subprocess.run(['git', 'rev-parse', ref], cwd=ROOT, capture_output=True, text=True)
        tracked.append({'ref': ref, 'commit': result.stdout.strip(), 'available': result.returncode == 0})
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()
                 if p.is_file() and p.name != 'manifest.json'}
    manifest = {'scope': 'Llama-3.2-1B-Instruct; Q4 only; existing measurements; no new inference',
                'source_model_sha256': quantization['control']['inputs']['source_sha256'],
                'input_sha256': INPUTS, 'output_sha256': artifacts, 'local_git_refs': tracked,
                'checks': ['complete seven-projection coverage', 'unique joins; no missing values',
                    'block KL checked against original Q8 control', 'local-error source table hashes',
                    'matched control/calibrated tensor layouts and quantization inputs',
                    'local control errors and stored types checked against original records',
                    'paired benchmark table and Q4 raw input hashes match publication audit',
                    'benchmark gains recomputed from raw metrics'],
                'unverified': ['cross-machine intervention source/text/build identity',
                    'fresh model-weight payloads and new model inference', 'run-to-run variability']}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Created Q4 Llama synthesis:', OUT)
    print(recovery_corr.to_string(index=False))


def write_report(prediction, preservation, correlations, recovery_corr, gains, kl):
    summary = []
    for variant in DOMAINS:
        g = preservation[preservation.variant.eq(variant)]
        k = kl[kl.variant.eq(variant)]
        summary.append(dict(calibration=variant,
            pooled_local_recovery_pct=100 * g.error_removed.sum() / g.control_error.sum(),
            mean_KL_reduction_pct=100 * (1 - k.kl_nats.mean() / k.control_kl_nats.mean())))
    summary = pd.DataFrame(summary)
    positives = int(preservation.error_removed.gt(0).sum())
    report = f'''# Final analysis: Q4 Llama quantization

This folder connects existing imatrix statistics, Sneha's measured block sensitivity,
local reconstruction recovery, and whole-model evaluations. No models were run or
quantized to produce it. Scope: Llama-3.2-1B-Instruct, blocks 0-15, code/math/chat.

## Start here: the combined story

**We predict which blocks matter, check those predictions against Sneha's
measurements, and then ask whether imatrix preserves them and improves the model.**

![Aggregated analysis overview](00_audience_overview.png)

Read the four panels from left to right, top to bottom:

1. **Prediction is partial.** Each method identifies only one or two of the three
   most sensitive blocks in a domain. The labels show the count out of three;
   bar heights show the equivalent percentage. This checks the highest-ranked
   blocks, rather than agreement across the full ranking.
2. **Preservation is broad.** Sensitive blocks benefit more as a group for code,
   but less for math and chat. Imatrix does not consistently concentrate its
   proportional improvements in the most sensitive blocks.
3. **Overall errors decrease.** Local reconstruction error falls by about 16%
   and output KL falls by 13-16%. These are separately normalized reductions in
   different errors; matching heights do not mean they measure the same thing.
4. **Task gains vary.** The clearest improvement is on GSM8K. Bars average the
   three calibration choices; dots retain their individual results, including
   the +7 pp math-calibrated GSM8K result. These dots are not uncertainty bounds.
   Only that Q4 result passes the existing correction across all 54 comparisons;
   paired intervals for every comparison appear in the detailed results below.

**Normalization:** panel 1 divides correct identifications by three. Panel 2
divides summed error removed by summed original control error, pooling the three
calibrations equally before division. It is not an average of block percentages.
The top-three group is selected separately for each domain using measured
sensitivity. Panel 3 also uses ratios of summed errors, pooling the three domains.
Panel 4 uses percentage-point score changes against matched controls; averaging
is across calibration choices within each task, never across different tasks.
The two group sizes in panel 2 differ, but each is normalized to its own original
error. They are not matched for parameter count or memory cost.

**Main conclusion:** imatrix helps overall reconstruction and output fidelity,
and sometimes task performance. Its block predictions are imperfect, and the
benefit is not consistently concentrated in the most sensitive blocks.

**Important boundary:** Sneha measures one Q4_K block on a Q8 background, while
calibration benefit is measured in standard Q4_K_M models. This is an exploratory
connection between experiments, not proof of a block-level cause of model gains.

For presentation, use `00_audience_overview.png`. Exact plotted values and
normalization numerators/denominators are in `audience_overview.csv`. All charts
are PNG only.

<details>
<summary>Detailed results, methods, and supporting block-level charts</summary>

## Main findings

- Imatrix-derived local error predicts measured sensitivity better than input energy
  for code and math at Q4; chat shows the opposite pattern. Prediction is partial.
- Calibration lowers local error in {positives} of the 144 block/domain/calibration
  combinations, with roughly 16% pooled error reduction for each calibration.
  It does **not consistently give the most sensitive blocks the largest recovery**:
  normalized-recovery rank agreement is near zero for code/chat and negative for
  math. The three most sensitive blocks have a larger pooled percentage recovery
  than the remaining blocks for code, but smaller recovery for math and chat.
- Whole-model mean output KL falls by about 13-16% for all three calibrations.
- Lower reconstruction error and lower output KL do not guarantee a benchmark gain.
  Math calibration improves GSM8K by 7.0 percentage points, but other task effects
  are smaller or mixed. This does not establish a universal best calibration.
- These comparisons show associations. They do not establish that recovery in a
  particular block caused the whole-model gain.

## 1. Do imatrix scores predict sensitive blocks?

![Normalized block rankings](simple_01_block_rankings.png)

**How to read this chart:** each block has three bars: activation energy,
weighted-error prediction, and measured sensitivity. All bars use the same
normalized rank scale: 100 is first place and 0 is last place among 16 blocks.
If a prediction bar and the measured bar have similar lengths, their rankings
agree for that block. Orange is the measured result from Sneha's experiment.
Average ranks are used for tied values.

This normalization compares ordering, not the size of the original measurements.
For example, a bar twice as long does not mean twice the damage. A zero bar means
lowest rank, not zero sensitivity. Raw values remain in `prediction_blocks.csv`;
normalized scores are in `normalized_block_rankings.csv`.

**What to notice:** block 1 is consistently measured as most sensitive. The
weighted-error prediction follows the measured ordering more closely than energy
for code and math, but agreement is incomplete.

<details>
<summary>Optional technical scatter plot</summary>

![Prediction versus sensitivity](01_prediction_vs_sensitivity.png)

</details>

The target is the output KL added when one block is Q4_K and all other blocks are
Q8_0, relative to an all-Q8_0 model. Higher means more sensitive. Both KL values
use BF16 as reference: this is a difference of two KL values, not KL(Q8 || Q4).

Two predictors are kept separate:

1. **Input energy:** sum of BF16 input second moments across the block's seven
   projections. It measures activation scale, not proven sensitivity.
2. **Normalized local error increment:** 100 times (summed pure-Q4 weighted error
   minus summed pure-Q8 weighted error), divided by summed BF16 diagonal reference
   energy. This includes actual weight distortion and aligns the local baseline
   with the Q8 intervention control. It is still a local approximation.

{table(correlations)}

Spearman measures rank agreement (1 = perfect, 0 = no monotonic agreement).
Each correlation uses 16 blocks, or 15 after excluding block 1. No correlation
significance tests are claimed. Top-three boundary ties are resolved by lower block
index. Rank correlation itself uses average ranks for ties.

Block 1 is the most sensitive measured block in all three domains. Block 0 is an
important counterexample to the local-error ranking. The outlier-exclusion column
checks how much agreement depends on block 1. `domain_rank_stability.csv` also
compares rankings between domains at this fixed Q4 setting.

## 2. Does imatrix preserve sensitive blocks?

**How to read these charts:** longer bars mean more of a block's original
reconstruction error was removed by imatrix. Every panel uses the same percentage
scale. Each percentage is divided by that block's own no-imatrix error, so large
and small blocks can be compared by proportional improvement. Zero means no
improvement; 100% would mean all local error was removed. This does not mean
100% benchmark accuracy or full recovery of the model's behavior.

Blocks are ordered from most to least sensitive within each measurement domain.
The orange bars identify its three most sensitive blocks. Compare orange bars
with the other bars: they are not consistently longer. Each column uses a
different calibration imatrix, while the text used to measure error is fixed
within each figure.

![Recovery measured on code](simple_02_code_block_recovery.png)

![Recovery measured on math](simple_02_math_block_recovery.png)

![Recovery measured on chat](simple_02_chat_block_recovery.png)

<details>
<summary>Optional technical scatter plot using a different normalization</summary>

![Sensitive block preservation](02_sensitive_block_preservation.png)

This original chart divides the amount of error removed by BF16 reference energy.
It answers a different question from the simpler charts, which divide by the
block's original control error. Their percentages should not be equated.

</details>

Each panel matches measured sensitivity to block reconstruction recovery in the
standard Q4_K_M sweep. Columns identify the imatrix used to quantize the model;
rows identify the domain used to measure reconstruction error. These are separate
dimensions: all three calibrations are measured under all three domains.

Errors and reference energies are summed over seven projections before division.
The control and calibrated models have identical stored tensor-type layouts.
Here preservation means less error at the same stored precision, rather than more
bits assigned to the sensitive blocks.

All 16 blocks are plotted; only blocks 0, 1, 2, 8 and 15 are labeled to keep the
figures readable. Exact values and ranks for every block are in the CSVs.

- **Absolute removed:** control error minus calibrated error; depends on scale.
- **Fraction removed (%):** 100 times absolute removed / control error.
- **Normalized removed (%):** 100 times absolute removed / BF16 reference energy.
  This is the plot's vertical axis. Color shows the fraction removed.

{table(recovery_corr)}

Positive rho means blocks measured as more sensitive tend to show larger recovery
under that definition; negative rho means the reverse. A highly sensitive block
does not necessarily have the most removable local error. Percent recovery can be
large for a block with little starting error, so inspect all three definitions.
`sensitive_group_recovery.csv` compares pooled recovery in the three most sensitive
blocks against the other thirteen, separately for each domain and calibration.
It is descriptive, not a random-assignment or size-matched protection experiment.

## 3. Does the whole model improve?

![Whole-model benefit](03_whole_model_benefit.png)

These comparisons use standard Q4_K_M models with and without imatrix, created
from the same BF16 source. Historical pure-Q4 scores are not the control.

{table(gains[['variant', 'task', 'control_score_pct', 'score_pct', 'gain_pp', 'paired_ci95_low_pp', 'paired_ci95_high_pp', 'passes_holm_0_05']])}

Scores use flexible-extraction exact match for GSM8K (1,000 examples), prompt-level
strict accuracy for IFEval (541), and length-normalized accuracy for HellaSwag
(2,000). Gains and confidence bounds are percentage points. Intervals and exact
paired tests come from the existing audited publication tables. Holm adjustment
retains the original family of 54 comparisons across all six precisions, even
though this report displays Q4 only. The intervals are pointwise; they do not
measure execution-to-execution variability.

{table(summary)}

Pooled local recovery is a ratio of summed errors across all blocks and domains.
Mean KL gives each domain equal weight before computing the reduction. These
percentages describe different measurements and must not be equated. Per-domain
KL values and reductions are in `output_kl.csv`; all use 1,020 scored tokens.
Do not correlate just the three calibration-level means to claim a mechanism.

## Limits and gaps

- **Different quantization settings:** measured sensitivity uses a pure-Q4_K block
  on a Q8 background. Preservation and benchmark benefit use standard Q4_K_M,
  including Q6_K projections. Rankings may not transfer unchanged between these
  settings. The mixed models' tensor layouts match each other, not the intervention.
- **Local approximation:** weighted error is sum_j E[x_j^2] times
  sum_i (W_quant[i,j] - W_BF16[i,j])^2. Reference energy replaces the weight
  difference by W_BF16. Input correlations and downstream propagation are omitted.
  Shared projection inputs can be counted several times; block sums are proxies,
  not measured residual-stream energy. Q4-minus-Q8 error is not direct Q4/Q8
  reconstruction error. Exported energy statistics have limited decimal precision.
- **Unequal calibration:** code/math/chat use 49/97/699 chunks of 512 tokens.
  Domain and calibration quantity are confounded. Calibration activations and
  held-out KL text serve different purposes and are not identical samples.
- **Provenance:** original intervention JSONs lack hashes sufficient to verify the
  exact source checkpoint, text and build against the recovery machine. Within-run
  KL direction, reference path, text path and token coverage were checked. This
  remains exploratory cross-experiment evidence.
- **Uncertainty:** blocks, domains and calibration variants are related observations,
  not 144 independent trials. No significance is assigned to block correlations.
  The intervention tool uncertainties are retained but not converted into paired
  sensitivity confidence intervals. A shared Q8 baseline does not justify treating
  the two KL estimates as independent.
- **Execution variability:** the existing Q8 audit reports identical tensor payloads
  with small benchmark differences. Small observed gains need repetition. This
  report reads saved measurements and audits; it does not rehash model weights.
- **Scope:** only Q4 and Llama are synthesized here. No conclusions about other
  precisions, context lengths, hardware or models follow from this folder. Code
  calibration lacks a code benchmark; IFEval/HellaSwag are not direct chat tests.

## What is complete, and what comes next?

Complete here: matched block joins, two prediction comparisons, sensitive-block
recovery analysis, domain-ranking stability, and matched whole-model benefit.
The main remaining scientific gap is a selective intervention: protect blocks
chosen by the predictor and compare against random or less-sensitive choices at
a comparable actual size budget, keeping calibration fixed. That would test
whether choosing those blocks helps, beyond ordinary imatrix calibration.

Before building that experiment, repeat the synthesis at Q2 and Q6 to see whether
the Q4 relationship persists. Longer held-out KL texts, equal calibration budgets
and repeated evaluations would strengthen conclusions, but were not run here.

## Files and reproduction

- `prediction_blocks.csv`: 48 matched domain/block rows, predictor values and ranks.
- `prediction_agreement.csv`: six predictor/domain comparisons and outlier checks.
- `preservation_blocks.csv`: 144 calibration/domain/block recovery observations.
- `preservation_agreement.csv`: nine sensitivity/recovery comparisons.
- `sensitive_group_recovery.csv`: top-three versus remaining-block pooled recovery.
- `domain_rank_stability.csv`: nine domain-pair/ranking comparisons at Q4.
- `benchmark_gains.csv`: nine task/calibration gains with saved paired uncertainty.
- `output_kl.csv`: twelve variant/domain KL rows, including matched controls.
- Three original figures, supplied as PNG only.
- Four simpler figures, supplied as PNG only: normalized block rankings
  and block error reduction separately for code, math and chat.
- `normalized_block_rankings.csv`: exact ranks and 0-100 rank scores for all bars.
- `00_audience_overview.png`: compact four-panel presentation figure.
- `audience_overview.csv`: aggregated values and normalization inputs.
- `manifest.json`: source/output hashes, local branch refs, checks and limitations.
- `build_analysis.py`: rebuild with `python results/final_analysis/build_analysis.py`
  from the repository root, using the project's numpy/pandas/matplotlib environment.
  The script writes only into this folder and launches no model processes.

Sources remain in `results/Block_Sensitivity_Differences/`,
`results/imatrix_full_pure_sweep/`, `results/important plots/`,
`results/Imatrix_Recovery_Sweep_full_kl4/`, and
`results/published/imatrix_recovery/`. The build validates the consumed paired
benchmark table and Q4 raw evaluation hashes against the existing publication
audit, checks complete block coverage, and rejects duplicate or missing join keys.

</details>
'''
    (OUT / 'README.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    from build_question_charts import main as build_questions
    build_questions()
