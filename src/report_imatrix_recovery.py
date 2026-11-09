"""Audit the completed imatrix sweep and publish compact recovery figures/data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
PRECISIONS = ['Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8']
VARIANTS = ['code', 'math', 'chat']
TASKS = {'gsm8k': ('exact_match', 'flexible-extract', 1000, 'GSM8K'),
         'ifeval': ('prompt_level_strict_acc', 'none', 541, 'IFEval'),
         'hellaswag': ('acc_norm', 'none', 2000, 'HellaSwag (length normalized)')}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sample_scores(records, task):
    key, selected_filter, count, _ = TASKS[task]
    result = {}
    for sample in records:
        if sample.get('filter') != selected_filter:
            continue
        doc = int(sample['doc_id'])
        if doc in result:
            raise ValueError('Repeated selected sample: ' + task)
        value = float(sample[key])
        if value not in (0, 1):
            raise ValueError('Expected binary example scores')
        result[doc] = (value, sample['doc_hash'], sample['prompt_hash'])
    if set(result) != set(range(count)):
        raise ValueError('Sample IDs/count mismatch: ' + task)
    return result


def paired_effect(control, candidate, rng):
    if set(control) != set(candidate):
        raise ValueError('Unmatched sample IDs')
    diffs = []
    for doc in sorted(control):
        if control[doc][1:] != candidate[doc][1:]:
            raise ValueError('Different question/prompt hashes for paired samples')
        diffs.append(candidate[doc][0] - control[doc][0])
    diffs = np.asarray(diffs)
    wins, losses = int((diffs == 1).sum()), int((diffs == -1).sum())
    ties, n = int((diffs == 0).sum()), len(diffs)
    counts = rng.multinomial(n, [wins / n, losses / n, ties / n], size=10000)
    bootstrap = 100 * (counts[:, 0] - counts[:, 1]) / n
    lo, hi = np.quantile(bootstrap, [.025, .975])
    return {'n_examples': n, 'gain_pp': 100 * diffs.mean(), 'paired_ci95_low_pp': lo,
            'paired_ci95_high_pp': hi, 'wins': wins, 'losses': losses, 'ties': ties,
            'mcnemar_exact_p': 1.0 if wins + losses == 0 else binomtest(wins, wins + losses, .5).pvalue}


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1)))
    result = np.empty(len(order))
    result[order] = adjusted
    return result


def transfer_tables(tables):
    """Keep benchmark transfer distinct from matched-domain text KL."""
    models = tables['model_metrics'].set_index(['precision', 'variant'])
    transfer = tables['benchmark_gains'].rename(columns={'variant': 'calibration_imatrix', 'task': 'benchmark'}).copy()
    before, after, meaning = [], [], []
    for row in transfer.itertuples():
        column = row.benchmark + '_score_pct'
        control = models.loc[(row.precision, 'control'), column]
        calibrated = models.loc[(row.precision, row.calibration_imatrix), column]
        if not np.isclose(calibrated - control, row.gain_pp):
            raise ValueError('Transfer gain does not match its same-precision control')
        before.append(control)
        after.append(calibrated)
        if row.calibration_imatrix == 'math' and row.benchmark == 'gsm8k':
            meaning.append('Related math-domain benchmark; distinct calibration/evaluation splits')
        elif row.calibration_imatrix == 'code':
            meaning.append('Cross-domain; no code benchmark was evaluated')
        elif row.calibration_imatrix == 'chat' and row.benchmark == 'ifeval':
            meaning.append('Instruction-following proxy; not a direct chat benchmark')
        else:
            meaning.append('Cross-domain benchmark transfer')
    transfer['control_score_pct'] = before
    transfer['imatrix_score_pct'] = after
    transfer['relationship'] = meaning
    domains = tables['kl_by_domain'].query("variant != 'control'").rename(columns={
        'variant': 'calibration_imatrix', 'domain': 'evaluation_domain'}).copy()
    domains['same_domain'] = domains.calibration_imatrix.eq(domains.evaluation_domain)
    domains['relationship'] = np.where(domains.same_domain, 'Same-domain held-out text', 'Cross-domain held-out text')
    return {'calibration_task_transfer': transfer, 'calibration_domain_transfer': domains}


def audit_and_summarize(folder):
    plan = json.loads((folder / 'sweep_plan.json').read_text())
    status = json.loads((folder / 'sweep_status.json').read_text())
    if set(status) != {p.lower() for p in PRECISIONS} or any(v['status'] != 'ok' for v in status.values()):
        raise ValueError('Full six-precision sweep has not completed successfully')
    baseline_path = ROOT / 'results/Pure_Sweep/BF16-pure-sweep/metrics.json'
    if sha(baseline_path) != plan['bf16_reference']['sha256']:
        raise ValueError('Historical BF16 score file changed')
    sources = {str((folder / 'sweep_plan.json').relative_to(ROOT)): sha(folder / 'sweep_plan.json'),
               str(baseline_path.relative_to(ROOT)): sha(baseline_path)}
    models, effects, kl_rows, local_rows, block_rows, type_rows = [], [], [], [], [], []
    all_completions, protocols = {}, set()
    rng = np.random.default_rng(473)
    for precision in PRECISIONS:
        p = precision.lower()
        quant_path = folder / f'quantization_manifest_{p}.json'
        quant = json.loads(quant_path.read_text())
        sources[str(quant_path.relative_to(ROOT))] = sha(quant_path)
        controls, kl_control = {}, None
        for variant in ['control'] + VARIANTS:
            run = folder / f'mixed-{p}-{variant}'
            completion_path = run / 'comparison_complete.json'
            completion = json.loads(completion_path.read_text())
            all_completions[(precision, variant)] = completion
            for name, expected in completion['outputs'].items():
                path = run / name
                if sha(path) != expected:
                    raise ValueError('Completed output hash mismatch: ' + str(path))
                sources[str(path.relative_to(ROOT))] = expected
            inputs = completion['inputs']
            if inputs['model_sha256'] != quant[variant]['model_sha256'] or quant[variant]['inputs']['source_sha256'] != plan['source_sha256']:
                raise ValueError('Incompatible model/source provenance')
            if quant[variant]['tensor_layout'] != quant['control']['tensor_layout']:
                raise ValueError('Tensor-type layouts differ across variants')
            meta = json.loads((run / 'metadata.json').read_text())
            if meta['actual_unique_samples'] != {t: spec[2] for t, spec in TASKS.items()}:
                raise ValueError('Wrong full-evaluation sample counts')
            if meta['selected_doc_ids'] != {t: list(range(spec[2])) for t, spec in TASKS.items()}:
                raise ValueError('Different selected document IDs')
            config = inputs['config']
            if config['pure_quantization'] or config['kl_chunks'] != 4 or config['context_size'] != 2048 or config['request_workers'] != 4:
                raise ValueError('Unexpected evaluation protocol')
            protocol = {k: inputs[k] for k in ['server_sha256', 'perplexity_sha256', 'text_sha256', 'pipeline_sha256']}
            protocols.add(json.dumps(protocol, sort_keys=True))
            metrics = json.loads((run / 'metrics.json').read_text())
            kl = json.loads((run / 'kl.json').read_text())
            samples = json.loads((run / 'samples.json').read_text())
            row = {'precision': precision, 'variant': variant, 'recipe': config['quantization_type'],
                   'size_mib': quant[variant]['size_bytes'] / 2**20,
                   'evaluation_seconds': completion['evaluation_seconds']}
            for task, (key, filt, count, label) in TASKS.items():
                metric_key = key + ',' + (filt if task == 'gsm8k' else 'none')
                row[task + '_score_pct'] = 100 * metrics[task][metric_key]
                row[task + '_stderr_pp'] = 100 * metrics[task][key + '_stderr,' + (filt if task == 'gsm8k' else 'none')]
                current = sample_scores(samples[task], task)
                if not np.isclose(np.mean([v[0] for v in current.values()]), metrics[task][metric_key]):
                    raise ValueError('Sample means do not reproduce aggregate metrics')
                if variant == 'control':
                    controls[task] = current
                else:
                    effects.append({'precision': precision, 'variant': variant, 'task': task,
                                    **paired_effect(controls[task], current, rng)})
            if variant == 'control':
                kl_control = kl['domains']
            for domain in ['code', 'math', 'chat']:
                item, control_item = kl['domains'][domain], kl_control[domain]
                if item['tokens'] != 1020 or control_item['tokens'] != 1020:
                    raise ValueError('Unexpected KL coverage')
                kl_rows.append({'precision': precision, 'variant': variant, 'domain': domain,
                    'kl_mean_nats': item['mean'], 'kl_uncertainty_nats': item['uncertainty'], 'tokens': item['tokens'],
                    'control_kl_nats': control_item['mean'],
                    'kl_reduction_pct': 100 * (1 - item['mean'] / control_item['mean'])})
            row['mean_kl_nats'] = np.mean([v['mean'] for v in kl['domains'].values()])
            row['model_sha256'] = quant[variant]['model_sha256']
            models.append(row)
            del samples
        print(f'Verified {precision}: four completed models, full samples and paired question/prompt hashes', flush=True)
        local_path = folder / 'local_errors' / p / 'local_errors.csv'
        local_manifest_path = local_path.with_name('local_errors_provenance.json')
        manifest = json.loads(local_manifest_path.read_text())
        if sha(local_path) != manifest['csv_sha256']:
            raise ValueError('Local reconstruction table changed: ' + p)
        if manifest['inputs']['model_sha256'] != {v: m['model_sha256'] for v, m in quant.items()}:
            raise ValueError('Local reconstruction and evaluated models differ')
        sources[str(local_path.relative_to(ROOT))] = sha(local_path)
        local = pd.read_csv(local_path)
        if len(local) != 1344 or local.duplicated(['variant', 'domain', 'tensor']).any():
            raise ValueError('Incomplete local tensor coverage')
        if not np.allclose(local.relative_error_pct, 100 * local.weighted_error / local.reference_energy):
            raise ValueError('Invalid relative reconstruction errors')
        for (variant, domain), group in local.groupby(['variant', 'domain']):
            local_rows.append({'precision': precision, 'variant': variant, 'domain': domain,
                'weighted_error': group.weighted_error.sum(), 'reference_energy': group.reference_energy.sum(),
                'control_error': group.control_error.sum(),
                'local_error_reduction_pct': 100 * (1 - group.weighted_error.sum() / group.control_error.sum())})
        for unit, destination in [('layer', block_rows), ('tensor_type', type_rows)]:
            for (variant, value), group in local.groupby(['variant', unit]):
                destination.append({'precision': precision, 'variant': variant, unit: value,
                    'local_error_reduction_pct': 100 * (1 - group.weighted_error.sum() / group.control_error.sum()),
                    'relative_error_pct': 100 * group.weighted_error.sum() / group.reference_energy.sum()})
    if len(protocols) != 1:
        raise ValueError('Recorded evaluation binaries, pipeline or held-out text differ between runs')
    effects = pd.DataFrame(effects)
    effects['holm_adjusted_p_54'] = holm(effects.mcnemar_exact_p)
    effects['passes_holm_0_05'] = effects.holm_adjusted_p_54 < .05
    tables = {'model_metrics': pd.DataFrame(models), 'benchmark_gains': effects,
              'kl_by_domain': pd.DataFrame(kl_rows), 'local_errors_by_domain': pd.DataFrame(local_rows),
              'block_local_recovery': pd.DataFrame(block_rows), 'tensor_type_local_recovery': pd.DataFrame(type_rows)}
    return tables, plan, sources, all_completions


def q8_payload_check(completions, gguf_python):
    if gguf_python.is_dir():
        sys.path.insert(0, str(gguf_python))
    import gguf
    results = {}
    for variant in ['control'] + VARIANTS:
        config = completions[('Q8', variant)]['inputs']['config']
        reader = gguf.GGUFReader(config['model_file'])
        h = hashlib.sha256()
        for tensor in sorted(reader.tensors, key=lambda t: t.name):
            h.update(json.dumps([tensor.name, tensor.tensor_type.name, list(map(int, tensor.shape))]).encode())
            h.update(tensor.data)
        results[variant] = h.hexdigest()
        print('Q8 stored-tensor payload checked: ' + variant, flush=True)
        del reader
    return {'payload_sha256': results, 'all_identical': len(set(results.values())) == 1,
            'definition': 'SHA256 over sorted tensor name, stored type, dimensions and raw tensor bytes; excludes GGUF metadata'}


def matrix(ax, grid, title, norm, label_format, cmap='RdBu_r'):
    im = ax.imshow(grid, aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid)), grid.index)
    ax.set_title(title, fontsize=11)
    for i, j in np.ndindex(grid.shape):
        value = grid.iloc[i, j]
        color = 'white' if abs(norm(value) - .5) > .32 else 'black'
        ax.text(j, i, format(value, label_format), ha='center', va='center', color=color, fontsize=10)
    return im


def make_figures(tables, plan, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    colors = {'control': '#222222', 'code': '#4477aa', 'math': '#ee7733', 'chat': '#228833'}
    models, gains, kl, local = [tables[n] for n in ['model_metrics', 'benchmark_gains', 'kl_by_domain', 'local_errors_by_domain']]
    baseline_columns = {'gsm8k': 'gsm8k_pct', 'ifeval': 'ifeval_pct', 'hellaswag': 'hellaswag_acc_norm_pct'}
    x = np.arange(6)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), layout='constrained')
    for ax, (task, spec) in zip(axes, TASKS.items()):
        for variant, color in colors.items():
            group = models[models.variant.eq(variant)].set_index('precision').loc[PRECISIONS]
            ax.plot(x, group[task + '_score_pct'], marker='o', color=color, label='No imatrix' if variant == 'control' else variant + ' imatrix')
        baseline = plan['bf16_reference']['scores_pct'][baseline_columns[task]]
        ax.axhline(baseline, color='gray', ls='--', label=f'BF16 ({baseline:.2f}%)')
        ax.set(title=spec[3], ylabel='Benchmark score (%)', xticks=x, xticklabels=PRECISIONS)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('1. Does imatrix calibration preserve benchmark quality?\nMatched standard recipes at each precision; full 1000 / 541 / 2000 example protocol', fontsize=14)
    fig.savefig(out / '01_benchmark_scores.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7), sharey=True, layout='constrained')
    for ax, (task, spec) in zip(axes, TASKS.items()):
        for j, variant in enumerate(VARIANTS):
            group = gains[gains.task.eq(task) & gains.variant.eq(variant)].set_index('precision').loc[PRECISIONS]
            offsets = x + (j - 1) * .24
            ax.bar(offsets, group.gain_pp, width=.23, color=colors[variant], label=variant)
            ax.errorbar(offsets, group.gain_pp,
                yerr=np.vstack([group.gain_pp - group.paired_ci95_low_pp, group.paired_ci95_high_pp - group.gain_pp]),
                fmt='none', color='#333333', linewidth=.8, capsize=2)
            for pos, row in zip(offsets, group.itertuples()):
                if row.passes_holm_0_05:
                    ax.text(pos, row.paired_ci95_high_pp + .25, '*', ha='center', fontsize=12)
        ax.axhline(0, color='black', lw=1)
        ax.axvspan(4.5, 5.5, color='gray', alpha=.08)
        ax.set(title=spec[3], xticks=x, xticklabels=PRECISIONS)
        ax.grid(axis='y', alpha=.2)
        ax.legend(title='Imatrix', fontsize=9)
    axes[0].set_ylabel('Score gain vs same-precision no-imatrix control (pp)')
    fig.suptitle('2. Accuracy recovery is task dependent\nBars: percentage-point gains; whiskers: paired 95% bootstrap intervals; * exact paired test with Holm correction over 54 comparisons', fontsize=12)
    fig.savefig(out / '02_benchmark_gains.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    candidate = kl[kl.variant.ne('control')]
    control_mean = models[models.variant.eq('control')].set_index('precision').mean_kl_nats
    mean = models[models.variant.ne('control')][['precision', 'variant', 'mean_kl_nats']].copy()
    mean['recovery'] = 100 * (1 - mean.mean_kl_nats / mean.precision.map(control_mean))
    limit = max(10, np.ceil(max(candidate.kl_reduction_pct.abs().max(), mean.recovery.abs().max()) / 10) * 10)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')
    grid = mean.pivot(index='precision', columns='variant', values='recovery').reindex(index=PRECISIONS, columns=VARIANTS)
    im = matrix(axes[0, 0], grid, 'Mean KL across code / math / chat', norm, '+.1f')
    for ax, domain in zip(list(axes.flat)[1:], ['code', 'math', 'chat']):
        grid = candidate[candidate.domain.eq(domain)].pivot(index='precision', columns='variant', values='kl_reduction_pct').reindex(index=PRECISIONS, columns=VARIANTS)
        im = matrix(ax, grid, domain.capitalize() + ' held-out text', norm, '+.1f')
    for ax in axes.flat:
        ax.set_xlabel('Calibration imatrix')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='KL reduction vs matched control (%) | red improves, blue worsens', shrink=.85)
    fig.suptitle('3. Where does imatrix calibration preserve output distributions?\nAll domains share one color scale; 1020 scored tokens per evaluation domain', fontsize=14)
    fig.savefig(out / '03_kl_recovery_by_dataset.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    pooled = local.groupby(['precision', 'variant'])[['weighted_error', 'control_error']].sum().reset_index()
    pooled['recovery'] = 100 * (1 - pooled.weighted_error / pooled.control_error)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout='constrained')
    left = pooled[pooled.variant.ne('control')].pivot(index='precision', columns='variant', values='recovery').reindex(index=PRECISIONS, columns=VARIANTS)
    right = mean.pivot(index='precision', columns='variant', values='recovery').reindex(index=PRECISIONS, columns=VARIANTS)
    bound = max(10, np.ceil(max(left.abs().max().max(), right.abs().max().max()) / 10) * 10)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    im = matrix(axes[0], left, 'Local reconstruction: calibration activations', norm, '+.1f')
    im = matrix(axes[1], right, 'Output KL: separate held-out text', norm, '+.1f')
    fig.colorbar(im, ax=axes.tolist(), label='Error reduction vs control (%) | red improves, blue worsens', shrink=.8)
    fig.suptitle('4. Local error recovery does not guarantee output recovery\nEqual-domain averages before ratios; local proxy omits cross-dimension correlations and downstream propagation', fontsize=13)
    fig.savefig(out / '04_local_vs_output_recovery.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    for key, unit, stem, title in [
        ('block_local_recovery', 'layer', '05_block_recovery', 'Blocks'),
        ('tensor_type_local_recovery', 'tensor_type', '06_tensor_type_recovery', 'Tensor types')]:
        frame = tables[key]
        frame = frame[frame.variant.ne('control')]
        bound = max(10, np.ceil(frame.local_error_reduction_pct.abs().max() / 10) * 10)
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        fig, axes = plt.subplots(1, 3, figsize=(17, 9 if unit == 'layer' else 6), sharey=True, layout='constrained')
        for ax, variant in zip(axes, VARIANTS):
            grid = frame[frame.variant.eq(variant)].pivot(index=unit, columns='precision', values='local_error_reduction_pct').reindex(columns=PRECISIONS).sort_index()
            im = matrix(ax, grid, variant.capitalize() + ' imatrix', norm, '+.1f')
            ax.set_xlabel('Target precision')
        axes[0].set_ylabel('Transformer block' if unit == 'layer' else 'Projection type')
        fig.colorbar(im, ax=axes.tolist(), label='Local weighted error reduction (%) | red improves, blue worsens', shrink=.8)
        fig.suptitle(f'{title}: where does imatrix quantization recover local computation?\nDomain errors averaged equally before pooling within each row; calibration proxy, not downstream sensitivity', fontsize=13)
        fig.savefig(out / (stem + '.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
    bound = max(1, np.ceil(gains.gain_pp.abs().max()))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    task_order = list(TASKS)
    task_labels = ['GSM8K\nmath', 'IFEval\ninstruction following', 'HellaSwag\ncommonsense']
    for ax, precision in zip(axes.flat, PRECISIONS):
        group = gains[gains.precision.eq(precision)]
        grid = group.pivot(index='variant', columns='task', values='gain_pp').reindex(index=VARIANTS, columns=task_order)
        im = matrix(ax, grid, precision + (' | identical tensor payloads' if precision == 'Q8' else ''), norm, '+.2f')
        ax.set_xticks(range(3), task_labels, fontsize=9)
        ax.set_yticks(range(3), ['Code imatrix', 'Math imatrix', 'Chat imatrix'])
        for row in group.itertuples():
            if row.passes_holm_0_05:
                i, j = VARIANTS.index(row.variant), task_order.index(row.task)
                ax.text(j + .32, i - .22, '*', fontsize=12, ha='center')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Benchmark gain (pp) | red improves, blue worsens', shrink=.85)
    fig.suptitle('7. Which benchmarks benefit from each calibration imatrix?\nMath-to-GSM8K is domain-related; no direct code/chat benchmark is present. * Holm-adjusted paired p < 0.05.', fontsize=13)
    fig.savefig(out / '07_calibration_to_task_improvement.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    domains = ['code', 'math', 'chat']
    values = kl[kl.variant.ne('control')]
    bound = max(10, np.ceil(values.kl_reduction_pct.abs().max() / 10) * 10)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    for ax, precision in zip(axes.flat, PRECISIONS):
        grid = values[values.precision.eq(precision)].pivot(index='variant', columns='domain', values='kl_reduction_pct').reindex(index=VARIANTS, columns=domains)
        im = matrix(ax, grid, precision, norm, '+.1f')
        ax.set_xticks(range(3), ['Code text', 'Math text', 'Chat text'])
        ax.set_yticks(range(3), ['Code imatrix', 'Math imatrix', 'Chat imatrix'])
        for i in range(3):
            ax.add_patch(Rectangle((i - .49, i - .49), .98, .98, fill=False, edgecolor='#222222', linewidth=2))
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Held-out KL reduction (%) | red improves, blue worsens', shrink=.85)
    fig.suptitle('8. Own-domain recovery versus transfer to other domains\nOutlined diagonal = calibration/evaluation domain match; all text is held out. This is output KL, not task accuracy.', fontsize=13)
    fig.savefig(out / '08_own_vs_other_domain_recovery.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/Imatrix_Recovery_Sweep_full_kl4')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/important plots/imatrix recovery')
    parser.add_argument('--publish-data', type=Path, default=ROOT / 'results/published/imatrix_recovery')
    parser.add_argument('--gguf-python', type=Path, default=Path(r'C:\llama.cpp\gguf-py'))
    parser.add_argument('--render-only', action='store_true', help='Rebuild plots from verified compact tables without original models or raw samples')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.publish_data.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        manifest_path = args.publish_data / 'experiment_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        tables = {}
        for name, info in manifest['tables'].items():
            path = args.publish_data / (name + '.csv')
            if sha(path) != info['sha256']:
                raise ValueError('Published table changed: ' + name)
            tables[name] = pd.read_csv(path)
            if len(tables[name]) != info['rows']:
                raise ValueError('Published row count changed: ' + name)
        derived = transfer_tables(tables)
        tables.update(derived)
        for name, frame in derived.items():
            frame.to_csv(args.publish_data / (name + '.csv'), index=False, lineterminator='\n')
            manifest['tables'][name] = {'rows': len(frame), 'sha256': sha(args.publish_data / (name + '.csv'))}
        make_figures(tables, {'bf16_reference': {'scores_pct': manifest['historical_bf16_scores_pct']}}, args.output)
        manifest['figures'] = {p.name: sha(p) for p in args.output.glob('*.png')}
        manifest['render_script_sha256'] = sha(Path(__file__))
        manifest['heatmap_colors'] = {'positive_improvement': 'red', 'zero': 'white', 'negative_worsening': 'blue'}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        print('Verified compact tables and rebuilt eight figures: ' + str(args.output))
        return
    tables, plan, sources, completions = audit_and_summarize(args.input)
    q8 = q8_payload_check(completions, args.gguf_python)
    tables.update(transfer_tables(tables))
    for name, frame in tables.items():
        frame.to_csv(args.publish_data / (name + '.csv'), index=False, lineterminator='\n')
    make_figures(tables, plan, args.output)
    significant = tables['benchmark_gains'].query('passes_holm_0_05')
    print('\nPaired example comparisons passing Holm correction:\n' + significant[['precision', 'variant', 'task', 'gain_pp', 'holm_adjusted_p_54']].to_string(index=False), flush=True)
    manifest = {'experiment': plan['experiment'], 'n_models': 24,
        'source_sha256': plan['source_sha256'], 'recipes': plan['recipes'], 'task_counts': {t: s[2] for t, s in TASKS.items()},
        'calibration_coverage': plan['calibration_coverage'], 'kl_tokens_per_domain': 1020,
        'historical_bf16_scores_pct': plan['bf16_reference']['scores_pct'],
        'q8_negative_control': q8, 'input_hashes': sources,
        'statistics': {'bootstrap': '10000 paired question resamples, compressed to multinomial wins/losses/ties; seed 473; pointwise 95% intervals',
                       'paired_test': 'Two-sided exact McNemar/binomial discordance test; Holm adjustment over all 54 task/precision/calibration comparisons',
                       'limits': 'Conditional on saved outputs and question-resampling assumptions; does not estimate model execution variability or calibration-repeat variability.'},
        'tables': {n: {'rows': len(f), 'sha256': sha(args.publish_data / (n + '.csv'))} for n, f in tables.items()},
        'figures': {p.name: sha(p) for p in args.output.glob('*.png')},
        'report_script_sha256': sha(Path(__file__)),
        'heatmap_colors': {'positive_improvement': 'red', 'zero': 'white', 'negative_worsening': 'blue'}}
    (args.publish_data / 'experiment_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Q8 stored tensor payloads identical:', q8['all_identical'], flush=True)
    print('Figures:', args.output, '\nCompact data:', args.publish_data, flush=True)


if __name__ == '__main__':
    main()
