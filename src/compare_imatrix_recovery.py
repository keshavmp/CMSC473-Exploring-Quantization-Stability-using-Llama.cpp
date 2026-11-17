"""Publish normalized Llama, Qwen and Gemma recovery from saved sweep outputs.

Run: python -m src.compare_imatrix_recovery
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .report_imatrix_recovery import PRECISIONS, VARIANTS, TASKS, sha

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = {
    'Llama': 'Imatrix_Recovery_Sweep_full_kl4',
    'Qwen': 'Qwen_Imatrix_Recovery_Sweep_full_kl4',
    'Gemma': 'Gemma_Imatrix_Recovery_Sweep_full_kl4',
}
COLORS = {'Llama': '#0072B2', 'Qwen': '#D55E00', 'Gemma': '#009E73'}
TYPES = ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down']
TYPE_LABELS = ['Attention Q', 'Attention K', 'Attention V', 'Attention output',
               'FFN gate', 'FFN up', 'FFN down']


def recovery(error, control):
    """Percent of matched control error removed; undefined ratios stay missing."""
    error, control = np.asarray(error, dtype=float), np.asarray(control, dtype=float)
    if not np.isfinite(error).all() or not np.isfinite(control).all() or (error < 0).any() or (control < 0).any():
        raise ValueError('Errors must be finite and nonnegative')
    return 100 * (1 - np.divide(error, control, out=np.full_like(error, np.nan), where=control > 0))


def pool_local(frame, units):
    """Equal weight per domain and calibration, then ratio of summed errors.

    Average errors before taking ratios: tiny-error tensors cannot dominate.
    """
    keys = ['model', 'precision'] + units
    result = frame.groupby(keys, as_index=False)[['weighted_error', 'control_error']].sum()
    result['error_recovered'] = result.control_error - result.weighted_error
    result['recovery_pct'] = recovery(result.weighted_error, result.control_error)
    return result


def load_sweeps(results):
    local_frames, output_rows, benchmark_rows = [], [], []
    sources, models, protocols = {}, {}, {}

    def read_json(path):
        sources[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    for model, name in SWEEPS.items():
        folder = results / name
        plan = read_json(folder / 'sweep_plan.json')
        status = read_json(folder / 'sweep_status.json')
        if set(status) != {p.lower() for p in PRECISIONS} or any(s['status'] != 'ok' for s in status.values()):
            raise ValueError('Incomplete sweep: ' + model)
        models[model] = {'source_model': plan['source_model'], 'source_sha256': plan['source_sha256'],
                         'recipes': plan['recipes'], 'calibration_coverage': plan['calibration_coverage']}
        block_count = None
        for precision in PRECISIONS:
            quant = read_json(folder / f'quantization_manifest_{precision.lower()}.json')
            kl_runs, scores = {}, {}
            for variant in ['control'] + VARIANTS:
                run = folder / f'mixed-{precision.lower()}-{variant}'
                completion = read_json(run / 'comparison_complete.json')
                inputs = completion['inputs']
                config = inputs['config']
                if (config['pure_quantization'] or config['kl_chunks'] != 4
                        or config['context_size'] != 2048 or config['request_workers'] != 4
                        or config['quantization_type'] != plan['recipes'][precision.lower()]):
                    raise ValueError('Unmatched quantization/evaluation configuration')
                if inputs['model_sha256'] != quant[variant]['model_sha256'] or quant[variant]['inputs']['source_sha256'] != plan['source_sha256']:
                    raise ValueError('Model/source provenance mismatch')
                if quant[variant]['tensor_layout'] != quant['control']['tensor_layout']:
                    raise ValueError('Unmatched tensor layouts')
                protocol = {k: inputs[k] for k in ['server_sha256', 'perplexity_sha256', 'text_sha256', 'pipeline_sha256']}
                protocols.setdefault(model, set()).add(json.dumps(protocol, sort_keys=True))
                # Verify the actual aggregate files used here against completion records.
                for filename in ['metrics.json', 'kl.json', 'metadata.json']:
                    path = run / filename
                    if sha(path) != completion['outputs'][filename]:
                        raise ValueError('Completed output changed: ' + str(path))
                meta = read_json(run / 'metadata.json')
                if meta['actual_unique_samples'] != {t: spec[2] for t, spec in TASKS.items()}:
                    raise ValueError('Mismatched benchmark coverage')
                if meta['selected_doc_ids'] != {t: list(range(spec[2])) for t, spec in TASKS.items()}:
                    raise ValueError('Mismatched benchmark document IDs')
                kl_runs[variant] = read_json(run / 'kl.json')
                if kl_runs[variant]['units'] != 'nats' or kl_runs[variant]['direction'] != 'BF16 || evaluated model':
                    raise ValueError('Unmatched KL units/direction')
                metrics = read_json(run / 'metrics.json')
                scores[variant] = {t: 100 * metrics[t][spec[0] + ',' + spec[1]] for t, spec in TASKS.items()}
            for variant in VARIANTS:
                for domain in VARIANTS:
                    item, base = kl_runs[variant]['domains'][domain], kl_runs['control']['domains'][domain]
                    if item['tokens'] != 1020 or base['tokens'] != 1020 or kl_runs[variant]['baseline'] != kl_runs['control']['baseline']:
                        raise ValueError('Unmatched KL references/coverage')
                    output_rows.append(dict(model=model, precision=precision, variant=variant, domain=domain,
                                            error=item['mean'], control_error=base['mean']))
                for task in TASKS:
                    score, base = scores[variant][task], scores['control'][task]
                    benchmark_rows.append(dict(model=model, precision=precision, variant=variant, task=task,
                                               score_pct=score, control_score_pct=base, gain_pp=score-base,
                                               gain_relative_pct=100*(score-base)/base if base > 0 else np.nan))
            path = folder / 'local_errors' / precision.lower() / 'local_errors.csv'
            provenance = read_json(path.with_name('local_errors_provenance.json'))
            if sha(path) != provenance['csv_sha256']:
                raise ValueError('Local error table changed: ' + str(path))
            sources[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = sha(path)
            if provenance['inputs']['model_sha256'] != {v: q['model_sha256'] for v, q in quant.items()}:
                raise ValueError('Local errors refer to different evaluated models')
            local = pd.read_csv(path)
            blocks = sorted(local.layer.unique())
            if blocks != list(range(len(blocks))) or (block_count is not None and block_count != len(blocks)):
                raise ValueError('Incomplete/inconsistent block indices')
            block_count = len(blocks)
            expected = pd.MultiIndex.from_product([['control'] + VARIANTS, VARIANTS, blocks, TYPES])
            actual = pd.MultiIndex.from_frame(local[['variant', 'domain', 'layer', 'tensor_type']])
            if actual.has_duplicates or len(actual) != len(expected) or len(expected.difference(actual)):
                raise ValueError('Incomplete local projection/domain/calibration coverage')
            controls = local[local.variant.eq('control')].set_index(['domain', 'tensor']).weighted_error
            matched = pd.MultiIndex.from_frame(local[['domain', 'tensor']]).map(controls)
            if not np.allclose(local.control_error, matched) or not np.allclose(local.relative_error_pct, 100*local.weighted_error/local.reference_energy):
                raise ValueError('Invalid local control or reference normalization')
            local = local[local.variant.ne('control')].copy()
            local['model'], local['precision'] = model, precision
            local['depth_pct'] = 100 * local.layer / max(1, block_count - 1)
            local_frames.append(local)
        models[model]['blocks'] = block_count
        print(f'Verified {model}: 24 runs, {block_count} blocks, six local-error tables', flush=True)
    if any(len(p) != 1 for p in protocols.values()):
        raise ValueError('Evaluation protocol changes within a model sweep')
    if len(set.union(*protocols.values())) != 1:
        raise ValueError('Evaluation binaries, pipeline or text differ between model sweeps')
    if any(info['recipes'] != models['Llama']['recipes'] for info in models.values()):
        raise ValueError('Quantization recipes differ between model sweeps')
    local = pd.concat(local_frames, ignore_index=True)
    local['error_recovered'] = local.control_error - local.weighted_error
    local['recovery_pct'] = recovery(local.weighted_error, local.control_error)
    output = pd.DataFrame(output_rows)
    output['error_recovered'] = output.control_error - output.error
    output['recovery_pct'] = recovery(output.error, output.control_error)
    tables = {'local_tensor_detail': local, 'output_by_domain': output,
              'benchmark_by_task': pd.DataFrame(benchmark_rows)}
    for name, units in [('precision', []), ('block', ['layer', 'depth_pct']), ('layer_type', ['tensor_type'])]:
        tables[name + '_recovery'] = pool_local(local, units)
        tables[name + '_by_calibration'] = pool_local(local, ['variant'] + units)
    pooled = output.groupby(['model', 'precision'], as_index=False)[['error', 'control_error']].sum()
    pooled['error_recovered'] = pooled.control_error - pooled.error
    pooled['recovery_pct'] = recovery(pooled.error, pooled.control_error)
    tables['output_recovery'] = pooled
    return tables, {'models': models, 'input_hashes': sources,
                    'protocols': {m: [json.loads(p) for p in ps] for m, ps in protocols.items()}}


def make_benchmark_bars(benchmarks, out):
    """Plot every calibration separately on a common percentage-point scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {'code': '#4477aa', 'math': '#ee7733', 'chat': '#228833'}
    x = np.arange(len(PRECISIONS))
    lo, hi = min(0, benchmarks.gain_pp.min()), max(0, benchmarks.gain_pp.max())
    padding = max(.5, (hi-lo)*.1)

    def panel(ax, model, task):
        selected = benchmarks[benchmarks.model.eq(model) & benchmarks.task.eq(task)]
        for j, variant in enumerate(VARIANTS):
            group = selected[selected.variant.eq(variant)].set_index('precision').loc[PRECISIONS]
            ax.bar(x + (j-1)*.24, group.gain_pp, width=.23, color=colors[variant])
        ax.axhline(0, color='#222222', linewidth=1.2)
        ax.set(xticks=x, xticklabels=PRECISIONS, ylim=(lo-padding, hi+padding))
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)

    def finish(fig, axes, models, title, stem):
        for row, model in zip(axes, models):
            for ax, (task, spec) in zip(row, TASKS.items()):
                panel(ax, model, task)
            row[0].set_ylabel((model + '\n' if len(models) > 1 else '') + 'Score gain vs control (pp)')
        for ax, spec in zip(axes[0], TASKS.values()):
            ax.set_title(spec[3], fontsize=12)
        for ax in axes[-1]:
            ax.set_xlabel('Target precision')
        fig.suptitle(title + '\nPercentage-point change vs same-model, same-precision no-imatrix control', fontsize=14)
        fig.legend(handles=[Patch(facecolor=colors[v], label=v.capitalize() + ' imatrix') for v in VARIANTS],
                   loc='outside lower center', ncol=3, frameon=False)
        fig.savefig(out / (stem + '.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)

    for index, model in enumerate(SWEEPS, start=6):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), sharey=True, layout='constrained', squeeze=False)
        finish(fig, axes, [model], model + ': accuracy recovery by task',
               f'{index:02d}_{model.lower()}_benchmark_gains')
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharey=True, layout='constrained')
    finish(fig, axes, list(SWEEPS), 'Accuracy recovery: Llama, Qwen and Gemma', '09_all_models_benchmark_gains')


def make_figures(tables, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    make_benchmark_bars(tables['benchmark_by_task'], out)

    def save(fig, stem, directory=out):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f'{stem}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)

    def style(ax):
        ax.axhline(0, color='#555555', linewidth=.9)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)

    def precision_lines(ax, frame, column='recovery_pct'):
        for model, color in COLORS.items():
            values = frame[frame.model.eq(model)].set_index('precision').reindex(PRECISIONS)
            ax.plot(PRECISIONS, values[column], 'o-', color=color, label=model, linewidth=2, markersize=5)
        style(ax)
        ax.set_xlabel('Target precision')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained', sharey=True)
    for ax, key, title in zip(axes, ['precision_recovery', 'output_recovery'],
                            ['Local weight reconstruction', 'Held-out output distribution (KL)']):
        precision_lines(ax, tables[key])
        ax.set_title(title)
    axes[0].set_ylabel('Control error removed (%)')
    axes[0].legend()
    fig.suptitle('Imatrix recovery by precision\n0% = unchanged; higher = better | code, math and chat calibrations pooled', fontsize=14)
    save(fig, '01_precision_comparison')

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained', sharex=True, sharey=True)
    blocks = tables['block_recovery']
    for ax, precision in zip(axes.flat, PRECISIONS):
        for model, color in COLORS.items():
            g = blocks[blocks.model.eq(model) & blocks.precision.eq(precision)].sort_values('layer')
            ax.plot(g.depth_pct, g.recovery_pct, '.-', color=color, label=model, linewidth=1.6, markersize=4)
        style(ax)
        ax.set(title=precision, xticks=[0, 50, 100])
    for ax in axes[-1]:
        ax.set_xlabel('Position in model (%, first to last block)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Local control error removed (%)')
    axes[0, 0].legend()
    fig.suptitle('Recovery across transformer blocks\nNative block positions normalized to 0–100% depth; no smoothing or interpolation', fontsize=14)
    save(fig, '02_block_comparison')

    def heatmaps(frame, row, columns, panels, panel_column, row_order, row_labels, stem, title, directory=out):
        finite = frame.recovery_pct.dropna()
        bound = max(1, float(np.ceil(finite.abs().max())))
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        rows = 2 if len(panels) == 6 else 1
        fig, axes = plt.subplots(rows, 3, figsize=(14, 8 if rows == 2 else max(5, len(row_order)*.28)),
                                 layout='constrained', squeeze=False)
        cmap = plt.get_cmap('BrBG').copy()
        cmap.set_bad('#dddddd')
        for ax, panel in zip(axes.flat, panels):
            grid = frame[frame[panel_column].eq(panel)].pivot(index=row, columns=columns, values='recovery_pct').reindex(index=row_order)
            grid = grid.reindex(columns=list(SWEEPS) if columns == 'model' else PRECISIONS)
            im = ax.imshow(grid, cmap=cmap, norm=norm, aspect='auto')
            ax.set(title=panel, xticks=range(len(grid.columns)), xticklabels=grid.columns,
                   yticks=range(len(grid)), yticklabels=row_labels)
            for i, j in np.ndindex(grid.shape):
                v = grid.iloc[i, j]
                ax.text(j, i, 'N/A' if not np.isfinite(v) else f'{v:+.1f}', ha='center', va='center',
                        color='white' if np.isfinite(v) and abs(v) > .65*bound else '#222222', fontsize=9)
        fig.colorbar(im, ax=list(axes.flat), label='Local control error removed (%) | green improves, brown worsens', shrink=.8)
        fig.suptitle(title, fontsize=14)
        save(fig, stem, directory)

    heatmaps(tables['layer_type_recovery'], 'tensor_type', 'model', PRECISIONS, 'precision', TYPES, TYPE_LABELS,
             '03_layer_type_comparison', 'Recovery by layer / projection type\nPooled across transformer blocks, evaluation domains and calibration choices')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained', sharey=True)
    benchmarks = tables['benchmark_by_task'].groupby(['model', 'precision', 'task'], as_index=False).gain_relative_pct.mean()
    for ax, (task, spec) in zip(axes, TASKS.items()):
        precision_lines(ax, benchmarks[benchmarks.task.eq(task)], 'gain_relative_pct')
        ax.set_title(spec[3].replace(' (length normalized)', ''))
    axes[0].set_ylabel('Score increase vs control (%)')
    axes[0].legend()
    fig.suptitle('Does local recovery translate to task quality?\nMean relative gain over code/math/chat calibrations; descriptive results, no significance claims', fontsize=14)
    save(fig, '04_benchmark_comparison')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained', sharey=True)
    for ax, variant in zip(axes, VARIANTS):
        precision_lines(ax, tables['precision_by_calibration'].query('variant == @variant'))
        ax.set_title(variant.capitalize() + ' imatrix')
    axes[0].set_ylabel('Local control error removed (%)')
    axes[0].legend()
    fig.suptitle('Calibration choices shown separately\nEach calibration evaluated over the same three local activation domains', fontsize=14)
    save(fig, '05_calibration_comparison')

    for model in SWEEPS:
        frame = tables['block_by_calibration'].query('model == @model')
        indices = sorted(frame.layer.unique())
        heatmaps(frame, 'layer', 'precision', VARIANTS, 'variant', indices, indices,
                 model.lower() + '_block_recovery', model + ': every block, precision and calibration', out / 'by model')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/imatrix recovery')
    parser.add_argument('--table-output', type=Path, default=ROOT / 'results/imatrix_recovery_tables',
                        help='Destination for the compact recovery CSVs and manifest')
    parser.add_argument('--tables-only', action='store_true', help='Export verified CSVs without redrawing figures')
    args = parser.parse_args()
    tables, manifest = load_sweeps(args.results.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    data = args.table_output
    data.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(data / (name + '.csv'), index=False, lineterminator='\n')
    if not args.tables_only:
        make_figures(tables, args.output)
    manifest['normalization'] = {
        'local_and_kl': '100 * (1 - sum(calibrated error) / sum(matched no-imatrix control error))',
        'error_recovered': 'control_error - weighted_error (local), or control_error - error (KL); pooled rows use sums, not means.',
        'pooling': 'Equal representation of three domains and three calibrations; sum errors before ratio. Within a domain, tensors are weighted by control error.',
        'benchmark': '100 * (calibrated score - control score) / control score, then arithmetic mean across calibrations; not percent of source-score gap recovered.',
        'benchmark_grouped_bars_06_to_09': 'Calibrated score (%) minus matched control score (%), in percentage points; all models/calibrations share one y-axis scale. Plain bars with no intervals or significance markers.',
        'depth': '100 * block_index / (number_of_blocks - 1); native positions, no interpolation',
        'undefined': 'Zero control denominators yield missing values, never clipped; negative recovery is retained.'}
    manifest['limitations'] = [
        'Local diagonal activation-weighted reconstruction is a proxy, not a measured block intervention or downstream sensitivity.',
        'Llama/Gemma use BF16 sources; Qwen uses F16. Model sizes, architectures and tokenizers differ. KL ratios compare within-model improvement, not absolute model quality.',
        'KL uses 1020 scored tokens/domain, with different tokenizers; normalized results do not establish statistical significance.',
        'Graphs summarize three fixed calibration choices, not independent repeated trials; benchmark panels have no inferential intervals.',
        'Q8 is a negative control and can ignore imatrix. Small output/benchmark changes at unchanged weights can reflect execution variation.',
        'This report verifies saved aggregate hashes, local CSV provenance, layouts and selected IDs; it does not rerun inference, revalidate individual sample scores or reopen model payloads.']
    manifest['tables'] = {n: {'rows': len(f), 'sha256': sha(data / (n+'.csv'))} for n, f in tables.items()}
    manifest['figures'] = {str(p.relative_to(args.output)): sha(p) for p in args.output.rglob('*.png') if 'llama legacy' not in p.parts}
    manifest['script_sha256'] = sha(Path(__file__))
    (data / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Saved normalized comparisons to ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
