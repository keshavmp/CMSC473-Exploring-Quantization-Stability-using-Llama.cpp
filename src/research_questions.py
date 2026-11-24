"""Rebuild the three research questions from saved evidence; never runs inference.

Run: python -m src.research_questions
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.report_imatrix_recovery import TASKS, paired_effect, holm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research_questions'
PRECISIONS = ['Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8']
DOMAINS = ['code', 'math', 'chat']
LABELS = {'gsm8k': 'Math reasoning (GSM8K)', 'ifeval': 'Instruction following (IFEval)',
          'hellaswag': 'Commonsense completion (HellaSwag)'}
COLORS = dict(code='#3378a8', math='#d47b27', chat='#498a65')
SWEEPS = {'Llama': 'Imatrix_Recovery_Sweep_full_kl4',
          'Qwen': 'Qwen_Imatrix_Recovery_Sweep_full_kl4',
          'Gemma': 'Gemma_Imatrix_Recovery_Sweep_full_kl4'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Evidence:
    def __init__(self):
        self.hashes = {}

    def track(self, path):
        path = Path(path)
        self.hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    def json(self, path):
        return json.loads(self.track(path).read_text(encoding='utf-8'))

    def csv(self, path):
        return pd.read_csv(self.track(path))


def scores(records, task, expected=None):
    """One binary score per prompt, with all pairing identities preserved."""
    key, filt, _, _ = TASKS[task]
    result = {}
    for sample in records:
        if sample['filter'] != filt:
            continue
        doc = int(sample['doc_id'])
        value = float(sample[key])
        if doc in result or value not in (0, 1):
            raise ValueError(f'Invalid/duplicate binary sample: {task}/{doc}')
        result[doc] = (value, sample['doc_hash'], sample['prompt_hash'], sample['target_hash'])
    if not result or (expected is not None and set(result) != set(range(expected))):
        raise ValueError(f'Unexpected sample coverage: {task}')
    return result


def load_run(folder, evidence, completed=False):
    metrics = evidence.json(folder / 'metrics.json')
    metadata = evidence.json(folder / 'metadata.json')
    samples = evidence.json(folder / 'samples.json')
    if completed:
        marker = evidence.json(folder / 'comparison_complete.json')
        for name in ['metrics.json', 'metadata.json', 'samples.json']:
            if marker['outputs'][name] != evidence.hashes[(folder / name).relative_to(ROOT).as_posix()]:
                raise ValueError(f'Completion hash mismatch: {folder}/{name}')
        if metadata['model_sha256'] != marker['inputs']['model_sha256']:
            raise ValueError(f'Model identity mismatch: {folder}')
    selected = {}
    for task, (key, filt, expected, _) in TASKS.items():
        current = scores(samples[task], task, expected)
        if not np.isclose(np.mean([v[0] for v in current.values()]), metrics[task][key + ',' + filt]):
            raise ValueError(f'Sample/aggregate mismatch: {folder}/{task}')
        if metadata['actual_unique_samples'][task] != expected:
            raise ValueError(f'Metadata count mismatch: {folder}/{task}')
        selected[task] = current
    return selected


def effect_row(before, after, rng):
    return {**paired_effect(before, after, rng),
            'control_score_pct': 100 * np.mean([v[0] for v in before.values()]),
            'score_pct': 100 * np.mean([v[0] for v in after.values()])}


def precision_tables(evidence, rng):
    folder = ROOT / 'results/Pure_Sweep'
    baseline = load_run(folder / 'BF16-pure-sweep', evidence)
    rows, kl = [], []
    for precision in PRECISIONS:
        run = folder / f'pure-{precision.lower()}-sweep'
        candidate = load_run(run, evidence)
        for task in TASKS:
            effect = effect_row(baseline[task], candidate[task], rng)
            rows.append({'precision': precision, 'task': task, **effect,
                         'loss_pp': -effect['gain_pp'],
                         'loss_ci95_low_pp': -effect['paired_ci95_high_pp'],
                         'loss_ci95_high_pp': -effect['paired_ci95_low_pp']})
        for domain, item in evidence.json(run / 'kl.json')['domains'].items():
            kl.append({'precision': precision, 'domain': domain, 'mean_kl_nats': item['mean'],
                       'scored_tokens': item['tokens'], 'tool_uncertainty_nats': item['uncertainty']})
    frame = pd.DataFrame(rows)
    frame['holm_p_18'] = holm(frame.mcnemar_exact_p)
    return frame, pd.DataFrame(kl)


def recovery_tables(evidence, rng):
    rows = []
    for model, name in SWEEPS.items():
        folder = ROOT / 'results' / name
        for precision in PRECISIONS:
            quant = evidence.json(folder / f'quantization_manifest_{precision.lower()}.json')
            control = load_run(folder / f'mixed-{precision.lower()}-control', evidence, True)
            control_marker = evidence.json(folder / f'mixed-{precision.lower()}-control/comparison_complete.json')
            for variant in DOMAINS:
                if quant[variant]['tensor_layout'] != quant['control']['tensor_layout']:
                    raise ValueError('Stored precision layouts differ')
                if quant[variant]['inputs']['source_sha256'] != quant['control']['inputs']['source_sha256']:
                    raise ValueError('Source checkpoints differ')
                candidate = load_run(folder / f'mixed-{precision.lower()}-{variant}', evidence, True)
                marker = evidence.json(folder / f'mixed-{precision.lower()}-{variant}/comparison_complete.json')
                for key in ['server_sha256', 'perplexity_sha256', 'pipeline_sha256', 'text_sha256']:
                    if marker['inputs'][key] != control_marker['inputs'][key]:
                        raise ValueError(f'Unmatched execution protocol: {model}/{precision}/{variant}/{key}')
                if marker['inputs']['model_sha256'] != quant[variant]['model_sha256']:
                    raise ValueError('Quantization/evaluation model identity mismatch')
                ignored = {'name', 'model_file'}
                before = {k:v for k,v in control_marker['inputs']['config'].items() if k not in ignored}
                after = {k:v for k,v in marker['inputs']['config'].items() if k not in ignored}
                if before != after:
                    raise ValueError(f'Unmatched evaluation settings: {model}/{precision}/{variant}')
                for task in TASKS:
                    rows.append({'model': model, 'precision': precision, 'variant': variant,
                                 'task': task, **effect_row(control[task], candidate[task], rng)})
            print(f'Paired audit: {model} {precision}', flush=True)
    frame = pd.DataFrame(rows)
    frame['holm_p_162'] = holm(frame.mcnemar_exact_p)
    frame['holm_p_54_within_model'] = frame.groupby('model').mcnemar_exact_p.transform(lambda p: holm(p))
    frame['passes_global_holm_005'] = frame.holm_p_162 < .05
    return frame


def block_tables(evidence):
    folder = ROOT / 'results/final_analysis/questions'
    prediction = evidence.csv(folder / 'prediction_blocks.csv')
    blocks = evidence.csv(folder / 'preservation_blocks.csv')
    keys = ['precision', 'variant', 'domain', 'block']
    if blocks.duplicated(keys).any() or len(blocks) != 3 * 3 * 3 * 16:
        raise ValueError('Incomplete/duplicate block coverage')
    if not np.allclose(blocks.error_removed, blocks.control_error - blocks.weighted_error):
        raise ValueError('Invalid reconstruction recovery')
    if not np.allclose(blocks.recovery_pct, 100 * blocks.error_removed / blocks.control_error):
        raise ValueError('Invalid reconstruction recovery percentages')
    # Independently choose sensitive blocks from interventions, never recovery.
    groups, correlations, profiles = [], [], []
    for (precision, domain), p in prediction.groupby(['precision', 'domain']):
        sensitive = set(p.sort_values(['sensitivity_nats', 'block'], ascending=[False, True]).head(3).block)
        b = blocks[blocks.precision.eq(precision) & blocks.domain.eq(domain)]
        for variant, g in b.groupby('variant'):
            pooled = 100 * g.error_removed.sum() / g.control_error.sum()
            blocks.loc[g.index, 'recovery_above_whole_model_pp'] = g.recovery_pct - pooled
            correlations.append({'precision': precision, 'domain': domain, 'variant': variant,
                                 'spearman_sensitivity_vs_fraction_recovered': spearmanr(g.sensitivity_nats, g.recovery_pct).statistic,
                                 'spearman_sensitivity_vs_absolute_removed': spearmanr(g.sensitivity_nats, g.error_removed).statistic})
            for selected, label in [(True, 'Most sensitive 3'), (False, 'Other 13')]:
                subset = g[g.block.isin(sensitive).eq(selected)]
                groups.append({'precision': precision, 'domain': domain, 'variant': variant,
                               'group': label, 'n_blocks': len(subset), 'control_error': subset.control_error.sum(),
                               'error_removed': subset.error_removed.sum(),
                               'recovery_pct': 100 * subset.error_removed.sum() / subset.control_error.sum()})
        for a, c in combinations(DOMAINS, 2):
            x, y = [b[b.variant.eq(v)].set_index('block').sort_index().recovery_pct for v in [a, c]]
            profiles.append({'precision': precision, 'measurement_domain': domain,
                             'calibration_a': a, 'calibration_b': c,
                             'rank_agreement': spearmanr(x, y).statistic,
                             'mean_absolute_centered_profile_difference_pp': np.abs((x-x.mean())-(y-y.mean())).mean()})
    return prediction, blocks, pd.DataFrame(groups), pd.DataFrame(correlations), pd.DataFrame(profiles)


def save(fig, path, footer):
    fig.text(.5, .015, footer, ha='center', va='bottom', fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_precision(frame, kl, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, task in zip(axes, TASKS):
        g = frame[frame.task.eq(task)].set_index('precision').loc[PRECISIONS]
        ax.errorbar(range(6), g.loss_pp, yerr=[g.loss_pp-g.loss_ci95_low_pp, g.loss_ci95_high_pp-g.loss_pp],
                    fmt='o-', color='#3378a8', capsize=4)
        ax.axhline(0, color='#555555', lw=1)
        ax.set(xticks=range(6), xticklabels=PRECISIONS, title=LABELS[task], xlabel='Weight precision')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Accuracy lost versus BF16 (percentage points)')
    fig.suptitle('1. How much quality does lower precision lose?', fontsize=16, fontweight='bold')
    save(fig, out/'01_precision_task_loss.png', 'Llama 1B pure quantization | Lower is better; negative means an observed gain.\nBars: paired question-bootstrap 95% intervals; they exclude execution variability.')
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for domain in DOMAINS:
        g = kl[kl.domain.eq(domain)].set_index('precision').loc[PRECISIONS]
        ax.plot(range(6), g.mean_kl_nats, 'o-', label=domain.title(), color=COLORS[domain])
    ax.set(yscale='log', xticks=range(6), xticklabels=PRECISIONS, ylabel='Output KL from BF16 (nats; logarithmic scale)', xlabel='Weight precision')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.2)
    fig.suptitle('1. Which text domains change most?', fontsize=16, fontweight='bold')
    save(fig, out/'02_precision_domain_kl.png', 'Lower is closer to BF16 | Historical fixed-text results: only 510 scored tokens per domain.\nText domains are distinct from the three benchmark tasks; no population uncertainty is implied.')


def plot_recovery(frame, out):
    for model in SWEEPS:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
        for ax, task in zip(axes, TASKS):
            for offset, variant in zip([-.2, 0, .2], DOMAINS):
                g = frame[frame.model.eq(model) & frame.task.eq(task) & frame.variant.eq(variant)].set_index('precision').loc[PRECISIONS]
                ax.errorbar(np.arange(6)+offset, g.gain_pp,
                            yerr=[g.gain_pp-g.paired_ci95_low_pp, g.paired_ci95_high_pp-g.gain_pp],
                            fmt='o', color=COLORS[variant], capsize=3, label=variant.title())
                for x, r in zip(np.arange(6)+offset, g.itertuples()):
                    if r.passes_global_holm_005:
                        ax.annotate('*', (x, r.paired_ci95_high_pp), xytext=(0, 5), textcoords='offset points', ha='center')
            ax.axhline(0, color='#555555', lw=1)
            ax.set(xticks=range(6), xticklabels=PRECISIONS, title=LABELS[task], xlabel='Weight precision')
            ax.grid(axis='y', alpha=.2)
        axes[0].set_ylabel('Accuracy gained over no-imatrix control (pp)')
        axes[-1].legend(title='Calibration data', frameon=False)
        fig.suptitle(f'2. Does imatrix recover task quality? — {model}', fontsize=16, fontweight='bold')
        save(fig, out/f'03_imatrix_task_gains_{model.lower()}.png', 'Higher is better | Same-precision, same-recipe controls; pure-sweep results are not controls.\nBars: pointwise paired 95% intervals. * passes Holm correction across all 162 comparisons. Calibration budgets differ.')


def plot_blocks(prediction, blocks, groups, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.7))
    positive = prediction.sensitivity_nats.clip(lower=1e-7)
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=positive.min(), vmax=positive.max())
    for ax, domain in zip(axes, DOMAINS):
        grid = prediction[prediction.domain.eq(domain)].pivot(index='block', columns='precision', values='sensitivity_nats').reindex(columns=['Q2','Q4','Q6'])
        im = ax.imshow(grid.clip(lower=1e-7), aspect='auto', cmap='YlOrRd', norm=norm)
        ax.set(xticks=range(3), xticklabels=grid.columns, yticks=range(16), ylabel='Block', title=domain.title(), xlabel='Lowered block precision')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Additional output KL versus all-Q8 (nats; log scale)', fraction=.025, pad=.03)
    fig.suptitle('1. Which blocks are most sensitive?', fontsize=16, fontweight='bold')
    fig.subplots_adjust(bottom=.16, top=.87, right=.85, wspace=.35)
    fig.text(.5,.035,'Llama: one lowered block on a Q8 background. Darker means more damage.\nShared scale; no repeat-based uncertainty. Checkpoint/text/build provenance across experiments is incomplete.',ha='center',fontsize=9)
    fig.savefig(out/'04_block_sensitivity.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    for precision in ['Q2', 'Q4', 'Q6']:
        for column, tag, title in [('recovery_pct','recovery','Original local error removed (%)'),
                                   ('recovery_above_whole_model_pp','concentration','Recovery above whole-model recovery (pp)')]:
            subset = blocks[blocks.precision.eq(precision)]
            limit = max(1, subset[column].abs().max())
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
            fig, axes = plt.subplots(1, 3, figsize=(12, 7))
            for ax, domain in zip(axes, DOMAINS):
                grid = subset[subset.domain.eq(domain)].pivot(index='block',columns='variant',values=column).reindex(columns=DOMAINS)
                im = ax.imshow(grid, aspect='auto', cmap='RdBu', norm=norm)
                for i in range(16):
                    for j in range(3):
                        value = grid.iloc[i,j]
                        ax.text(j,i,f'{value:.0f}',ha='center',va='center',fontsize=8,
                                color='white' if abs(value)>limit*.6 else '#222222')
                ax.set(xticks=range(3),xticklabels=[d.title() for d in DOMAINS],yticks=range(16),
                       title=f'Measured on {domain}',xlabel='Imatrix calibration data',ylabel='Block')
            fig.subplots_adjust(bottom=.17,top=.87,right=.86,wspace=.35)
            fig.colorbar(im,ax=axes.ravel().tolist(),label=title,fraction=.025,pad=.03)
            fig.suptitle(f'3. Which blocks benefit from imatrix? — {precision}',fontsize=16,fontweight='bold')
            foot = ('Blue: improvement; red: worsening.' if tag=='recovery' else 'Blue: more proportional recovery than the whole model; red: less. Zero: whole-model recovery.')
            fig.text(.5,.035,foot+'\nLlama | Calibration budgets differ; these are exploratory calibration-local reconstruction results.',ha='center',fontsize=9)
            fig.savefig(out/f'05_blocks_{precision.lower()}_{tag}.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig, axes=plt.subplots(1,3,figsize=(12,4.8),sharey=True)
    for ax,domain in zip(axes,DOMAINS):
        for offset,group,color in [(-.18,'Most sensitive 3','#d47b27'),(.18,'Other 13','#3378a8')]:
            g=groups[groups.domain.eq(domain)&groups.group.eq(group)].groupby('precision')[['control_error','error_removed']].sum().loc[['Q2','Q4','Q6']]
            values=100*g.error_removed/g.control_error
            bars=ax.bar(np.arange(3)+offset,values,width=.34,label=group,color=color)
            ax.bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
        ax.set(xticks=range(3),xticklabels=['Q2','Q4','Q6'],title=f'Measured on {domain}')
        ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    axes[0].set_ylabel('Original local error removed (%)');axes[-1].legend(frameon=False)
    fig.suptitle('3. Do the most sensitive blocks benefit more?',fontsize=16,fontweight='bold')
    save(fig,out/'06_sensitive_vs_other_blocks.png','Each bar pools errors across three calibrations before division. No independent-repeat error bars are available.\nSensitivity: pure-block interventions; recovery: mixed recipes. This tests an association, not a block-level cause.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args(); out=args.output;out.mkdir(parents=True,exist_ok=True)
    evidence=Evidence();rng=np.random.default_rng(473)
    precision,kl=precision_tables(evidence,rng)
    recovery=recovery_tables(evidence,rng)
    prediction,blocks,groups,correlations,profiles=block_tables(evidence)
    tables={'precision_task_effects':precision,'precision_domain_kl':kl,'imatrix_task_effects':recovery,
            'block_sensitivity':prediction,'block_recovery':blocks,'sensitive_group_recovery':groups,
            'sensitivity_recovery_agreement':correlations,'calibration_profile_agreement':profiles}
    for name,frame in tables.items():frame.to_csv(out/f'{name}.csv',index=False)
    plot_precision(precision,kl,out);plot_recovery(recovery,out);plot_blocks(prediction,blocks,groups,out)
    significant=recovery[recovery.passes_global_holm_005]
    lines=['# Three research questions: results guide','',
           'Open the PNGs in numerical order. All quantities come from saved results; this build launches no inference.','',
           '1. **Sensitivity:** `01` shows accuracy lost against BF16; `02` shows domain KL; `04` locates block sensitivity.',
           '2. **Recovery:** the three `03` figures show imatrix gains against matched controls. Positive means improvement.',
           '3. **Blocks:** `05` shows recovery and its concentration within a model; `06` compares sensitive blocks with the rest.','',
           '## Benchmark results passing global Holm correction','',
           'Tests cover all 162 model/precision/calibration/task comparisons. This is stricter than the earlier Llama-only family of 54.','']
    for r in significant.itertuples():
        lines.append(f'- {r.model} {r.precision}, {r.variant} calibration, {r.task}: {r.gain_pp:+.2f} pp (pointwise paired 95% interval {r.paired_ci95_low_pp:+.2f} to {r.paired_ci95_high_pp:+.2f}); adjusted p={r.holm_p_162:.4g}.')
    lines += ['', '## Interpretation', '',
              '- Each task has its own difficulty and baseline; percentage-point losses describe these particular benchmarks, not universal task difficulty.',
              '- Bootstrap intervals resample matched questions. They do not capture execution or calibration-sample variability.',
              '- Question identities include document, prompt and target hashes. Binary sample means are checked against aggregate metrics.',
              '- Precision effects use pure models; imatrix gains use same-recipe mixed controls. Do not subtract across those studies.',
              '- Historical calibration budgets are unequal. Domain-specific recovery is descriptive until the equal-budget study finishes.',
              '- Calibration-local weighted error omits correlations and downstream propagation. It is not accuracy loss.',
              '- Block recovery above whole-model recovery subtracts each calibration/measurement pair\'s pooled recovery. It reveals concentration beyond overall improvement.',
              '- Blocks and domains are related observations. No significance test treats block rows as independent repeated experiments.',
              '- Block sensitivity provenance has cross-experiment limitations; the copied input manifest is hashed below.',
              '- New equal-budget and repeated-inference results, when completed, are under `results/controlled_q4_study/`.','',
              'Rebuild: `python -m src.research_questions`. CSVs contain the plotted values and correctness transitions (wins/losses/ties).']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    evidence.track(ROOT/'results/final_analysis/questions/manifest.json')
    manifest={'seed':473,'bootstrap_resamples':10000,'interval':'pointwise paired question bootstrap, 95%',
              'multiplicity':{'precision_family':18,'imatrix_global_family':162,'secondary_within_model_family':54},
              'inputs':evidence.hashes,'code_sha256':sha(Path(__file__)),
              'outputs':{p.name:sha(p) for p in out.iterdir() if p.suffix in ['.csv','.png','.md']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {len(tables)} tables and {len(list(out.glob("*.png")))} plots to {out}',flush=True)


if __name__=='__main__':main()
