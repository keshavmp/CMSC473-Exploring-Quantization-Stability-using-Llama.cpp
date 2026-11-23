"""Screen fixed BF16 imatrix statistics against measured block sensitivity.

No fitting or choosing signs separately for precision/domain. PNG only.
"""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

import build_analysis as base

OUT = Path(__file__).resolve().parent / 'questions' / 'statistics'
LEVELS = ['Q2', 'Q4', 'Q6']
FEATURES = {
    'local_increment_pct': ('Weighted error', 'Existing precision-specific, reference-normalized Q-minus-Q8 local error'),
    'input_energy': ('Activation energy', 'Sum of seven tensor energies'),
    'concentration': ('Energy concentration', 'Mean of 100 minus normalized energy entropy; lower entropy ranks higher'),
    'kurtosis': ('Outlier strength', 'Mean tensor excess kurtosis of dimension-wise mean activations'),
    'abs_skew': ('Activation asymmetry', 'Mean absolute tensor skewness of dimension-wise mean activations'),
    'stddev': ('Activation spread', 'Mean tensor standard deviation of dimension-wise mean activations'),
    'abs_mean': ('Mean activation size', 'Mean absolute tensor mean; not mean absolute per-token activation'),
}


def build(prediction=None):
    OUT.mkdir(parents=True, exist_ok=True)
    if prediction is None:
        pieces = []
        for precision in LEVELS:
            frame = base.load_prediction(precision)
            frame['precision'] = precision
            pieces.append(frame)
        prediction = pd.concat(pieces, ignore_index=True)
    stats = base.csv('results/imatrix_full_pure_sweep/tensor_statistics.csv')
    stats = stats[stats.model.eq('Llama-3.2-1B-Instruct-BF16')].copy()
    stats['tensor_type'] = stats.tensor.str.extract(r'blk\.\d+\.(.*)\.weight')[0]
    base.coverage(stats, ['task', 'layer'])
    base.require(len(stats) == 336, 'Expected 336 BF16 tensor/domain rows')
    stats['abs_mean'] = stats['mean'].abs()
    stats['abs_skew'] = stats['skew'].abs()
    stats['concentration'] = 100 - stats.entropy_pct
    cols = ['abs_mean', 'abs_skew', 'concentration', 'stddev', 'kurtosis']
    base.require(np.isfinite(stats[cols].to_numpy()).all(), 'Nonfinite BF16 statistics')
    block = stats.groupby(['task', 'layer'], as_index=False)[cols].mean()
    block = block.rename(columns={'task': 'domain', 'layer': 'block'})
    joined = prediction.merge(block, on=['domain', 'block'], how='left', validate='many_to_one')
    base.require(len(joined) == 144 and joined[cols].notna().all().all(), 'Incomplete feature join')
    rows = []
    for (precision, domain), group in joined.groupby(['precision', 'domain']):
        measured = top3(group, 'sensitivity_nats')
        for feature in FEATURES:
            predicted = top3(group, feature)
            small = group[group.block.ne(1)]
            rows.append(dict(precision=precision, domain=domain, feature=feature,
                label=FEATURES[feature][0], n_blocks=16,
                rank_agreement=base.rho(group, feature),
                rank_agreement_without_block1=base.rho(small, feature),
                top3_hits=len(measured & predicted), top3_possible=3,
                predicted_top3=','.join(map(str, sorted(predicted))),
                measured_top3=','.join(map(str, sorted(measured)))))
    results = pd.DataFrame(rows)
    base.require(len(results) == 63 and results.notna().all().all(), 'Incomplete feature comparisons')
    by_precision = results.groupby(['feature', 'label', 'precision'], as_index=False).agg(
        rank_agreement=('rank_agreement', 'mean'),
        rank_agreement_without_block1=('rank_agreement_without_block1', 'mean'),
        top3_hits=('top3_hits', 'sum'), top3_possible=('top3_possible', 'sum'))
    by_precision['top3_identified_pct'] = 100 * by_precision.top3_hits / by_precision.top3_possible
    summary = results.groupby(['feature', 'label'], as_index=False).agg(
        mean_rank_agreement=('rank_agreement', 'mean'),
        mean_rank_agreement_without_block1=('rank_agreement_without_block1', 'mean'),
        top3_hits=('top3_hits', 'sum'), top3_possible=('top3_possible', 'sum'),
        positive_agreement_groups=('rank_agreement', lambda values: int(values.gt(0).sum())))
    summary['top3_identified_pct'] = 100 * summary.top3_hits / summary.top3_possible
    baseline = results[results.feature.eq('local_increment_pct')][['precision', 'domain', 'rank_agreement']].rename(
        columns={'rank_agreement': 'weighted_error_agreement'})
    results = results.merge(baseline, on=['precision', 'domain'], validate='many_to_one')
    results['difference_vs_weighted_error'] = results.rank_agreement - results.weighted_error_agreement
    summary = summary.sort_values('mean_rank_agreement', ascending=False)
    order = summary.feature.tolist()
    charts(by_precision, order)
    neighbor_summary = neighbor_checks(stats, joined)
    for filename, frame in {'feature_blocks.csv': joined, 'comparisons_by_domain.csv': results,
                            'comparisons_by_precision.csv': by_precision, 'summary.csv': summary}.items():
        frame.to_csv(OUT / filename, index=False, lineterminator='\n')
    definitions = pd.DataFrame([dict(feature=k, label=v[0], definition=v[1], direction='Higher predicts more sensitivity')
                                for k, v in FEATURES.items()])
    definitions.to_csv(OUT / 'feature_definitions.csv', index=False, lineterminator='\n')
    write_report(summary, neighbor_summary)
    manifest = {'scope': 'Exploratory fixed-feature screen: Llama BF16 predictors, Q2/Q4/Q6 block sensitivity',
                'input_sha256': dict(base.INPUTS),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json'},
                'design': ['seven complete scores on 16 blocks; four additional neighbor scores on matched 15-block subsets',
                           'higher always predicts greater sensitivity; 11 distinct scores screened in total',
                           'fixed equal mean of seven tensor summaries, except summed energy and pooled local error',
                           'Spearman calculated separately over 16 blocks in each of nine groups',
                           'summary averages signed correlations equally across domains/precisions',
                           'no feature fitting, no sign flipping, no p-values or held-out validation']}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(summary.to_string(index=False))
    return summary


def neighbor_checks(stats, joined):
    """Neighbor-comparison fields lack block 0; compare every score on the same 15 blocks."""
    extra = stats.copy()
    extra['neighbor_difference'] = 1 - extra.pearson
    extra['abs_covariance'] = extra.covariance.abs()
    fields = ['gain', 'l2_distance', 'neighbor_difference', 'abs_covariance']
    base.require(extra[extra.layer.eq(0)][fields].isna().all().all(), 'Unexpected block-0 neighbor features')
    base.require(extra[extra.layer.ne(0)][fields].notna().all().all(), 'Incomplete neighbor features')
    blocks = extra.groupby(['task', 'layer'], as_index=False)[fields].mean().rename(columns={'task': 'domain', 'layer': 'block'})
    data = joined.merge(blocks, on=['domain', 'block'], validate='many_to_one')
    data = data[data.block.ne(0)]
    labels = {key: value[0] for key, value in FEATURES.items()}
    labels.update(gain='Neighbor energy gain', l2_distance='Neighbor activation distance',
                  neighbor_difference='Neighbor activation difference (1 - Pearson)',
                  abs_covariance='Absolute neighbor covariance')
    rows = []
    for (precision, domain), group in data.groupby(['precision', 'domain']):
        for feature, label in labels.items():
            small = group[group.block.ne(1)]
            rows.append(dict(precision=precision, domain=domain, feature=feature, label=label,
                n_blocks=15, rank_agreement=base.rho(group, feature),
                rank_agreement_without_block1=base.rho(small, feature),
                top3_hits=len(top3(group, feature) & top3(group, 'sensitivity_nats'))))
    detail = pd.DataFrame(rows)
    base.require(len(detail) == 99 and detail.notna().all().all(), 'Incomplete neighbor comparison')
    summary = detail.groupby(['feature', 'label'], as_index=False).agg(
        mean_rank_agreement=('rank_agreement', 'mean'),
        mean_rank_agreement_without_block1=('rank_agreement_without_block1', 'mean'),
        top3_hits=('top3_hits', 'sum'))
    summary = summary.sort_values('mean_rank_agreement', ascending=False)
    detail.to_csv(OUT / 'neighbor_comparisons_15_blocks.csv', index=False, lineterminator='\n')
    summary.to_csv(OUT / 'neighbor_summary_15_blocks.csv', index=False, lineterminator='\n')
    return summary


def top3(group, column):
    return set(group.sort_values([column, 'block'], ascending=[False, True]).head(3).block)


def charts(by_precision, order):
    labels = [FEATURES[k][0] for k in order]
    for metric, name, title in [
        ('top3_identified_pct', '01_find_critical_blocks', 'Which score finds the most sensitive blocks?'),
        ('rank_agreement', '02_match_full_ranking', 'Which score best matches the full block order?')]:
        values = by_precision.pivot(index='feature', columns='precision', values=metric).loc[order, LEVELS]
        fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        if metric == 'rank_agreement':
            im = ax.imshow(values, cmap='RdBu', norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect='auto')
            foot = 'Average rank agreement across code / math / chat.\n1 = same order; 0 = no trend; -1 = opposite order.'
            ticks = [-1, 0, 1]
        else:
            im = ax.imshow(values, cmap='Blues', vmin=0, vmax=100, aspect='auto')
            foot = 'Correct top-3 selections across code / math / chat: 9 possible per precision.\nDarker blue = more correct identifications.'
            ticks = [0, 50, 100]
        for r in range(len(order)):
            for c in range(3):
                value = values.iloc[r, c]
                if metric == 'rank_agreement':
                    label = f'{value:+.2f}'
                else:
                    n = int(by_precision[(by_precision.feature.eq(order[r])) &
                                         by_precision.precision.eq(LEVELS[c])].top3_hits.iloc[0])
                    label = f'{n}/9'
                ax.text(c, r, label, ha='center', va='center', fontsize=13,
                        color='white' if (metric != 'rank_agreement' and value > 55) or
                        (metric == 'rank_agreement' and abs(value) > .6) else '#162c40')
        ax.set_xticks(range(3), LEVELS)
        ax.set_yticks(range(len(order)), labels)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=14)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(im, ax=ax, shrink=.7, ticks=ticks,
                     label='Rank agreement' if metric == 'rank_agreement' else 'Identified (%)')
        fig.supxlabel(foot, fontsize=10)
        fig.savefig(OUT / f'{name}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)


def write_report(summary, neighbor_summary):
    text = '''# Can other imatrix statistics predict sensitivity better?

**It depends on the goal.** Energy concentration identifies more of the top-three
sensitive blocks. Weighted error still best matches the full ordering of blocks.

## Find the critical blocks

![Identify critical blocks](01_find_critical_blocks.png)

Energy concentration identifies **18 of 27** domain-specific top-three selections,
compared with **13 of 27** for weighted error and **14 of 27** for activation energy.
These are repeated selections across nine precision/domain groups, not 27 unique blocks.

## Rank all 16 blocks

![Match full ordering](02_match_full_ranking.png)

Weighted error has the strongest average agreement (**0.39**), followed by outlier
strength (**0.19**). Energy concentration averages **0.10**. No additional statistic
beats weighted error on full-order agreement averaged over the nine groups.

## What each score means

'''
    defs = pd.DataFrame([{'Score': label, 'Definition': definition} for label, definition in FEATURES.values()])
    text += base.table(defs)
    text += '''

The added features use the BF16 model only. Mean/spread/skew/kurtosis describe the
distribution of dimension-wise **mean activations**, not the distribution of
individual token activations. Concentration uses **100 minus normalized energy
entropy**. It is an inverse-entropy score, not a new physical quantity.

Each block score is the equal mean of its seven tensor summaries, except energy
(sum) and weighted error (ratio of pooled errors to pooled reference energy).
Average tensor kurtosis or entropy is not a pooled block-distribution statistic.
Absolute mean and skew are taken per tensor before averaging. All directions
are fixed: larger values predict greater sensitivity. Signs are never flipped
after examining a precision or domain. Negative correlation means the chosen
direction fails, not that the statistic has been converted into a good predictor.

## Checks and interpretation limits

- Correlation means Spearman rank agreement, calculated within each 16-block
  precision/domain group. Summary values are equal arithmetic averages of signed
  correlations, not correlations over pooled raw measurements. Increasing rescaling
  does not change these correlations, so no arbitrary 0-100 feature scaling is needed.
- Top-three overlap is normalized by nine selections per precision. Ties at the
  boundary are resolved by lower block index; rank correlation uses average ties.
- Removing block 1 weakens mean agreement: weighted error falls to about **0.26**,
  outlier strength to **0.01**, and concentration to **-0.09**. Success on a few
  critical blocks does not imply a reliable full ranking.
- The primary charts screen seven complete scores on the existing data. Results are exploratory; choosing
  the best score here needs validation on new text or interventions. There are no
  p-values or claims that one score is statistically superior. Related domains and
  precisions are not independent trials. No regression was fitted to 16 blocks.
- Gain, L2 distance, Pearson and covariance compare a tensor with the corresponding
  tensor in a preceding block. They are present for blocks 1-15 but unavailable for
  block 0. The separate comparison below uses exactly the same 15 blocks for every
  predictor, rather than filling block 0 with a fabricated value. Signed raw
  mean/skew were not separately searched, nor were many alternative aggregations.
- Source summary values are rounded. Imatrix budgets differ across domains. Measured
  sensitivity uses one lowered block on a Q8 background; cross-machine provenance
  limitations from the main report still apply. No new model inference was run.

**Recommendation:** keep weighted error as the current full-ranking baseline and
energy concentration as a promising top-block selector. Do not claim that one
statistic is best for every purpose or precision.

Exact comparisons and outlier checks: `comparisons_by_domain.csv`,
`comparisons_by_precision.csv`, and `summary.csv`. Joined block features:
`feature_blocks.csv`. Definitions: `feature_definitions.csv`. Hashes: `manifest.json`.

Rebuild: `python results/final_analysis/build_statistic_comparison.py`.
Charts are PNG only.
'''
    text += '''

## Additional check: neighboring-block statistics

These results exclude block 0 for **every** score. Weighted error still has the
strongest average full-ranking agreement (about 0.50). Neighbor activation
difference, distance and energy gain are promising alternatives, but none beats
that matched baseline. These numbers must not be directly compared with the
16-block chart above: the target set and its top three have changed.

'''
    text += base.table(neighbor_summary[['label', 'mean_rank_agreement', 'mean_rank_agreement_without_block1', 'top3_hits']])
    text += '''

Gain is the ratio of current to preceding tensor RMS energy; L2 and Pearson
compare dimension-wise mean-activation vectors in the exported nonlegacy data.
We use 1 minus Pearson so larger means more neighbor difference, and absolute
covariance so sign does not cancel across tensors. All are equal means of seven
tensor summaries. These transformations are fixed for every domain and precision.
The local llama.cpp statistic implementation and existing repository explanations
were inspected to distinguish neighbor comparisons from quantized-versus-BF16 drift.

See `neighbor_comparisons_15_blocks.csv` and `neighbor_summary_15_blocks.csv`.
Across primary and supplemental checks, 11 distinct scores were screened; this
remains an exploratory comparison on the same measured sensitivity data.
'''
    (OUT / 'README.md').write_text(text, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    build()
