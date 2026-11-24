"""P2: correlate importance-matrix features with measured quantization outcomes.

Inputs (CSV):
  results/imatrix_analysis/shared_tensor_features.csv  (from imatrix_research.py analyze)
  results/important plots/activation_weighted_errors.csv (optional, actual weights)
  results/imatrix_analysis/p1_layer_sensitivity.csv (optional, from layer ablations)

Outputs: only CSV, PNG, Markdown. No benchmark data is invented.
Run from your repository root, regardless of where this script lives:
  python src/imatrix_correlations.py
  python src/imatrix_correlations.py --weighted-errors 'results/important plots/activation_weighted_errors.csv' --p1 results/imatrix_analysis/p1_layer_sensitivity.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURES = ['mean_energy', 'median_energy', 'std_energy', 'p95_energy',
            'max_energy', 'tensor_rank', 'cross_domain_ratio']
OUTCOMES = ['weighted_error', 'sensitivity', 'sensitivity_abs',
            'gsm8k_drop', 'ifeval_drop', 'mmlu_drop']


def repo_root():
    """Search upwards for repository anchor, including nested installation."""
    for parent in Path(__file__).resolve().parents:
        if (parent / 'run_evaluation.py').exists() and (parent / 'models').is_dir():
            return parent
    return Path.cwd().resolve()


def read_csv(path, required):
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'{path} missing columns: {sorted(missing)}')
    return df


def correlations(frame, features, outcomes, grouping, label):
    rows = []
    for keys, group in frame.groupby(grouping, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(grouping, keys))
        for x in features:
            if x not in group:
                continue
            for y in outcomes:
                if y not in group:
                    continue
                pair = group[[x, y]].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
                n = len(pair)
                pearson = spearman = np.nan
                if n >= 3 and pair[x].nunique() > 1 and pair[y].nunique() > 1:
                    pearson = pair[x].corr(pair[y], method='pearson')
                    spearman = pair[x].corr(pair[y], method='spearman')
                rows.append({**meta, 'analysis': label, 'feature': x, 'outcome': y,
                             'n': n, 'pearson_r': pearson, 'spearman_rho': spearman})
    return pd.DataFrame(rows)


def scatter(df, x, y, title, destination):
    subset = df[[x, y]].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(subset[x], subset[y], s=30, alpha=.7)
    ax.set(xlabel=x.replace('_',' '), ylabel=y.replace('_',' '), title=title)
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(destination,dpi=180); plt.close(fig)


def heatmap(corr, path, title):
    if corr.empty:
        return
    grid = corr.pivot_table(index='feature',columns='outcome',values='spearman_rho',aggfunc='first')
    if grid.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(grid.columns)*1.6),max(4,len(grid)*.65)))
    im = ax.imshow(grid.to_numpy(float), cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(grid.columns)),grid.columns,rotation=30,ha='right')
    ax.set_yticks(range(len(grid.index)),grid.index)
    for (i,j), value in np.ndenumerate(grid.to_numpy(float)):
        ax.text(j,i, f'{value:.2f}' if np.isfinite(value) else 'NA', ha='center',va='center',fontsize=8)
    ax.set_title(title); fig.colorbar(im,ax=ax,label='Spearman rank correlation')
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def main(argv=None):
    root = repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=root/'results/imatrix_analysis/shared_tensor_features.csv')
    p.add_argument('--weighted-errors',type=Path,default=root/'results/important plots/activation_weighted_errors.csv')
    p.add_argument('--p1',type=Path,default=root/'results/imatrix_analysis/p1_layer_sensitivity.csv')
    p.add_argument('--output',type=Path,default=root/'results/imatrix_correlations')
    args=p.parse_args(argv)
    features=read_csv(args.features,['domain','tensor','layer','mean_energy'])
    if features is None:
        raise SystemExit(f'No imatrix features: {args.features}. Run imatrix_research.py analyze first.')
    args.output.mkdir(parents=True,exist_ok=True)
    features=features.copy()
    if features.duplicated(['domain','tensor']).any():
        raise ValueError('Duplicate domain/tensor features')
    features['layer']=pd.to_numeric(features['layer'],errors='coerce')
    feat_cols=[c for c in FEATURES if c in features.columns]
    if 'mean_energy' not in feat_cols:
        raise ValueError('Missing mean_energy')
    rows=[]; notes=['# Imatrix feature correlations','',
        'These are exploratory associations, NOT evidence that activation importance causes quantization sensitivity.',
        'The imatrix datasets are calibration domains, distinct from benchmark test sets.','']
    # Domain comparison based on the exact same tensor across calibration corpora.
    pivot=features.pivot(index='tensor',columns='domain',values='mean_energy')
    domain_rows=[]
    for i,a in enumerate(pivot.columns):
        for b in pivot.columns[i+1:]:
            pair=pivot[[a,b]].replace([np.inf,-np.inf],np.nan).dropna()
            valid=len(pair)>=3 and pair[a].nunique()>1 and pair[b].nunique()>1
            domain_rows.append(dict(domain_a=a,domain_b=b,n_tensors=len(pair),
                spearman_rho=pair[a].corr(pair[b],method='spearman') if valid else np.nan,
                pearson_r=pair[a].corr(pair[b],method='pearson') if valid else np.nan))
    pd.DataFrame(domain_rows,columns=['domain_a','domain_b','n_tensors','spearman_rho','pearson_r']).to_csv(args.output/'domain_tensor_correlations.csv',index=False)
    notes.append(f'## Cross-domain correlations\nCompared matched tensor energy across {len(domain_rows)} domain pairs. See `domain_tensor_correlations.csv`.')

    errors=read_csv(args.weighted_errors,['task','tensor','weighted_error'])
    if errors is not None:
        errors=errors.rename(columns={'task':'domain'}).copy()
        # This source can include multiple quantization models and levels; never pool them blindly.
        strata=[c for c in ['model','precision'] if c in errors.columns]
        if not strata:
            raise ValueError('Weighted errors require model or precision identifier to prevent pooled correlation')
        keys=['domain','tensor']+strata
        if errors.duplicated(keys).any():
            raise ValueError('Repeated weighted-error rows for same domain, tensor and model/precision')
        merged=features.merge(errors,on=['domain','tensor'],how='inner',suffixes=('_imatrix','_error'),validate='one_to_many')
        if merged.empty:
            notes.append('## Weighted error\nNo matching domain/tensor pairs were found.')
        else:
            merged.to_csv(args.output/'matched_tensor_errors.csv',index=False)
            analysis=correlations(merged,feat_cols,['weighted_error'],['domain']+strata,'tensor_error')
            rows.append(analysis)
            notes.append(f'## Weighted error\nMatched {len(merged)} domain/tensor/model rows. Comparisons are stratified by quantized model and precision. These errors are local diagonal approximations, not benchmark accuracy.')
            # Choose a deterministic single stratum to illustrate without aggregating across model types.
            subset=merged.sort_values(['domain']+strata)
            selected=subset.iloc[0]
            selection=np.ones(len(subset),dtype=bool)
            for k in ['domain']+strata:
                selection &= subset[k].eq(selected[k]).to_numpy()
            scatter(subset.loc[selection], 'mean_energy','weighted_error',
                    'Activation energy vs local quantization error',args.output/'tensor_energy_vs_weighted_error.png')
    else:
        notes.append('## Weighted error\nNot available. Supply `--weighted-errors` from actual BF16/quantized weights. No values were fabricated.')

    p1=read_csv(args.p1,['layer','sensitivity'])
    if p1 is not None:
        p1=p1.copy(); p1['layer']=pd.to_numeric(p1['layer'],errors='raise')
        if p1.layer.duplicated().any():
            raise ValueError('P1 has repeated layer measurements; aggregate controlled replicates before correlating')
        # Aggregate each imatrix feature by layer and domain, then equal-weight domains.
        by_domain=features.groupby(['domain','layer'],as_index=False)[feat_cols].mean()
        by_layer=by_domain.groupby('layer',as_index=False)[feat_cols].mean()
        matched=by_layer.merge(p1,on='layer',how='inner',validate='one_to_one')
        matched.to_csv(args.output/'matched_p1_layers.csv',index=False)
        outcomes=[c for c in OUTCOMES if c in matched.columns]
        if len(matched)>=3:
            rows.append(correlations(matched.assign(scope='all_layers'), feat_cols, outcomes,
                                     ['scope'], 'p1_layer_sensitivity'))
            scatter(matched,'mean_energy','sensitivity','Imatrix energy vs measured P1 layer sensitivity',args.output/'p1_energy_vs_sensitivity.png')
        notes.append(f'## P1 layer sensitivity\nMatched {len(matched)} measured P1 layers. Layer scores average tensor features within domains and weight each domain equally. Correlation is descriptive and depends on the P1 perturbation protocol.')
    else:
        notes.append('## P1 layer sensitivity\nNot yet available. Run controlled single-layer ablations; correlations will be computed when their CSV is supplied.')

    if rows:
        results=pd.concat(rows,ignore_index=True)
    else:
        results=pd.DataFrame(columns=['analysis','feature','outcome','n','pearson_r','spearman_rho'])
    results.to_csv(args.output/'feature_outcome_correlations.csv',index=False)
    if not results.empty:
        for kind,group in results.groupby('analysis'):
            # Heatmap only for P1 full-layer summary, otherwise strata require separate comparisons.
            if kind=='p1_layer_sensitivity':
                heatmap(group,args.output/'p1_correlation_heatmap.png','Imatrix features vs measured layer sensitivity')
        significant=results[np.isfinite(results['spearman_rho'])].copy()
        for row in significant.reindex(significant.spearman_rho.abs().sort_values(ascending=False).index).head(8).itertuples():
            notes.append(f'- {row.analysis}: {row.feature} vs {row.outcome}: Spearman rho={row.spearman_rho:.3f} (n={row.n})')
    notes += ['', '## Limitations','Correlations are not causal, small samples may be unstable, and layers/tensors from one model are not independent experimental replicates. No significance or confidence claims are made. Compare consistent task metrics and model checkpoints.']
    (args.output/'CORRELATION_REPORT.md').write_text('\n\n'.join(notes)+'\n',encoding='utf-8')
    print('Results saved to',args.output)
    print('Rows of feature/outcome correlations:',len(results))

if __name__=='__main__':
    main()
