"""P2 imatrix research pipeline. CSV/PNG/Markdown outputs, no JSON required.

Commands: analyze, recommend, validate, compare. Run --help for options.
"""
from __future__ import annotations
import argparse
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DOMAINS = ('chat', 'math', 'code')
LAYER_RE = re.compile(r'(?:^|\.)blk\.(\d+)\.')
EPS = 1e-12


def parse_layer(name: str):
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def load_domain(path: Path, domain: str) -> pd.DataFrame:
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise SystemExit('Install dependencies: python -m pip install gguf numpy pandas matplotlib') from exc
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Run src/imatrix.py first.')
    reader = GGUFReader(str(path))
    tensors = {t.name: t for t in reader.tensors}
    rows = []
    for name, tensor in tensors.items():
        if not name.endswith('.in_sum2'):
            continue
        base = name[:-8]
        counts_t = tensors.get(base + '.counts')
        if counts_t is None:
            raise ValueError(f'Missing counts for {base}')
        sums = np.asarray(tensor.data, dtype=np.float64).reshape(-1)
        counts = np.asarray(counts_t.data, dtype=np.float64).reshape(-1)
        if len(counts) == 0 or len(sums) % len(counts):
            raise ValueError(f'Unexpected imatrix dimensions: {base}')
        values = sums.reshape(len(counts), -1)
        valid = counts > 0
        if not np.any(valid):
            continue
        # Each row of the imatrix is averaged over the row's count; use
        # an equally weighted summary of activation coordinates.
        energies = (values[valid] / counts[valid, None]).ravel()
        energies = energies[np.isfinite(energies)]
        if not len(energies):
            continue
        layer = parse_layer(base)
        rows.append(dict(domain=domain, tensor=base, layer=layer,
                         coordinates=len(energies), observations=float(counts[valid].sum()),
                         mean_energy=float(np.mean(energies)),
                         median_energy=float(np.median(energies)),
                         max_energy=float(np.max(energies)),
                         std_energy=float(np.std(energies)),
                         p95_energy=float(np.percentile(energies, 95))))
    if not rows:
        raise ValueError(f'No .in_sum2 / .counts pairs found in {path}; check GGUF format')
    return pd.DataFrame(rows)


def scaled_score(values: pd.Series) -> pd.Series:
    # Per-domain tensor normalization by relative rank limits distortions
    # from unlike tensor families or absolute units.
    return values.rank(method='average', pct=True)


def write_report(path: Path, title: str, lines: list[str]):
    path.write_text('# ' + title + '\n\n' + '\n\n'.join(lines) + '\n', encoding='utf-8')


def analyze(indir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    frames = [load_domain(indir / f'{d}.imatrix.gguf', d) for d in DOMAINS]
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(out / 'tensor_statistics.csv', index=False)
    # Only tensor names present in ALL domains can support a fair comparison.
    common = set.intersection(*(set(f.tensor) for f in frames))
    df = raw[raw.tensor.isin(common) & raw.layer.notna()].copy()
    if df.empty:
        raise ValueError('No shared transformer-block tensors across domains')
    df['layer'] = df.layer.astype(int)
    # Scores are a deliberately simple heuristic, NOT a measured quantization loss.
    df['energy_log'] = np.log10(df.mean_energy.clip(lower=EPS))
    df['tensor_rank'] = df.groupby('domain')['energy_log'].transform(scaled_score)
    # Compare the same tensor between domains without assuming raw units match
    df['cross_domain_ratio'] = df.mean_energy / df.groupby('tensor').mean_energy.transform('mean').clip(lower=EPS)
    df.to_csv(out / 'shared_tensor_features.csv', index=False)
    layers = (df.groupby(['domain','layer'], as_index=False)
              .agg(tensor_count=('tensor','nunique'), log_energy=('energy_log','mean'),
                   heuristic_score=('tensor_rank','mean'),
                   relative_domain_energy=('cross_domain_ratio','mean')))
    layers.to_csv(out / 'layer_statistics.csv', index=False)
    scores = layers.pivot(index='layer', columns='domain', values='heuristic_score').reindex(columns=DOMAINS)
    scores.to_csv(out / 'layer_scores.csv', index_label='layer')
    corr = scores.corr(method='spearman')
    corr.to_csv(out / 'domain_correlations.csv')
    # Aggregate across domains via rank, so all domains contribute equally.
    aggregate = scores.dropna().copy()
    aggregate['score'] = aggregate.mean(axis=1)
    aggregate = aggregate.sort_values('score', ascending=False)
    aggregate['predicted_rank'] = range(1, len(aggregate)+1)
    aggregate.reset_index().to_csv(out / 'predicted_layer_ranking.csv', index=False)

    fig, ax = plt.subplots(figsize=(12,5))
    for d in DOMAINS:
        ax.plot(scores.index, scores[d], marker='.', label=d)
    ax.set(xlabel='Transformer layer', ylabel='Mean within-domain tensor percentile',
           title='Imatrix heuristic score by layer (not measured sensitivity)')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/'layer_importance.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12,4))
    img = ax.imshow(scores.T.to_numpy(), aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_yticks(range(3), DOMAINS)
    ax.set_xticks(range(len(scores.index)), [str(x) for x in scores.index], rotation=90)
    ax.set(xlabel='Transformer layer', title='Layer importance score by calibration domain')
    fig.colorbar(img, ax=ax, label='Mean tensor percentile')
    fig.tight_layout(); fig.savefig(out/'domain_heatmap.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(3), DOMAINS); ax.set_yticks(range(3), DOMAINS)
    for i in range(3):
        for j in range(3):
            v = corr.iloc[i,j]
            ax.text(j,i,'NA' if not np.isfinite(v) else f'{v:.2f}',ha='center',va='center')
    ax.set_title('Spearman rank correlation across layer scores')
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out/'domain_correlation.png', dpi=180); plt.close(fig)

    bullets = [f'- {d}: {len(f)} tensors with imatrix data' for d,f in zip(DOMAINS, frames)]
    top = ', '.join(str(int(x)) for x in aggregate.head(5).index)
    write_report(out/'REPORT.md','P2 Imatrix domain analysis',[
        '## Input coverage\n'+'\n'.join(bullets),
        f'## Preliminary layer ranking\nTop five layers by **heuristic** score: {top}.',
        '## Interpretation\nThe scores rank average within-domain tensor activation energy. This is an exploratory proxy, **not** verified quantization sensitivity. Raw values vary by tensor type, so absolute comparisons should be treated cautiously.',
        '## Validation plan\nCompare `predicted_layer_ranking.csv` against independent P1 layer-level perturbation measurements. Use `validate` with a `layer,sensitivity` CSV, where higher sensitivity means greater damage.',
        '## Reproducibility\nSame BF16 checkpoint, same dataset splits and seed, and same imatrix generation options must be retained for meaningful comparisons.'
    ])
    print('Wrote analysis outputs to', out)
    print('Top layers:', top)
    return aggregate


def recommend(out: Path, budget: float, low: str, high: str):
    if not 0 <= budget <= 1:
        raise ValueError('budget must be a fraction between 0 and 1')
    ranking_path = out/'predicted_layer_ranking.csv'
    if not ranking_path.exists():
        raise FileNotFoundError('Run analyze first')
    ranked = pd.read_csv(ranking_path).sort_values('score',ascending=False)
    n = len(ranked)
    high_n = round(n*budget)
    chosen = set(ranked.head(high_n).layer.astype(int))
    rows=[]
    # Exact weight-tensor names (from imatrix) will be used for overrides;
    # tensor type rules are verified by inspecting actual output GGUF later.
    features = pd.read_csv(out/'shared_tensor_features.csv')
    names = sorted(set(features.tensor))
    for name in names:
        layer = parse_layer(name)
        if layer is None:
            continue
        rows.append((name, layer, high if layer in chosen else low))
    plan = pd.DataFrame(rows,columns=['tensor','layer','type'])
    plan.to_csv(out/'precision_plan.csv',index=False)
    # llama-quantize accepts whitespace-separated tensor_name=ggml_type entries.
    (out/'tensor_types.txt').write_text(''.join(f'{r.tensor}={r.type}\n' for r in plan.itertuples()),encoding='utf-8')
    write_report(out/'RECOMMENDATION.md','Mixed-precision preliminary allocation',[
        f'Layers: {n}. High-precision layers: {len(chosen)} ({budget:.0%} budget).',
        f'Heuristic top layers: {", ".join(map(str,sorted(chosen)))}.',
        f'Low type: `{low}`. High type: `{high}`.',
        'This is a **fraction of layers**, not an exact memory budget. Actual GGUF size and accuracy must be measured; tensors without matching imatrix entries retain llama.cpp default behavior. Verify quantization type fallbacks and the output model tensor types.',
        'To build, run `quantize`; evaluate the resulting GGUF with your existing `run_evaluation.py`, then compare its CSV task scores with uniform baselines.'
    ])
    print('Wrote',out/'tensor_types.txt','and precision_plan.csv')


def quantize(out: Path, executable: Path, source: Path, destination: Path, imatrix: Path | None, dry_run: bool):
    if not executable.is_file() or not source.is_file():
        raise FileNotFoundError('llama-quantize executable and BF16 source GGUF must exist')
    overrides = out/'tensor_types.txt'
    if not overrides.is_file():
        raise FileNotFoundError('Run recommend first')
    destination.parent.mkdir(parents=True,exist_ok=True)
    cmd = [str(executable),'--tensor-type-file',str(overrides)]
    if imatrix is not None:
        if not imatrix.is_file():
            raise FileNotFoundError(imatrix)
        cmd += ['--imatrix',str(imatrix)]
    if dry_run:
        cmd += ['--dry-run']
    cmd += [str(source),str(destination),'Q4_K_M']
    print('Running:',subprocess.list2cmdline(cmd))
    subprocess.run(cmd, check=True)
    if not dry_run and destination.is_file():
        pd.DataFrame([{'model':destination.name,'size_mib':destination.stat().st_size/(1024**2)}]).to_csv(out/'mixed_model_size.csv',index=False)


def validate(out: Path, p1: Path):
    # P1 requires independent layer-level perturbation scores, not whole-model Q scores.
    measured=pd.read_csv(p1)
    if not {'layer','sensitivity'}.issubset(measured.columns):
        raise ValueError('P1 CSV must have columns layer,sensitivity; higher means more damage')
    pred=pd.read_csv(out/'predicted_layer_ranking.csv')
    measured = measured[['layer','sensitivity']].dropna()
    if measured.layer.duplicated().any():
        raise ValueError('P1 CSV contains repeated layers; aggregate comparable experiments first')
    match=pred.merge(measured,on='layer',how='inner')
    if len(match)<3:
        raise ValueError('Need >=3 matching layers to correlate P1 and P2')
    match.to_csv(out/'p1_p2_matched.csv',index=False)
    rho=match[['score','sensitivity']].corr(method='spearman').iloc[0,1]
    pearson=match[['score','sensitivity']].corr(method='pearson').iloc[0,1]
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(match.score,match.sensitivity)
    for r in match.itertuples():
        ax.annotate(str(r.layer),(r.score,r.sensitivity),fontsize=7)
    ax.set(xlabel='P2 imatrix heuristic score',ylabel='P1 observed sensitivity (higher = more damage)',title='P2 prediction vs P1 measurement')
    fig.tight_layout(); fig.savefig(out/'p1_validation.png',dpi=180);plt.close(fig)
    write_report(out/'P1_VALIDATION.md','P1 vs P2 validation',[
        f'Matched layers: {len(match)}. Spearman rho: **{rho:.3f}**. Pearson r: **{pearson:.3f}**.',
        'These are descriptive correlations, not causal proof. Ensure P1 measurements arise from controlled same-checkpoint, same-evaluation layer perturbations.'
    ])
    print(f'P1 validation: matched={len(match)}, spearman={rho:.3f}, pearson={pearson:.3f}')


def compare(out: Path, scores: Path):
    # User-controlled evaluation CSV from actual runs, not simulated results.
    # CSV format: model,method,size_mib,gsm8k,ifeval,mmlu[,kl]
    df=pd.read_csv(scores)
    required={'model','method','size_mib','gsm8k','ifeval','mmlu'}
    if not required.issubset(df.columns):
        raise ValueError(f'Evaluation CSV requires: {sorted(required)}')
    if df.empty: raise ValueError('Evaluation CSV is empty')
    df=df.copy()
    df['mean_accuracy']=df[['gsm8k','ifeval','mmlu']].mean(axis=1)
    # Non-dominated points: lower size, higher average accuracy.
    df['pareto']=True
    for i,a in df.iterrows():
        dominated=((df.size_mib<=a.size_mib)&(df.mean_accuracy>=a.mean_accuracy)&
                   ((df.size_mib<a.size_mib)|(df.mean_accuracy>a.mean_accuracy))).any()
        df.loc[i,'pareto']=not bool(dominated)
    df.to_csv(out/'quality_size_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5))
    for method,group in df.groupby('method'):
        ax.scatter(group.size_mib,group.mean_accuracy,label=method)
        for r in group.itertuples(): ax.annotate(str(r.model),(r.size_mib,r.mean_accuracy),fontsize=7)
    frontier=df[df.pareto].sort_values('size_mib')
    if len(frontier)>1: ax.plot(frontier.size_mib,frontier.mean_accuracy,linestyle='--',alpha=.7)
    ax.set(xlabel='GGUF size (MiB)',ylabel='Mean benchmark accuracy (consistent scale)',title='Measured quality-size Pareto frontier')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'pareto_frontier.png',dpi=180);plt.close(fig)
    write_report(out/'MODEL_COMPARISON.md','Mixed vs uniform quantization',[
        f'Measured models: {len(df)}; non-dominated on quality/size: {len(frontier)}.',
        'Only include benchmark scores produced with matched evaluation tasks, dataset versions, prompts and metric definitions. This comparison uses simple mean of the three scores; report per-task outcomes separately. KL, if supplied, is not incorporated in the frontier.'
    ])
    print('Wrote quality-size comparison and Pareto chart')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['analyze','recommend','quantize','validate','compare'])
    p.add_argument('--input-dir',type=Path,default=ROOT/'models'/'imatrix')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results'/'imatrix_analysis')
    p.add_argument('--high-fraction',type=float,default=0.25)
    p.add_argument('--low',default='Q4_K',help='Tensor ggml type; e.g. Q4_K')
    p.add_argument('--high',default='Q6_K',help='Tensor ggml type; e.g. Q6_K')
    p.add_argument('--quantizer',type=Path,default=Path.home()/'llama.cpp'/'build-cpu'/'bin'/'llama-quantize')
    p.add_argument('--source',type=Path,default=ROOT/'models'/'Llama-3.2-1B-Instruct-BF16.gguf')
    p.add_argument('--destination',type=Path,default=ROOT/'models'/'mixed-q4-q6.gguf')
    p.add_argument('--imatrix',type=Path,default=None)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--p1-csv',type=Path)
    p.add_argument('--scores-csv',type=Path)
    a=p.parse_args()
    if a.command=='analyze': analyze(a.input_dir,a.output_dir)
    elif a.command=='recommend': recommend(a.output_dir,a.high_fraction,a.low,a.high)
    elif a.command=='quantize': quantize(a.output_dir,a.quantizer,a.source,a.destination,a.imatrix,a.dry_run)
    elif a.command=='validate':
        if a.p1_csv is None: p.error('--p1-csv is required')
        validate(a.output_dir,a.p1_csv)
    elif a.command=='compare':
        if a.scores_csv is None: p.error('--scores-csv is required')
        compare(a.output_dir,a.scores_csv)

if __name__=='__main__': main()
