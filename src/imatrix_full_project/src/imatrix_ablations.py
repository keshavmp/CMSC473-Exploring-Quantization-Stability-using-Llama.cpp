"""Build reproducible P1 layer perturbations and P2 mixed/uniform baselines.

Outputs CSV manifests and log files, not fabricated benchmark metrics.  GGUF
models stay inside models/ and should not be committed to ordinary Git.

Use: python src/imatrix_ablations.py plan
     python src/imatrix_ablations.py build --only mixed-chat
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / 'results' / 'imatrix_analysis'
MODEL_DIR = ROOT / 'models' / 'imatrix_experiments'
SOURCE = ROOT / 'models' / 'Llama-3.2-1B-Instruct-BF16.gguf'
BINARY = Path.home() / 'llama.cpp' / 'build-cpu' / 'bin' / 'llama-quantize'
DOMAINS = ('chat', 'math', 'code')


def load_tensors(out: Path) -> pd.DataFrame:
    path = out / 'shared_tensor_features.csv'
    if not path.exists():
        raise FileNotFoundError('Run imatrix_research.py analyze first')
    frame = pd.read_csv(path)
    if not {'tensor', 'layer', 'domain'}.issubset(frame.columns):
        raise ValueError('Feature CSV missing tensor, layer, or domain columns')
    frame = frame.dropna(subset=['layer']).copy()
    frame['layer'] = frame.layer.astype(int)
    return frame


def load_rank(out: Path, domain: str) -> pd.DataFrame:
    scores = pd.read_csv(out / 'layer_scores.csv')
    if domain not in scores.columns:
        raise ValueError(f'Missing domain column {domain}')
    return scores[['layer', domain]].dropna().sort_values(domain, ascending=False)


def overrides(out: Path, name: str, mapping: dict[int, str]) -> Path:
    names = load_tensors(out).drop_duplicates('tensor')
    pairs = [(row.tensor, mapping[row.layer]) for row in names.itertuples()
             if row.layer in mapping]
    if not pairs:
        raise ValueError(f'No matching imatrix tensors for {name}')
    target = out / 'overrides' / f'{name}.txt'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(''.join(f'{tensor}={qtype}\n' for tensor, qtype in sorted(pairs)), encoding='utf-8')
    return target


def plan(out: Path, model_dir: Path, high_fraction: float, 
         low_type: str, high_type: str, ablations: bool) -> pd.DataFrame:
    if not 0 <= high_fraction <= 1:
        raise ValueError('--high-fraction must be 0..1')
    out.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted(set(load_tensors(out).layer))
    records = []

    def record(name: str, method: str, domain: str, base: str, ov: Path | None,
               imatrix: Path | None, ablated_layer: int | None = None):
        records.append(dict(name=name, method=method, domain=domain,
                            base_type=base, override_file=str(ov) if ov else '',
                            imatrix_file=str(imatrix) if imatrix else '',
                            output_model=str(model_dir / f'{name}.gguf'),
                            ablated_layer=ablated_layer, status='planned',
                            size_mib=float('nan')))

    # Two uniform baselines plus domain-guided uniform variants.
    record('uniform-q4', 'uniform', '', 'Q4_K_M', None, None)
    record('uniform-q6', 'uniform', '', 'Q6_K', None, None)
    for domain in DOMAINS:
        imatrix = ROOT / 'models' / 'imatrix' / f'{domain}.imatrix.gguf'
        record(f'uniform-q4-{domain}', 'uniform-imatrix', domain,
               'Q4_K_M', None, imatrix)

        rank = load_rank(out, domain)
        k = round(high_fraction * len(rank))
        top = set(rank.head(k).layer.astype(int))
        mapping = {layer: high_type if layer in top else low_type for layer in layers}
        ov = overrides(out, f'mixed-{domain}', mapping)
        record(f'mixed-{domain}', 'mixed-imatrix', domain, 'Q4_K_M', ov, imatrix)

    # P1 experiment: perturb exactly one block from high to low precision,
    # holding all other selected transformer block weights at high precision.
    # This is a useful layer ablation, not a substitute for BF16-as-reference KL.
    if ablations:
        base = {layer: high_type for layer in layers}
        ov = overrides(out, 'p1-all-high', base)
        record('p1-all-high', 'ablation-control', '', 'Q6_K', ov, None)
        for layer in layers:
            mapping = dict(base)
            mapping[layer] = low_type
            ov = overrides(out, f'p1-layer-{layer:02d}', mapping)
            record(f'p1-layer-{layer:02d}', 'layer-ablation', '', 'Q6_K',
                   ov, None, layer)

    df = pd.DataFrame(records)
    df.to_csv(out / 'experiment_plan.csv', index=False)
    print(f'Planned {len(df)} GGUF variants; CSV: {out / "experiment_plan.csv"}')
    print(f'P1 perturbations: {len(layers) if ablations else 0}; P2 mixed variants: {len(DOMAINS)}')
    return df


def command_for(row, quantizer: Path, source: Path, dry_run: bool = False) -> list[str]:
    cmd = [str(quantizer)]
    if pd.notna(row.override_file) and str(row.override_file):
        cmd += ['--tensor-type-file', str(row.override_file)]
    if pd.notna(row.imatrix_file) and str(row.imatrix_file):
        cmd += ['--imatrix', str(row.imatrix_file)]
    if dry_run:
        cmd.append('--dry-run')
    cmd += [str(source), str(row.output_model), str(row.base_type)]
    return cmd


def build(out: Path, source: Path, quantizer: Path, only: str | None,
          dry_run: bool, overwrite: bool):
    path = out / 'experiment_plan.csv'
    if not path.exists():
        raise FileNotFoundError('Run plan first')
    if not source.is_file():
        raise FileNotFoundError(source)
    if not quantizer.is_file():
        raise FileNotFoundError(quantizer)
    df = pd.read_csv(path)
    names = set(df.name)
    if only and only not in names:
        raise ValueError(f'Unknown variant {only}; select from {sorted(names)}')
    for idx, row in df.iterrows():
        if only and row['name'] != only:
            continue
        target = Path(row.output_model)
        if target.is_file() and not overwrite and not dry_run:
            df.at[idx, 'status'] = 'existing'
            df.at[idx, 'size_mib'] = target.stat().st_size / (1024**2)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = command_for(row, quantizer, source, dry_run)
        print('Executing:', ' '.join(cmd))
        log = out / 'logs' / f'{row["name"]}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        # Keep partial success records when a model fails.
        try:
            with log.open('w', encoding='utf-8') as handle:
                subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)
            df.at[idx, 'status'] = 'dry-run' if dry_run else 'built'
            if not dry_run:
                if not target.is_file():
                    raise RuntimeError(f'Quantizer succeeded but no output: {target}')
                df.at[idx, 'size_mib'] = target.stat().st_size / (1024**2)
        except Exception:
            df.at[idx, 'status'] = 'failed'
            df.to_csv(path, index=False)
            print(f'Check quantizer log: {log}')
            raise
        df.to_csv(path, index=False)
    print('Updated', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'build'))
    parser.add_argument('--output-dir', type=Path, default=RESULTS)
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--quantizer', type=Path, default=BINARY)
    parser.add_argument('--low-type', default='Q4_K')
    parser.add_argument('--high-type', default='Q6_K')
    parser.add_argument('--high-fraction', type=float, default=0.25)
    parser.add_argument('--no-ablations', action='store_true')
    parser.add_argument('--only', help='Build just this experiment (e.g. mixed-chat)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.command == 'plan':
        plan(args.output_dir, args.model_dir, args.high_fraction,
             args.low_type, args.high_type, not args.no_ablations)
    else:
        build(args.output_dir, args.source, args.quantizer,
              args.only, args.dry_run, args.overwrite)

if __name__ == '__main__':
    main()
