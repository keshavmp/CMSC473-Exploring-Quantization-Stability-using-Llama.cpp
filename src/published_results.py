"""Export concise findings tables from local outputs, or verify a published bundle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / 'results/important plots'
OUT = ROOT / 'results/published'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def column_normalize(frame, column, value):
    frame = frame.copy()
    frame['column_mean_error_pct'] = frame.groupby(column)[value].transform('mean')
    frame['deviation_from_column_mean_pct'] = 100 * (frame[value] / frame.column_mean_error_pct - 1)
    return frame


def export():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {
        'precision_summary.csv': PLOTS / 'normalized comparison/precision_summary.csv',
        'dataset_summary.csv': PLOTS / 'normalized comparison/dataset_summary.csv',
        'block_summary.csv': PLOTS / 'by precision/by block/precision_weighted_errors.csv',
        'tensor_type_summary.csv': PLOTS / 'normalized comparison/tensor_type_summary.csv',
        'block_dataset_summary.csv': PLOTS / 'top graphs/05_blocks_by_dataset.csv',
        'tensor_type_dataset_summary.csv': PLOTS / 'top graphs/06_tensors_by_dataset.csv',
    }
    for name, path in inputs.items():
        frame = pd.read_csv(path)
        if name in ('block_summary.csv', 'tensor_type_summary.csv'):
            frame = column_normalize(frame, 'precision', 'relative_error_pct')
        if name.endswith('_dataset_summary.csv'):
            frame = frame.rename(columns={'column_deviation_pct': 'deviation_from_column_mean_pct',
                                          'dataset_column_mean_error_pct': 'column_mean_error_pct'})
        frame.to_csv(OUT / name, index=False)
    weights = json.loads((PLOTS / 'weight_error_provenance.json').read_text())['inputs']
    inventory = json.loads((ROOT / 'results/imatrix_full_pure_sweep/inventory.json').read_text())
    collection_fields = ['model_sha256', 'calibration_sha256', 'chunks', 'context', 'gpu_layers',
                         'parse_special', 'batch_size', 'tool_version', 'tool_sha256', 'matrix_sha256']
    collections = [{
        'model': m['model'], 'domain': m['task'],
        **{k: m['collection'][k] for k in collection_fields},
    } for m in inventory['matrices']]
    evaluations = []
    for p in ['BF16', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8']:
        run = ROOT / 'results/Pure_Sweep' / ('BF16-pure-sweep' if p == 'BF16' else f'pure-{p.lower()}-sweep')
        c = json.loads((run / 'config.json').read_text())
        evaluations.append({'precision': p, **{k: c[k] for k in ['context_size', 'request_workers',
            'batch_size', 'num_fewshot', 'quantization_type', 'pure_quantization', 'model_size_bytes']},
            'metrics_sha256': sha256(run / 'metrics.json'), 'kl_sha256': sha256(run / 'kl.json')})
    manifest = {
        'model': 'Llama-3.2-1B-Instruct', 'baseline': 'BF16',
        'reference_sha256': weights['reference']['sha256'],
        'quantized_models': [{'name': m['name'], 'sha256': m['sha256']} for m in weights['models']],
        'calibration': {d: {k: v[k] for k in ['sha256', 'calibration_sha256', 'chunks', 'chunk_size']}
                        for d, v in weights['imatrices'].items()},
        'collections': collections, 'evaluations': evaluations,
        'methods': {
            'relative_error_pct': '100 * weighted_error / BF16 diagonal reference_energy',
            'column_deviation_pct': '100 * (relative_error_pct / arithmetic mean over displayed column rows - 1)',
            'precision_tables': 'equal-domain average of tensor errors and references, then ratio of sums within unit',
            'dataset_tables': 'arithmetic mean of unit relative errors over Q2,Q3,Q4,Q5,Q6,Q8; BF16 excluded',
            'dataset_normalization': 'normalize after averaging precisions; compare rows within each dataset column',
            'hellaswag': 'Both acc and acc_norm retained; figures use acc_norm',
            'kl': 'KL(BF16 || quantized), nats; 510 scored tokens per domain',
        },
        'local_input_sha256': {name: sha256(path) for name, path in inputs.items()},
        'tables': {name: {'rows': len(pd.read_csv(OUT / name)), 'sha256': sha256(OUT / name)} for name in inputs},
        'figures': {p.name: sha256(p) for p in sorted((PLOTS / 'top graphs').glob('*.png'))},
    }
    (OUT / 'experiment_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def verify():
    manifest = json.loads((OUT / 'experiment_manifest.json').read_text())
    expected = {'precision_summary.csv': 7, 'dataset_summary.csv': 18, 'block_summary.csv': 96,
                'tensor_type_summary.csv': 42, 'block_dataset_summary.csv': 48, 'tensor_type_dataset_summary.csv': 21}
    for name, n in expected.items():
        path = OUT / name
        if sha256(path) != manifest['tables'][name]['sha256']:
            raise ValueError(f'Hash mismatch: {name}')
        frame = pd.read_csv(path)
        if len(frame) != n:
            raise ValueError(f'Unexpected row count: {name}')
        if {'weighted_error', 'reference_energy', 'relative_error_pct'} <= set(frame):
            if not frame.reference_energy.gt(0).all() or not np.allclose(frame.relative_error_pct, 100 * frame.weighted_error / frame.reference_energy):
                raise ValueError(f'Invalid relative error: {name}')
        if 'deviation_from_column_mean_pct' in frame:
            key = 'task' if name.endswith('_dataset_summary.csv') else 'precision'
            if not np.allclose(frame.column_mean_error_pct, frame.groupby(key).relative_error_pct.transform('mean')):
                raise ValueError(f'Invalid column mean: {name}')
            if not np.allclose(frame.deviation_from_column_mean_pct, 100 * (frame.relative_error_pct / frame.column_mean_error_pct - 1)):
                raise ValueError(f'Invalid normalization: {name}')
        if 'n_precisions' in frame and not frame.n_precisions.eq(6).all():
            raise ValueError(f'Invalid precision coverage: {name}')
    for name, digest in manifest['figures'].items():
        if sha256(PLOTS / 'top graphs' / name) != digest:
            raise ValueError(f'Figure hash mismatch: {name}')
    print(f'Verified {sum(expected.values())} rows in six tables and {len(manifest["figures"])} PNG figures.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', action='store_true', help='Export from the full local analysis output tree')
    args = parser.parse_args()
    if args.export:
        export()
    verify()
