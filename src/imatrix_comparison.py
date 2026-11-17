"""Controlled mixed Q4_K_M comparison: no imatrix versus code/math/chat imatrices."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from src.config import Config
from src.quantize import PURE_QUANT_TYPES

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['gsm8k', 'ifeval', 'hellaswag']
VARIANTS = ['control', 'code', 'math', 'chat']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')


def configurations(base, limit, kl_chunks, precision='q4'):
    caps = dict.fromkeys(TASKS, limit) if limit is not None else {'gsm8k': 1000, 'ifeval': None, 'hellaswag': 2000}
    return [base.with_overrides(name=f'mixed-{precision}-' + variant,
        model_file=base.models_dir / f'{precision}-imatrix-comparison' / (f'mixed-{precision}-' + variant + '.gguf'),
        tasks=TASKS.copy(), task_sample_caps=caps.copy(), num_examples=None,
        quantization_type=PURE_QUANT_TYPES[precision], pure_quantization=False, kl_enabled=True,
        kl_chunks=kl_chunks, num_fewshot=0, hf_repo=None, hf_filename=None)
        for variant in VARIANTS]


def quantize_command(config, matrix, output):
    command = [str(config.quantize_tool)]
    if matrix is not None:
        command += ['--imatrix', str(matrix)]
    return command + [str(config.source_model), str(output), config.quantization_type or 'Q4_K_M']


def load_references(base, inventory_path, gguf):
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    selected = [m for m in inventory['matrices'] if m['model'] == base.source_model.stem]
    if len(selected) != 3 or {m['task'] for m in selected} != {'code', 'math', 'chat'}:
        raise ValueError('Need exactly one provenanced BF16 imatrix per domain')
    source_hash = digest(base.source_model)
    matrices, coverage = {}, {}
    for item in selected:
        matrix = Path(item['path'])
        collection = item['collection']
        if collection['model_sha256'] != source_hash or digest(matrix) != item['sha256']:
            raise ValueError('BF16/imatrix hash mismatch: ' + item['task'])
        if digest(collection['calibration_path']) != collection['calibration_sha256']:
            raise ValueError('Calibration text changed: ' + item['task'])
        reader = gguf.GGUFReader(matrix)
        matrices[item['task']] = matrix
        coverage[item['task']] = {
            'matrix_sha256': item['sha256'], 'calibration_sha256': collection['calibration_sha256'],
            'chunks': int(reader.fields['imatrix.chunk_count'].contents()),
            'chunk_size': int(reader.fields['imatrix.chunk_size'].contents()),
        }
    return matrices, coverage, source_hash


def prepare_models(configs, matrices, source_hash, gguf):
    manifests = {}
    for variant, config in zip(VARIANTS, configs):
        model = config.resolve_model_file()
        model.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = model.with_suffix('.quantization.json')
        matrix = matrices.get(variant)
        identity = {'source_sha256': source_hash, 'tool_sha256': digest(config.quantize_tool),
                    'quantization_type': config.quantization_type, 'pure': False,
                    'imatrix_sha256': digest(matrix) if matrix else None}
        if model.exists():
            if not manifest_path.exists():
                raise ValueError(f'Existing model has no manifest; refusing to overwrite: {model}')
            saved = json.loads(manifest_path.read_text())
            if saved['inputs'] != identity or saved['model_sha256'] != digest(model):
                raise ValueError(f'Model provenance mismatch; choose a separate model directory: {model}')
            print(f'Reuse verified {model.name}', flush=True)
        else:
            partial = model.with_suffix('.partial.gguf')
            command = quantize_command(config, matrix, partial)
            print(f'Quantizing {variant}: {config.quantization_type} without --pure, imatrix={matrix}', flush=True)
            started = time.monotonic()
            with model.with_suffix('.quantization.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            partial.replace(model)
            saved = {'inputs': identity, 'model_sha256': digest(model), 'command': command,
                     'size_bytes': model.stat().st_size, 'quantization_seconds': time.monotonic() - started}
            write_json(manifest_path, saved)
        reader = gguf.GGUFReader(model)
        saved['tensor_types'] = dict(Counter(t.tensor_type.name for t in reader.tensors))
        saved['tensor_layout'] = {t.name: t.tensor_type.name for t in reader.tensors}
        if config.quantization_type not in ('Q6_K', 'Q8_0') and len({v for k, v in saved['tensor_layout'].items() if k.startswith('blk.') and k.endswith('.weight') and v != 'F32'}) < 2:
            raise ValueError('Expected a mixed-precision layout')
        write_json(manifest_path, saved)
        manifests[variant] = saved
    control = manifests['control']['tensor_layout']
    for variant in VARIANTS[1:]:
        if manifests[variant]['tensor_layout'] != control:
            raise ValueError(f'Tensor precisions differ between control and {variant}; cannot isolate calibration')
    return manifests


def run_evaluations(configs, manifests, status_filename='run_status.json'):
    from src.experiment import run_experiment
    # Avoid accidental downloads: this experiment uses the existing held-out text files.
    for config in configs:
        for domain in ['code', 'math', 'chat']:
            if not (config.project_root / 'data/kl' / (domain + '.txt')).is_file():
                raise FileNotFoundError('Missing held-out KL texts; prepare them before this comparison')
    summaries = []
    for variant, config in zip(VARIANTS, configs):
        config = config.with_overrides(kl_text_dir=config.project_root / 'data/kl')
        identity = {'config': json.loads(json.dumps(asdict(config), default=str)),
                    'model_sha256': manifests[variant]['model_sha256'],
                    'server_sha256': digest(config.llama_server), 'perplexity_sha256': digest(config.llama_perplexity),
                    'text_sha256': {p.name: digest(p) for p in config.kl_text_dir.glob('*.txt')},
                    'pipeline_sha256': {n: digest(ROOT / 'src' / n) for n in ['evaluate.py', 'gguf.py', 'server.py', 'kl.py']}}
        marker = config.run_dir / 'comparison_complete.json'
        started = time.monotonic()
        try:
            if marker.exists():
                saved = json.loads(marker.read_text())
                if saved['inputs'] != identity:
                    raise ValueError('Completed run settings changed; choose a new --experiment name')
                for name, expected in saved['outputs'].items():
                    if digest(config.run_dir / name) != expected:
                        raise ValueError('Completed result hash mismatch: ' + name)
                print('Reuse completed evaluation: ' + variant, flush=True)
            else:
                print('\nEvaluating ' + variant, flush=True)
                run_experiment(config)
                outputs = {n: digest(config.run_dir / n) for n in ['metrics.json', 'metadata.json', 'samples.json', 'kl.json']}
                saved = {'inputs': identity, 'outputs': outputs, 'evaluation_seconds': time.monotonic() - started}
                write_json(marker, saved)
            summaries.append({'variant': variant, 'status': 'ok', 'seconds': saved['evaluation_seconds']})
        except Exception as error:
            print(f'{variant} FAILED: {error}', flush=True)
            summaries.append({'variant': variant, 'status': 'failed', 'error': str(error)})
        write_json(config.results_dir / status_filename, summaries)
    return summaries


def summarize(configs, manifests, filename='summary.csv', successful_variants=None):
    import pandas as pd
    rows = []
    for variant, config in zip(VARIANTS, configs):
        if successful_variants is not None and variant not in successful_variants:
            continue
        metrics, kl = config.run_dir / 'metrics.json', config.run_dir / 'kl.json'
        if not (config.run_dir / 'comparison_complete.json').exists():
            continue
        m, k = json.loads(metrics.read_text()), json.loads(kl.read_text())
        row = {'variant': variant, 'size_mib': config.resolve_model_file().stat().st_size / 2**20,
               'gsm8k_pct': 100 * m['gsm8k']['exact_match,flexible-extract'],
               'ifeval_pct': 100 * m['ifeval']['prompt_level_strict_acc,none'],
               'hellaswag_acc_norm_pct': 100 * m['hellaswag']['acc_norm,none']}
        for domain, item in k['domains'].items():
            row['kl_' + domain] = item['mean']
            row['kl_' + domain + '_tokens'] = item['tokens']
        rows.append(row)
    if rows:
        frame = pd.DataFrame(rows)
        frame['mean_kl_nats'] = frame[['kl_chat', 'kl_code', 'kl_math']].mean(axis=1)
        control = frame.loc[frame.variant.eq('control'), 'mean_kl_nats']
        if len(control) == 1 and control.iloc[0] > 0:
            frame['mean_kl_reduction_vs_control_pct'] = 100 * (1 - frame.mean_kl_nats / control.iloc[0])
        frame.to_csv(configs[0].results_dir / filename, index=False)
        print('\n' + frame.to_string(index=False), flush=True)
    elif successful_variants is not None:
        (configs[0].results_dir / filename).unlink(missing_ok=True)


def analyze(configs, matrices, manifests, gguf, output_folder=None):
    import numpy as np
    import pandas as pd
    from src.important_imatrix_plots import column_squared_error, input_second_moment, weighted_error, tensor_key
    folder = output_folder or configs[0].results_dir
    folder.mkdir(parents=True, exist_ok=True)
    identity = {'model_sha256': {k: v['model_sha256'] for k, v in manifests.items()},
                'matrix_sha256': {k: digest(v) for k, v in matrices.items()},
                'source_sha256': digest(configs[0].source_model),
                'analysis_sha256': digest(Path(__file__)),
                'weight_error_code_sha256': digest(ROOT / 'src/important_imatrix_plots.py'),
                'dequantizer_sha256': digest(Path(gguf.__file__).parent / 'quants.py')}
    csv_path, marker = folder / 'local_errors.csv', folder / 'local_errors_provenance.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['inputs'] == identity and csv_path.is_file() and saved['csv_sha256'] == digest(csv_path):
            print('Reuse verified local errors', flush=True)
            return
    vectors = {}
    for domain, path in matrices.items():
        reader = gguf.GGUFReader(path)
        tensors = {t.name: t for t in reader.tensors}
        vectors[domain] = {n.removesuffix('.in_sum2'): input_second_moment(t.data, tensors[n.removesuffix('.in_sum2') + '.counts'].data)
                           for n, t in tensors.items() if n.endswith('.in_sum2')}
    names = sorted(vectors['code'], key=tensor_key)
    if len(names) != 112 or any(set(v) != set(names) for v in vectors.values()):
        raise ValueError('Expected the same 112 projections in all BF16 imatrices')
    source_reader = gguf.GGUFReader(configs[0].source_model)
    sources = {t.name: t for t in source_reader.tensors}
    readers = {v: gguf.GGUFReader(c.resolve_model_file()) for v, c in zip(VARIANTS, configs)}
    tensors = {v: {t.name: t for t in reader.tensors} for v, reader in readers.items()}
    rows = []
    for number, name in enumerate(names, 1):
        original = sources[name]
        column_energy = np.zeros(int(original.shape[0]), dtype=np.float64)
        for start in range(0, int(original.shape[1]), 256):
            values = gguf.dequantize(original.data[start:start + 256], original.tensor_type).astype(np.float64)
            column_energy += np.einsum('ij,ij->j', values, values)
        for variant in VARIANTS:
            error = column_squared_error(original, tensors[variant][name], gguf.dequantize, 256)
            for domain in matrices:
                reference = weighted_error(vectors[domain][name], column_energy)
                value = weighted_error(vectors[domain][name], error)
                rows.append({'variant': variant, 'domain': domain, 'tensor': name,
                             'layer': tensor_key(name)[0], 'tensor_type': name.split('.')[2],
                             'stored_type': tensors[variant][name].tensor_type.name,
                             'weighted_error': value, 'reference_energy': reference,
                             'relative_error_pct': 100 * value / reference})
        if number % 7 == 0:
            print(f'Local errors: {number}/112 projections', flush=True)
    frame = pd.DataFrame(rows)
    control = frame[frame.variant.eq('control')][['domain', 'tensor', 'weighted_error']].rename(columns={'weighted_error': 'control_error'})
    frame = frame.merge(control, on=['domain', 'tensor'], validate='many_to_one')
    frame['error_reduction_vs_control_pct'] = 100 * (1 - frame.weighted_error / frame.control_error)
    frame.to_csv(csv_path, index=False)
    write_json(marker, {'inputs': identity, 'csv_sha256': digest(csv_path)})
    summary = frame.groupby(['variant', 'domain'])[['weighted_error', 'reference_energy', 'control_error']].sum()
    summary['relative_error_pct'] = 100 * summary.weighted_error / summary.reference_energy
    summary['error_reduction_vs_control_pct'] = 100 * (1 - summary.weighted_error / summary.control_error)
    summary.to_csv(folder / 'local_error_summary.csv')
    print('\nLocal reconstruction comparison:\n' + summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    size = parser.add_mutually_exclusive_group()
    size.add_argument('--limit', type=int, default=5, help='Matched questions per benchmark (default: 5, smoke test only)')
    size.add_argument('--full', action='store_true', help='Explicitly use the original 1000/all/2000 task counts')
    parser.add_argument('--kl-chunks', type=int, default=2)
    parser.add_argument('--stage', choices=['all', 'quantize', 'evaluate', 'analyze'], default='all')
    parser.add_argument('--experiment', help='New result folder name when changing protocol')
    parser.add_argument('--gguf-python', type=Path, default=Path(r'C:\llama.cpp\gguf-py'))
    args = parser.parse_args()
    if args.limit < 1 or args.kl_chunks < 1:
        parser.error('Limits and chunk counts must be positive')
    if args.gguf_python.is_dir():
        sys.path.insert(0, str(args.gguf_python))
    import gguf
    limit = None if args.full else args.limit
    scope = args.experiment or f'Q4_Imatrix_Comparison_{"full" if args.full else "smoke" + str(limit)}_kl{args.kl_chunks}'
    base = Config(experiment=scope)
    configs = configurations(base, limit, args.kl_chunks)
    base.results_dir.mkdir(parents=True, exist_ok=True)
    matrices, coverage, source_hash = load_references(base, ROOT / 'results/imatrix_full_pure_sweep/inventory.json', gguf)
    write_json(base.results_dir / 'comparison_plan.json', {'recipe': 'Q4_K_M', 'pure': False,
        'variants': VARIANTS, 'task_caps': configs[0].task_sample_caps, 'kl_chunks': args.kl_chunks,
        'smoke_test': not args.full, 'source_sha256': source_hash, 'calibration_coverage': coverage,
        'limitation': 'Existing domain imatrices use unequal calibration token budgets; smoke results do not establish domain superiority.'})
    manifests = prepare_models(configs, matrices, source_hash, gguf)
    write_json(base.results_dir / 'quantization_manifest.json', manifests)
    statuses = []
    if args.stage in ('all', 'evaluate'):
        statuses = run_evaluations(configs, manifests)
        summarize(configs, manifests)
    if args.stage in ('all', 'analyze'):
        analyze(configs, matrices, manifests, gguf)
    if any(s['status'] != 'ok' for s in statuses):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
