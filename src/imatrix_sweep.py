"""Matched no-imatrix/code/math/chat sweep across all available project precisions."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import socket
import sys
import time

from src.config import Config
from src.imatrix_comparison import (
    VARIANTS, configurations, load_references, prepare_models, run_evaluations,
    summarize, analyze, write_json, digest,
)
from src.quantize import PURE_QUANT_TYPES

ROOT = Path(__file__).resolve().parents[1]


def make_plan(base, precisions, limit, chunks):
    configs = {p: configurations(base, limit, chunks, p) for p in precisions}
    plan = {'experiment': base.experiment, 'precisions': precisions,
        'variants': VARIANTS, 'n_models': 4 * len(precisions), 'pure': False,
        'task_caps': next(iter(configs.values()))[0].task_sample_caps,
        'kl_chunks': chunks, 'kl_context_size': 512, 'kl_scored_tokens_per_domain': chunks * 255,
        'recipes': {p: PURE_QUANT_TYPES[p] for p in precisions},
        'source_model': str(base.source_model),
        'imatrix_source': 'Verified BF16 code/math/chat imatrices from existing full-pure-sweep inventory',
        'limitations': [
            'Existing imatrices have unequal calibration lengths; do not attribute domain differences solely to content.',
            'Q8_0 may ignore importance weights; it is included as a negative control, not an assumed improvement.',
            'Identical recipes must have identical tensor-type layouts across calibration variants.',
            'Complete model evaluations are reused. An interrupted model evaluation restarts that model.',
            'Full uses the established 1000 GSM8K / all 541 IFEval / 2000 HellaSwag protocol, not every available dataset example.',
        ]}
    # Preserve compatibility with existing default-port sweep plans.
    if base.port != 8080:
        plan['server_port'] = base.port
    return configs, plan


def matched_baseline(base, source_hash, limit):
    """Only reuse published BF16 scores when their recorded protocol matches this sweep."""
    if limit is not None:
        return None
    folder = ROOT / 'results/Pure_Sweep/BF16-pure-sweep'
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    config = metadata['project_configuration']
    expected = {'context_size': base.context_size, 'request_workers': base.request_workers,
                'batch_size': base.batch_size, 'num_fewshot': 0}
    if metadata.get('model_sha256') != source_hash or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Existing BF16 reference does not match model/settings; do not compute recovered-loss percentages from it')
    if metadata.get('lm_eval_version') != version('lm_eval'):
        raise ValueError('Existing BF16 reference uses a different evaluation harness version')
    wanted = {'gsm8k': 1000, 'ifeval': 541, 'hellaswag': 2000}
    if metadata.get('actual_unique_samples') != wanted or metadata.get('selected_doc_ids') != {t: list(range(n)) for t, n in wanted.items()}:
        raise ValueError('Existing BF16 reference has different sample counts or selected IDs')
    metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    return {'source_metrics': str(folder / 'metrics.json'), 'sha256': digest(folder / 'metrics.json'),
        'scores_pct': {'gsm8k_pct': 100 * metrics['gsm8k']['exact_match,flexible-extract'],
                       'ifeval_pct': 100 * metrics['ifeval']['prompt_level_strict_acc,none'],
                       'hellaswag_acc_norm_pct': 100 * metrics['hellaswag']['acc_norm,none']},
        'selected_doc_ids_sha256': __import__('hashlib').sha256(json.dumps(metadata['selected_doc_ids'], sort_keys=True).encode()).hexdigest(),
        'limitation': 'Historical BF16 scores match recorded model, sample IDs, harness version and key settings; cross-run numerical variation remains possible.'}


def update_summary(folder, precisions, baseline):
    import pandas as pd
    frames = []
    for p in precisions:
        path = folder / f'summary_{p}.csv'
        if path.exists():
            f = pd.read_csv(path)
            f.insert(0, 'precision', p.upper())
            frames.append(f)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    for column in ['gsm8k_pct', 'ifeval_pct', 'hellaswag_acc_norm_pct']:
        controls = combined[combined.variant.eq('control')].set_index('precision')[column]
        control_score = combined.precision.map(controls)
        combined[column + '_gain_pp_vs_control'] = combined[column] - control_score
        if baseline is not None:
            gap = baseline['scores_pct'][column] - control_score
            combined[column + '_bf16_loss_recovered_pct'] = 100 * (combined[column] - control_score) / gap.where(gap > 0)
    controls = combined[combined.variant.eq('control')].set_index('precision').mean_kl_nats
    mean_control = combined.precision.map(controls)
    combined['mean_kl_reduction_vs_control_pct'] = 100 * (1 - combined.mean_kl_nats / mean_control.where(mean_control > 0))
    combined.to_csv(folder / 'summary.csv', index=False)
    (folder / 'summary_notes.txt').write_text(
        'Recovery is measured against the no-imatrix control at the SAME precision and recipe.\n'
        'Accuracy gain is a percentage-point difference. BF16 loss recovered (%) = 100*(imatrix-control)/(BF16-control).\n'
        'Recovered loss is undefined when the BF16-control gap is zero or negative; values can be negative or exceed 100%.\n'
        'Small BF16 gaps produce unstable recovery ratios; interpret raw scores and gains first.\n'
        'Mean KL averages code/math/chat equally; lower is closer to BF16.\n'
        'Q8 is a negative control: supplying imatrix data may not change Q8_0 weights.\n'
        'Historical pure-quantization results are not the matched controls for this sweep.\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    size = parser.add_mutually_exclusive_group()
    size.add_argument('--full', action='store_true', help='1000 GSM8K, 541 IFEval, 2000 HellaSwag per model')
    size.add_argument('--limit', type=int, help='Smoke/pilot questions per benchmark')
    parser.add_argument('--plan', action='store_true', help='Show the plan without running or quantizing models')
    parser.add_argument('--precisions', nargs='+', choices=list(PURE_QUANT_TYPES), default=list(PURE_QUANT_TYPES), type=str.lower)
    parser.add_argument('--kl-chunks', type=int, default=4)
    parser.add_argument('--experiment', help='Separate result folder name')
    parser.add_argument('--port', type=int, default=8080, help='Server port; parallel terminals need distinct ports AND experiment folders')
    parser.add_argument('--stage', choices=['all', 'quantize', 'evaluate'], default='all')
    parser.add_argument('--local-errors', action='store_true', help='Also compute 112-projection reconstruction errors for each precision')
    parser.add_argument('--gguf-python', type=Path, default=Path(r'C:\llama.cpp\gguf-py'))
    args = parser.parse_args()
    if args.kl_chunks < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Limits/chunks must be positive')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if not args.full and args.limit is None and not args.plan:
        parser.error('Choose --full or --limit N explicitly')
    if len(set(args.precisions)) != len(args.precisions):
        parser.error('Do not repeat precisions')
    limit = None if args.full or args.limit is None else args.limit
    scope = args.experiment or f'Imatrix_Recovery_Sweep_{"full" if limit is None else "limit" + str(limit)}_kl{args.kl_chunks}'
    if args.experiment is None and args.port != 8080:
        scope += f'_port{args.port}'
    base = Config(experiment=scope, port=args.port)
    configs, plan = make_plan(base, args.precisions, limit, args.kl_chunks)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if args.gguf_python.is_dir():
        sys.path.insert(0, str(args.gguf_python))
    import gguf
    # Fail before expensive quantization if a different server already owns our port.
    if args.stage != 'quantize':
        with socket.socket() as connection:
            if connection.connect_ex((base.host, base.port)) == 0:
                raise RuntimeError(f'Port {base.port} is already occupied; stop that server before running this sweep')
    base.results_dir.mkdir(parents=True, exist_ok=True)
    matrices, coverage, source_hash = load_references(base, ROOT / 'results/imatrix_full_pure_sweep/inventory.json', gguf)
    baseline = matched_baseline(base, source_hash, limit)
    plan.update(source_sha256=source_hash, calibration_coverage=coverage, bf16_reference=baseline)
    plan_path = base.results_dir / 'sweep_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Existing sweep plan differs; choose a new --experiment NAME')
    write_json(plan_path, plan)
    states = {}
    for precision in args.precisions:
        group = configs[precision]
        started = time.monotonic()
        print(f'\n=== {precision.upper()}: control + code/math/chat imatrices ===', flush=True)
        try:
            manifests = prepare_models(group, matrices, source_hash, gguf)
            write_json(base.results_dir / f'quantization_manifest_{precision}.json', manifests)
            if args.stage != 'quantize':
                statuses = run_evaluations(group, manifests, status_filename=f'run_status_{precision}.json')
                summarize(group, manifests, filename=f'summary_{precision}.csv',
                          successful_variants={s['variant'] for s in statuses if s['status'] == 'ok'})
                update_summary(base.results_dir, args.precisions, baseline)
                states[precision] = {'status': 'ok' if all(s['status'] == 'ok' for s in statuses) else 'failed', 'models': statuses}
            else:
                states[precision] = {'status': 'quantized'}
            if args.local_errors:
                analyze(group, matrices, manifests, gguf, output_folder=base.results_dir / 'local_errors' / precision)
            states[precision]['seconds_this_invocation'] = time.monotonic() - started
        except Exception as error:
            states[precision] = {'status': 'failed', 'error': str(error)}
            print(f'{precision} FAILED: {error}', flush=True)
        write_json(base.results_dir / 'sweep_status.json', states)
    print('Sweep status: ' + str(base.results_dir / 'sweep_status.json'), flush=True)
    print('Recovery summary: ' + str(base.results_dir / 'summary.csv'), flush=True)
    if any(s['status'] == 'failed' for s in states.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
