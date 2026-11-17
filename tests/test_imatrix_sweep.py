import pandas as pd
import pytest

from src.config import Config
from src.imatrix_comparison import configurations, quantize_command
from src.imatrix_sweep import make_plan, update_summary
from src.quantize import PURE_QUANT_TYPES


def test_all_precisions_have_matched_controls_unique_models_and_full_counts(tmp_path):
    configs, plan = make_plan(Config(project_root=tmp_path, experiment='full'), list(PURE_QUANT_TYPES), None, 4)
    assert plan['n_models'] == 24
    paths = [c.resolve_model_file() for group in configs.values() for c in group]
    assert len(set(paths)) == 24
    for precision, group in configs.items():
        assert len(group) == 4
        assert all(c.quantization_type == PURE_QUANT_TYPES[precision] and not c.pure_quantization for c in group)
        assert all(c.task_sample_caps == {'gsm8k': 1000, 'ifeval': None, 'hellaswag': 2000} for c in group)
        assert all(c.kl_chunks == 4 and c.kl_enabled for c in group)
        assert quantize_command(group[1], tmp_path / 'imatrix.gguf', paths[0])[-1] == PURE_QUANT_TYPES[precision]


def test_q4_keeps_existing_model_paths(tmp_path):
    old = configurations(Config(project_root=tmp_path), 5, 2)
    new = configurations(Config(project_root=tmp_path), None, 4, 'q4')
    assert [c.resolve_model_file() for c in old] == [c.resolve_model_file() for c in new]


def test_parallel_partitions_use_separate_ports_models_and_result_paths(tmp_path):
    left, left_plan = make_plan(Config(project_root=tmp_path, experiment='low', port=8080), ['q2', 'q3', 'q4'], None, 4)
    right, right_plan = make_plan(Config(project_root=tmp_path, experiment='high', port=8081), ['q5', 'q6', 'q8'], None, 4)
    left_configs = [c for group in left.values() for c in group]
    right_configs = [c for group in right.values() for c in group]
    assert all(c.port == 8080 for c in left_configs)
    assert all(c.port == 8081 for c in right_configs)
    assert {c.resolve_model_file() for c in left_configs}.isdisjoint({c.resolve_model_file() for c in right_configs})
    assert {c.run_dir for c in left_configs}.isdisjoint({c.run_dir for c in right_configs})
    assert 'server_port' not in left_plan
    assert right_plan['server_port'] == 8081


def test_recovery_uses_matched_control_and_handles_no_bf16_gap(tmp_path):
    pd.DataFrame([
        dict(variant='control', gsm8k_pct=10, ifeval_pct=50, hellaswag_acc_norm_pct=60, mean_kl_nats=.2),
        dict(variant='code', gsm8k_pct=15, ifeval_pct=55, hellaswag_acc_norm_pct=65, mean_kl_nats=.1),
    ]).to_csv(tmp_path / 'summary_q4.csv', index=False)
    baseline = {'scores_pct': {'gsm8k_pct': 20, 'ifeval_pct': 50, 'hellaswag_acc_norm_pct': 55}}
    update_summary(tmp_path, ['q4'], baseline)
    result = pd.read_csv(tmp_path / 'summary.csv').set_index('variant').loc['code']
    assert result.gsm8k_pct_gain_pp_vs_control == 5
    assert result.gsm8k_pct_bf16_loss_recovered_pct == 50
    assert pd.isna(result.ifeval_pct_bf16_loss_recovered_pct)
    assert pd.isna(result.hellaswag_acc_norm_pct_bf16_loss_recovered_pct)
    assert result.mean_kl_reduction_vs_control_pct == 50
