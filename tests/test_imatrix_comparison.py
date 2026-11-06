from src.config import Config
from src.imatrix_comparison import configurations, quantize_command


def test_comparison_changes_only_calibration_at_matched_mixed_precision(tmp_path):
    base = Config(project_root=tmp_path, experiment='smoke')
    configs = configurations(base, 5, 2)
    assert len(configs) == 4
    assert len({c.resolve_model_file() for c in configs}) == 4
    for config in configs:
        assert config.pure_quantization is False
        assert config.quantization_type == 'Q4_K_M'
        assert config.task_sample_caps == {'gsm8k': 5, 'ifeval': 5, 'hellaswag': 5}
        assert config.kl_enabled and config.kl_chunks == 2
        assert config.source_model == base.source_model
        assert config.experiment == 'smoke'
    assert base.task_sample_caps == {}


def test_imatrix_does_not_change_recipe_or_enable_pure(tmp_path):
    config = Config()
    output = tmp_path / 'q4.gguf'
    matrix = tmp_path / 'code.imatrix.gguf'
    control = quantize_command(config, None, output)
    calibrated = quantize_command(config, matrix, output)
    assert '--pure' not in control + calibrated
    assert '--imatrix' not in control
    assert calibrated[1:3] == ['--imatrix', str(matrix)]
    assert control[-3:] == calibrated[-3:] == [str(config.source_model), str(output), 'Q4_K_M']


def test_full_counts_require_explicit_selection():
    configs = configurations(Config(), None, 4)
    assert all(c.task_sample_caps == {'gsm8k': 1000, 'ifeval': None, 'hellaswag': 2000} for c in configs)
    assert all(c.kl_chunks == 4 for c in configs)
