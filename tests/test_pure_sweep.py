import pytest

import run_evaluation as sweep
from src.config import Config


def test_limit_keeps_models_and_tasks(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_evaluation.py", "--limit", "10"])
    monkeypatch.setattr(sweep, "check_models", lambda: None)
    runs = []
    monkeypatch.setattr(sweep, "run_experiments", lambda configs: runs.extend(configs) or [])
    sweep.main()
    assert len(runs) == len(sweep.CONFIGS)
    for config in runs:
        assert config.tasks == ["gsm8k", "ifeval", "hellaswag"]
        assert config.task_sample_caps == {"gsm8k": 10, "ifeval": 10, "hellaswag": 10}
        assert config.name.endswith("-limit10")
        assert config.kl_enabled is False
    assert sweep.BASE.task_sample_caps == {"gsm8k": 1000, "ifeval": None, "hellaswag": 2000}


def test_no_limit_runs_the_configured_experiments(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_evaluation.py"])
    monkeypatch.setattr(sweep, "check_models", lambda: None)
    runs = []
    monkeypatch.setattr(sweep, "run_experiments", lambda configs: runs.extend(configs) or [])
    sweep.main()
    assert runs == sweep.CONFIGS
    assert all(config.kl_enabled for config in runs)


def test_missing_quantized_model_gives_manual_instructions(monkeypatch, tmp_path):
    source = tmp_path / "BF16.gguf"
    source.touch()
    base = Config(source_model=source)
    monkeypatch.setattr(sweep, "BASE", base)
    monkeypatch.setattr(sweep, "CONFIGS", [base.with_overrides(model_file=tmp_path / "pure-q4.gguf")])
    with pytest.raises(SystemExit, match="python src/quantize.py"):
        sweep.check_models()


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_missing_source_asks_before_downloading(monkeypatch, tmp_path, answer):
    source = tmp_path / "BF16.gguf"
    base = Config(source_model=source, hf_repo="class/model", hf_filename="BF16.gguf")
    monkeypatch.setattr(sweep, "BASE", base)
    monkeypatch.setattr(sweep, "CONFIGS", [base.with_overrides(model_file=source)])
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    downloads = []

    def download(config):
        downloads.append(config)
        source.write_bytes(b"source")
        return source

    monkeypatch.setattr(sweep, "get_model_file", download)
    if answer == "yes":
        sweep.check_models()
        assert len(downloads) == 1
        assert source.is_file()
    else:
        with pytest.raises(SystemExit):
            sweep.check_models()
        assert not downloads
