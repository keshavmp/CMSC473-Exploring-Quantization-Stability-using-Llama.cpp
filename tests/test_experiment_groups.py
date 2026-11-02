import json

import pandas as pd
import pytest

import run_evaluation as sweep
from src import report
from src.config import Config


def write_run(root, experiment, name, correct):
    cfg = Config(project_root=root, experiment=experiment, name=name)
    cfg.run_dir.mkdir(parents=True)
    cfg.results_file.write_text(json.dumps({"ifeval": {"prompt_level_strict_acc,none": float(correct)}}))
    cfg.samples_file.write_text(json.dumps({"ifeval": [{
        "doc_hash": "doc", "prompt_hash": "prompt", "target_hash": "target",
        "filter": "none", "prompt_level_strict_acc": correct,
    }]}))
    return cfg


def test_experiment_paths_keep_model_paths_unchanged(tmp_path):
    original = Config(project_root=tmp_path, name="q4")
    grouped = original.with_overrides(experiment="my-experiment")
    assert grouped.run_dir == tmp_path / "results" / "my-experiment" / "q4"
    assert grouped.log_file.parent == grouped.samples_file.parent == grouped.results_file.parent == grouped.run_dir
    assert grouped.resolve_model_file() == original.resolve_model_file()
    assert original.results_dir == tmp_path / "results"


@pytest.mark.parametrize("name", ["", " ", ".", "..", "../outside", "nested/run", "nested\\run", "C:drive", "bad*name", "trailing."])
def test_invalid_experiment_names(name):
    with pytest.raises(ValueError, match="single folder name"):
        Config(experiment=name)


def test_evaluation_cli_groups_limited_sweep(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["run_evaluation.py", "--experiment", "smoke", "--limit", "10"])
    monkeypatch.setattr(sweep, "check_models", lambda: None)
    original = [config.with_overrides(project_root=tmp_path) for config in sweep.CONFIGS]
    monkeypatch.setattr(sweep, "CONFIGS", original)
    runs = []
    monkeypatch.setattr(sweep, "run_experiments", lambda configs: runs.extend(configs) or [])
    sweep.main()
    assert len(runs) == len(original)
    for config, before in zip(runs, original):
        assert config.experiment == "smoke"
        assert config.run_dir == tmp_path / "results" / "smoke" / config.name
        assert config.name.endswith("-limit10")
        assert config.resolve_model_file() == before.resolve_model_file()
        assert config.kl_enabled is False
    assert all(config.experiment is None for config in original)


def test_single_experiment_report_stays_in_its_folder(monkeypatch, tmp_path):
    write_run(tmp_path, "selected", "BF16-pure-sweep", True)
    run = write_run(tmp_path, "selected", "q4", False)
    write_run(tmp_path, "other", "q4", True)
    monkeypatch.setattr(report, "Config", lambda: Config(project_root=tmp_path))
    monkeypatch.setattr("sys.argv", ["report.py", "--experiment", "selected", "q4"])
    report.main()
    table = pd.read_csv(run.results_dir / "summary.csv", index_col=0)
    assert list(table.index) == ["q4"]
    assert table.loc["q4", "ifeval"] == 0.0
    stability = pd.read_csv(run.results_dir / "stability.csv")
    assert stability.loc[0, "correct_to_incorrect"] == 1
    assert (run.results_dir / "summary.png").is_file()
    assert not (tmp_path / "results" / "summary.csv").exists()
    assert not (tmp_path / "results" / "other" / "summary.csv").exists()


def test_comparison_preserves_duplicate_run_names_and_local_baselines(monkeypatch, tmp_path):
    write_run(tmp_path, "first", "BF16-pure-sweep", True)
    write_run(tmp_path, "first", "q4", False)
    write_run(tmp_path, "second", "BF16-pure-sweep", False)
    write_run(tmp_path, "second", "q4", True)
    output = tmp_path / "comparison"
    monkeypatch.setattr(report, "Config", lambda: Config(project_root=tmp_path))
    monkeypatch.setattr("sys.argv", ["report.py", "--experiment", "first", "--experiment", "second",
                                    "--output-dir", str(output), "q4"])
    report.main()
    table = pd.read_csv(output / "summary.csv", index_col=0)
    assert list(table.index) == ["first/q4", "second/q4"]
    assert list(table["ifeval"]) == [0.0, 100.0]
    stability = pd.read_csv(output / "stability.csv").set_index("run")
    assert stability.loc["first/q4", "correct_to_incorrect"] == 1
    assert stability.loc["second/q4", "incorrect_to_correct"] == 1


@pytest.mark.parametrize("arguments", [
    ["--experiment", "unknown"],
    ["--experiment", "../outside"],
    ["--experiment", "first", "--experiment", "second"],
    ["--experiment", "first", "--baseline", "missing"],
    ["--experiment", "first", "missing-run"],
])
def test_report_rejects_invalid_selection_before_writing(monkeypatch, tmp_path, arguments):
    write_run(tmp_path, "first", "q4", True)
    monkeypatch.setattr(report, "Config", lambda: Config(project_root=tmp_path))
    monkeypatch.setattr("sys.argv", ["report.py", *arguments])
    with pytest.raises(SystemExit) as error:
        report.main()
    assert error.value.code == 2
    assert not list(tmp_path.rglob("summary.csv"))


def test_nested_experiment_finds_model_size_without_snapshot(tmp_path):
    run = write_run(tmp_path, "experiment", "q4", True)
    run.models_dir.mkdir()
    run.resolve_model_file().write_bytes(bytes(1024))
    assert report.model_size_mib(run.results_file) == 0.0  # rounded MiB


def test_limited_baseline_and_missing_auto_baseline(tmp_path):
    run = write_run(tmp_path, "smoke", "BF16-pure-sweep-limit10", True)
    assert report.select_baseline([run.results_file], None, "source") == run.results_file
    assert report.select_baseline([], None, "source") is None
