import json

import pandas as pd
import pytest

import rohan.quantize_family as quantize
import rohan.report_families as report
import rohan.run_family_sweep as sweep


@pytest.mark.parametrize("family", list(sweep.FAMILIES))
def test_family_configs_stay_under_rohan(family):
    configs = sweep.family_configs(family)
    assert [config.name for config in configs] == [
        "BF16-pure-sweep", "pure-q2-sweep", "pure-q3-sweep", "pure-q4-sweep",
        "pure-q5-sweep", "pure-q6-sweep", "pure-q8-sweep",
    ]
    for config in configs:
        assert config.run_dir.is_relative_to(sweep.ROHAN_DIR / "results")
        assert config.model_file.is_relative_to(sweep.ROHAN_DIR / "models" / family)
        assert config.source_model == configs[0].model_file
        assert config.task_sample_caps == {"gsm8k": 100, "ifeval": 50, "hellaswag": 200}
        assert config.gpu_layers == "0"
        assert config.kl_enabled
    assert all(config.pure_quantization for config in configs[1:])
    assert configs[0].hf_repo == sweep.FAMILIES[family]["hf_repo"]
    model = sweep.FAMILIES[family]["model"]
    assert [config.model_file.name for config in configs[1:]] == [
        f"{model}-pure-{recipe}.gguf" for recipe in sweep.PURE_QUANT_TYPES.values()
    ]


def run_main(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["run_family_sweep", *argv])
    monkeypatch.setattr(sweep, "check_models", lambda families: None)
    runs = []
    monkeypatch.setattr(sweep, "run_experiments", lambda configs: runs.extend(configs) or [])
    sweep.main()
    return runs


def test_families_run_in_requested_order(monkeypatch):
    runs = run_main(monkeypatch, "--family", "gemma", "qwen")
    assert len(runs) == 14
    assert {config.experiment for config in runs[:7]} == {"Gemma_Pure_Sweep"}
    assert {config.experiment for config in runs[7:]} == {"Qwen_Pure_Sweep"}


def test_limit_uses_separate_folders_and_skips_kl(monkeypatch):
    runs = run_main(monkeypatch, "--family", "qwen", "--limit", "3")
    for config in runs:
        assert config.name.endswith("-limit3")
        assert config.experiment == "Qwen_Pure_Sweep-limit3"
        assert config.task_sample_caps == {"gsm8k": 3, "ifeval": 3, "hellaswag": 3}
        assert config.kl_enabled is False


def test_resume_skips_finished_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "BASE", sweep.BASE.with_overrides(project_root=tmp_path))
    finished, partial = sweep.family_configs("qwen")[:2]
    for config in (finished, partial):
        config.run_dir.mkdir(parents=True)
        (config.run_dir / "metrics.json").write_text("{}")
    (finished.run_dir / "kl.json").write_text("{}")
    runs = run_main(monkeypatch, "--family", "qwen", "--resume")
    assert [config.name for config in runs][:1] == ["pure-q2-sweep"]
    assert len(runs) == 6


def test_missing_variant_points_to_quantize_family(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "BASE", sweep.BASE.with_overrides(project_root=tmp_path))
    monkeypatch.setattr(sweep, "check_tools", lambda: None)
    source = sweep.family_configs("gemma")[0].model_file
    source.parent.mkdir(parents=True)
    source.touch()
    with pytest.raises(SystemExit, match="rohan.quantize_family --family gemma"):
        sweep.check_models(["gemma"])


def test_quantize_family_uses_source_and_skips_existing(monkeypatch, tmp_path):
    monkeypatch.setattr(quantize, "family_configs",
                        lambda family: [config.with_overrides(project_root=tmp_path)
                                        for config in sweep.family_configs(family)])
    monkeypatch.setattr(sweep, "BASE", sweep.BASE.with_overrides(project_root=tmp_path))
    source, *variants = sweep.family_configs("llama")
    source.model_file.parent.mkdir(parents=True)
    source.model_file.touch()
    variants[0].model_file.touch()
    calls = []
    monkeypatch.setattr(quantize, "quantize_model", lambda *args, **kwargs: calls.append((args, kwargs)))
    quantize.quantize_family("llama")
    assert len(calls) == 5
    for (tool, src, out, recipe), kwargs in calls:
        assert src == source.model_file
        assert kwargs == {"pure": True}
    assert [args[3] for args, _ in calls] == ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]


def test_family_rows_maps_levels_and_averages_kl():
    table = pd.DataFrame({
        "size_mib": [100.0, 30.0, 999.0],
        "gsm8k": [20.0, 10.0, 5.0],
        "ifeval": [40.0, 30.0, 1.0],
        "hellaswag": [45.0, 40.0, 1.0],
        "kl_chat_mean": [0.0, 0.2, 9.0],
        "kl_math_mean": [0.0, 0.4, 9.0],
        "kl_chat_p95": [0.0, 5.0, 9.0],
    }, index=["BF16-pure-sweep", "pure-q4-sweep", "BF16-pure-sweep-limit3"])
    rows = report.family_rows("qwen", table)
    assert rows["level"].tolist() == ["BF16", "Q4"]
    assert rows.loc[1, "kl_mean"] == pytest.approx(0.3)
    assert rows.loc[1, "gsm8k"] == 10.0


def test_report_family_writes_standard_outputs(tmp_path):
    run = tmp_path / "BF16-pure-sweep"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({"gsm8k": {"exact_match,flexible-extract": 0.2}}))
    table = report.report_family(tmp_path)
    assert table.loc["BF16-pure-sweep", "gsm8k"] == 20.0
    for name in ("summary.csv", "summary.png", "stability.csv"):
        assert (tmp_path / name).is_file()
    assert report.report_family(tmp_path / "empty") is None
