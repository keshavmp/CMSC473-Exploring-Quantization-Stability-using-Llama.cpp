import json

import pytest

import rohan.run_long_context as lc


def all_runs():
    return [(part, run) for part, runs in lc.PARTS.items() for run in runs]


def test_run_names_are_unique_within_each_experiment_folder():
    for folder in {lc.experiment_for(part) for part in lc.PARTS}:
        names = [run.name for part, run in all_runs() if lc.experiment_for(part) == folder]
        assert len(names) == len(set(names)), folder


def test_night1_is_the_bf16_vs_q4_core_grid():
    runs = lc.PARTS["night1"]
    assert len(runs) == 2 * 3 * 2
    assert {run.level for run in runs} == {"BF16", "Q4"}
    assert {run.task for run in runs} == {"niah_multikey_2", "ruler_vt"}
    assert all(run.n == 20 for run in runs)
    # Length-major: every 4K run comes before any 8K run.
    lengths = [run.length for run in runs]
    assert lengths == sorted(lengths)


def test_every_level_has_a_model_and_runs_stay_under_rohan():
    models = lc.level_models()
    assert set(models) == {"BF16", "Q8", "Q6", "Q5", "Q4", "Q3", "Q2", "Q2_K-mixed"}
    for part, run in all_runs():
        config = lc.run_config(run, part)
        assert config.model_file == models[run.level]
        assert config.run_dir.is_relative_to(lc.ROHAN_DIR / "results")
        assert config.context_size == run.length + lc.GENERATION_RESERVE
        assert config.gpu_layers == "0"


def test_kl_windows_cover_the_same_tokens():
    kl_runs = [run for _, run in all_runs() if run.kind == "kl"]
    assert kl_runs
    for run in kl_runs:
        config = lc.run_config(run, "night4")
        assert config.kl_context_size * config.kl_chunks == lc.KL_TOKENS
        assert config.source_model == lc.level_models()["BF16"]


def test_kl_quant_levels_build_references_before_bf16():
    kl = [run for run in lc.PARTS["night4"] if run.kind == "kl"]
    assert kl[0].level == "Q2"
    assert kl[-1] == lc.Run("kl", "BF16", lc.K4)
    # 32K logits (n_ctx x 128K vocab x 4 bytes) do not fit in 16 GB of RAM.
    assert max(run.length for run in kl) <= lc.K16


def test_kv_cache_size_matches_llama_cpp_report():
    # llama.cpp printed 528.00 MiB for a 16896-token context.
    assert lc.kv_cache_mib(16896) == pytest.approx(528.0)


def test_timing_stats_summarizes_cache_hits_and_speed():
    timings = [
        {"prompt_n": 1000, "cache_n": 0, "prompt_ms": 5000, "predicted_n": 10, "predicted_ms": 500},
        {"prompt_n": 100, "cache_n": 900, "prompt_ms": 500, "predicted_n": 30, "predicted_ms": 1500},
    ]
    stats = lc.timing_stats(timings, questions=2, seconds=10)
    assert stats["mean_prompt_tokens"] == 1000
    assert stats["cache_hit_rate"] == pytest.approx(0.45)
    assert stats["prompt_tokens_per_second"] == pytest.approx(200)
    assert stats["generation_tokens_per_second"] == pytest.approx(20)
    assert stats["mean_time_to_first_token_s"] == pytest.approx(2.75)
    assert stats["seconds_per_question"] == 5


def test_timed_backend_keeps_server_timings(monkeypatch):
    responses = [{"timings": {"prompt_n": 5}}, {"tokens": [1, 2]}]
    monkeypatch.setattr(lc.GGUFLM, "_post", lambda self, endpoint, payload: responses.pop(0))
    with lc.TimedGGUFLM(base_url="http://test") as model:
        model._post("v1/completions", {})
        model._post("tokenize", {})
    assert model.timings == [{"prompt_n": 5}]


class FakeTokenizer:
    """One token per word."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


def test_padding_has_the_requested_length_and_prompt_ends_with_sweep_prompt(monkeypatch):
    monkeypatch.setattr(lc, "essay_text", lambda: " ".join(f"w{i}" for i in range(5000)))
    filler = lc.padding(300, FakeTokenizer())
    assert len(filler.split()) == 300
    prompt = lc.gsm8k_prompt(filler, "What is 2+2?")
    assert prompt.endswith("\n\nQuestion: What is 2+2?\nAnswer:")
    with pytest.raises(ValueError):
        lc.padding(10_000, FakeTokenizer())


def test_resume_uses_stats_file(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "BASE", lc.BASE.with_overrides(project_root=tmp_path))
    run = lc.PARTS["night1"][0]
    config = lc.run_config(run, "night1")
    assert not lc.is_done(config)
    config.run_dir.mkdir(parents=True)
    (config.run_dir / "metrics.json").write_text(json.dumps({}))
    assert not lc.is_done(config)  # metrics alone may be from a crashed run
    (config.run_dir / "stats.json").write_text(json.dumps({}))
    assert lc.is_done(config)
