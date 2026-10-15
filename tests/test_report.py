import json

from src.report import build_table, find_result_files, save_csv, save_plot, score_for


def write_result(results_dir, name, tasks):
    path = results_dir / f"{name}.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")
    return path


def test_find_result_files_skips_samples(tmp_path):
    write_result(tmp_path, "chat-q4", {"gsm8k": {}})
    write_result(tmp_path, "chat-q4_samples", [{"prompt": "hi"}])

    assert [path.name for path in find_result_files(tmp_path)] == ["chat-q4.json"]


def test_score_for_prefers_the_primary_metric():
    task_results = {
        "exact_match,strict-match": 0.28,
        "exact_match_stderr,strict-match": 0.06,
        "exact_match,flexible-extract": 0.31,
    }

    assert score_for("gsm8k", task_results) == 0.28


def test_score_for_falls_back_to_the_first_real_number():
    task_results = {"acc,none": 0.30, "acc_stderr,none": 0.065}

    assert score_for("some_new_task", task_results) == 0.30


def test_score_for_returns_none_when_everything_is_na():
    assert score_for("ifeval", {"inst_level_strict_acc,none": "N/A"}) is None


def test_build_table_rows_are_runs_and_columns_are_tasks(tmp_path):
    write_result(
        tmp_path,
        "chat-q4",
        {
            "gsm8k": {"exact_match,strict-match": 0.28},
            "ifeval": {"prompt_level_strict_acc,none": 0.36},
        },
    )
    write_result(
        tmp_path,
        "uniform-q4",
        {"gsm8k": {"exact_match,strict-match": 0.25}},  # no ifeval
    )

    table = build_table(find_result_files(tmp_path))

    assert list(table.index) == ["chat-q4", "uniform-q4"]
    assert list(table.columns) == ["gsm8k", "ifeval"]
    assert table.loc["chat-q4", "gsm8k"] == 28.0
    assert table.loc["chat-q4", "ifeval"] == 36.0
    assert table.loc["uniform-q4", "ifeval"] != table.loc["uniform-q4", "ifeval"]  # blank


def test_save_csv_and_plot_create_files(tmp_path):
    write_result(tmp_path, "chat-q4", {"gsm8k": {"exact_match,strict-match": 0.28}})
    table = build_table(find_result_files(tmp_path))

    save_csv(table, tmp_path / "summary.csv")
    save_plot(table, tmp_path / "summary.png")

    assert "chat-q4,28.0" in (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert (tmp_path / "summary.png").stat().st_size > 0


def test_nested_results_include_kl_without_scaling(tmp_path):
    directory = tmp_path / "demo"
    directory.mkdir()
    (directory / "metrics.json").write_text(json.dumps({"gsm8k": {"exact_match,strict-match": 0.5}}))
    (directory / "kl.json").write_text(json.dumps({"domains": {"math": {
        "mean": 0.02, "uncertainty": 0.001, "median": 0.01, "p95": 0.1}}}))
    (directory / "samples.json").write_text("[]")
    (directory / "config.json").write_text("{}")
    write_result(tmp_path, "demo", {"gsm8k": {"exact_match,strict-match": 0.1}})
    write_result(tmp_path, "old", {"gsm8k": {"exact_match,strict-match": 0.3}})
    files = find_result_files(tmp_path)
    assert len(files) == 2
    table = build_table(files)
    assert table.loc["demo", "gsm8k"] == 50.0
    assert table.loc["demo", "kl_math_mean"] == 0.02
    assert table.loc["old", "kl_math_mean"] != table.loc["old", "kl_math_mean"]
    from src.report import save_kl_plot
    save_plot(table, tmp_path / "scores.png")
    save_kl_plot(table, tmp_path / "kl.png")
    assert (tmp_path / "kl.png").stat().st_size > 0


def test_report_accepts_kv_cache_task_metrics(tmp_path):
    path = write_result(tmp_path, "kv", {"quality": {"arc_easy": {"acc,none": 0.4}},
                                         "performance": {"prompt_tokens": 10}})
    table = build_table([path])
    assert list(table.columns) == ["arc_easy"]
    assert table.loc["kv", "arc_easy"] == 40.0


def test_model_size_prefers_snapshot_and_plots_quality(tmp_path):
    from src.report import model_size_mib, save_size_plot
    run = tmp_path / "results" / "demo"
    run.mkdir(parents=True)
    metrics = run / "metrics.json"
    metrics.write_text(json.dumps({"gsm8k": {"exact_match,strict-match": 0.5}}))
    models = tmp_path / "models"
    models.mkdir()
    (models / "demo.gguf").write_bytes(bytes(1024**2))
    assert model_size_mib(metrics) == 1.0
    (run / "config.json").write_text(json.dumps({"model_size_bytes": 2 * 1024**2}))
    table = build_table([metrics])
    assert table.loc["demo", "size_mib"] == 2.0
    assert table.loc["demo", "gsm8k"] == 50.0
    save_size_plot(table, tmp_path / "size.png")
    save_plot(table, tmp_path / "scores.png")
    assert (tmp_path / "size.png").stat().st_size > 0


def test_stability_matches_prompts_and_ignores_alternate_filters(tmp_path):
    from src.report import build_stability_table
    def sample(number, correct, filter_name="strict-match", prompt="same"):
        return {"doc_hash": str(number), "prompt_hash": prompt, "target_hash": "target",
                "filter": filter_name, "exact_match": correct}
    def result(name, samples):
        run = tmp_path / name
        run.mkdir()
        path = run / "metrics.json"
        path.write_text("{}")
        (run / "samples.json").write_text(json.dumps({"gsm8k": samples}))
        return path
    baseline = result("BF16", [sample(1, 1), sample(2, 0), sample(3, 1), sample(4, 0)])
    run = result("q4", [sample(3, 1), sample(2, 1), sample(1, 0), sample(4, 1, prompt="different"),
                        sample(1, 1, filter_name="flexible-extract"), sample(5, 0)])
    table = build_stability_table([run], baseline)
    assert table.iloc[0].to_dict() == {"run": "q4", "task": "gsm8k", "compared": 3,
                                     "correct_to_incorrect": 1, "incorrect_to_correct": 1,
                                     "unchanged": 1}
    assert build_stability_table([run], None).empty
    assert build_stability_table([baseline], baseline).empty


def test_stability_uses_binary_primary_metrics_for_ifeval_and_mmlu():
    from src.report import sample_scores
    common = {"doc_hash": "doc", "prompt_hash": "prompt", "target_hash": "target", "filter": "none"}
    assert list(sample_scores([dict(common, prompt_level_strict_acc=False,
                                    inst_level_strict_acc=[True])], "ifeval").values()) == [False]
    assert list(sample_scores([dict(common, acc=1.0)], "mmlu_college_computer_science").values()) == [True]
    assert sample_scores([dict(common, acc=0.5)], "mmlu_college_computer_science") == {}
