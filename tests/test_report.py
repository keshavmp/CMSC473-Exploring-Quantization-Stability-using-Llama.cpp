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
