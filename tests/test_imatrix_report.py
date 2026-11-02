import math
import json

import pandas as pd
import pytest

from src.imatrix_report import PROTOCOL_KEYS, complete_domain_average, generate_report, summarize


def example_data():
    rows = []
    values = {"BF16": {"chat": [1, 1], "math": [1, 1]},
              "q2": {"chat": [3, -1], "math": [2, 2]},
              "q4": {"chat": [2, 2], "math": [1, 1]}}
    for model, tasks in values.items():
        for task, gains in tasks.items():
            for layer, gain in enumerate(gains):
                rows.append(dict(protocol="same", model=model, task=task, kind="layer", statistic="gain",
                                 layer=str(layer), tensor="", calibration_sha256=task, value=gain,
                                 matrix=f"{model}/{task}"))
    return pd.DataFrame(rows)


def test_signed_changes_do_not_hide_absolute_changes():
    tables = summarize(example_data(), "BF16")
    row = tables["model_task_changes"].query("model == 'q2' and task == 'chat'").iloc[0]
    assert row.mean_signed_delta == 0
    assert row.mean_absolute_delta == 2
    assert row.n_matched_units == 2
    overall = tables["overall_change_averages"].iloc[0]
    assert overall.mean_absolute_delta == 1
    assert overall.n_models == 2
    unit = tables["unit_task_averages"].query("task == 'chat' and layer == '0'").iloc[0]
    assert unit.mean_value == 2
    assert unit.n_models == 3


def test_task_contrasts_isolate_change_from_baseline():
    data = example_data()
    data.loc[data["task"] == "math", "value"] += 10
    tables = summarize(data, "BF16")
    row = tables["task_contrasts"].query("model == 'q2'").iloc[0]
    assert row.mean_task_b_minus_a == 11
    assert row.mean_change_b_minus_a == 1
    assert row.mean_absolute_change_difference == 2


def test_model_averages_weight_tasks_equally_despite_unit_counts():
    data = example_data()
    extra = data.query("task == 'math'").copy()
    extra["layer"] = extra["layer"].astype(int).add(2).astype(str)
    tables = summarize(pd.concat([data, extra]), "BF16")
    row = tables["model_change_averages"].query("model == 'q2'").iloc[0]
    assert row.mean_signed_delta == 0.5
    assert row.mean_absolute_delta == 1.5
    assert row.n_tasks == 2


def test_mismatched_calibration_is_not_compared_with_baseline():
    data = example_data()
    data.loc[data["model"] == "q2", "calibration_sha256"] = "different"
    tables = summarize(data, "BF16")
    assert "q2" not in set(tables["paired_changes"]["model"])
    row = tables["task_contrasts"].query("model == 'q2'").iloc[0]
    assert row.n_baseline_matched_units == 0
    assert math.isnan(row.mean_change_b_minus_a)


def test_undefined_values_do_not_count_as_zero_changes():
    data = example_data()
    data.loc[(data["model"] == "q2") & (data["task"] == "chat") & (data["layer"] == "0"), "value"] = float("nan")
    row = summarize(data, "BF16")["model_task_changes"].query("model == 'q2' and task == 'chat'").iloc[0]
    assert row.n_matched_units == 1
    assert row.mean_signed_delta == -2


def test_ambiguous_repeat_runs_are_rejected():
    data = example_data()
    with pytest.raises(ValueError, match="Multiple runs"):
        summarize(pd.concat([data, data]), "BF16")


def test_report_from_export_marks_partial_coverage_and_handles_windows_json(tmp_path):
    data = example_data()
    frame = data[["matrix", "model", "task", "layer", "value"]].rename(columns={"value": "gain"})
    frame.to_csv(tmp_path / "layer_statistics.csv", index=False)
    frame["tensor"] = frame["layer"].map(lambda layer: f"blk.{layer}.attn_q.weight")
    frame.to_csv(tmp_path / "tensor_statistics.csv", index=False)
    inventory = []
    for row in data[["matrix", "task"]].drop_duplicates().itertuples():
        settings = {key: "same" for key in PROTOCOL_KEYS}
        settings["calibration_sha256"] = row.task
        inventory.append({"matrix": row.matrix, "collection": settings})
    (tmp_path / "inventory.json").write_text(json.dumps({"matrices": inventory}))
    (tmp_path / "collection_plan.json").write_text(json.dumps({
        "expected_matrices": 21, "models": ["BF16", "q2", "q4"]}), encoding="utf-8-sig")
    generate_report(tmp_path, "BF16")
    assert "PARTIAL: 6/21" in (tmp_path / "analysis.txt").read_text()
    exported = pd.read_csv(tmp_path / "overall_change_averages.csv")
    assert set(exported["mean_absolute_delta"]) == {1.0}
    assert list(tmp_path.glob("gain_heatmap_*.png"))
    assert list(tmp_path.glob("gain_by_precision_*.png"))
    assert list((tmp_path / "task_comparisons").glob("gain_tasks_*.png"))


def test_layer_averages_exclude_unfinished_models_and_preserve_task_weighting():
    data = example_data()
    data = data[~((data["model"] == "q4") & (data["task"] == "math"))]
    result = complete_domain_average(data, ["chat", "math"])
    assert "q4" not in set(result["model"])
    assert result.query("model == 'q2' and layer == '0'").iloc[0]["value"] == 2.5
    assert result.query("model == 'q2' and layer == '1'").iloc[0]["value"] == 0.5
