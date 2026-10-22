import json
from unittest.mock import patch
from unittest.mock import Mock

from src.config import Config
from src.evaluate import evaluate_model
from src.evaluate import resolve_tasks

FAKE_RESULTS = {
    "results": {"gsm8k,none": {"exact_match,strict-match": 0.5}},
    "samples": [{"prompt": "hi", "completion": "there"}],
}


@patch("src.evaluate.lm_eval.simple_evaluate")
def test_evaluate_model_writes_results_and_samples(mock_eval, tmp_path):
    mock_eval.return_value = FAKE_RESULTS
    cfg = Config(project_root=tmp_path, name="demo")

    output = evaluate_model(cfg, limit=3)

    assert output == tmp_path / "results" / "demo" / "metrics.json"
    assert json.loads(output.read_text(encoding="utf-8")) == FAKE_RESULTS["results"]

    samples = tmp_path / "results" / "demo" / "samples.json"
    assert json.loads(samples.read_text(encoding="utf-8")) == FAKE_RESULTS["samples"]

    assert mock_eval.call_args.kwargs["limit"] == 3


@patch("src.evaluate.TaskManager")
def test_task_caps_select_identical_leading_ids_and_record_actual_split(manager):
    tasks = {}
    for name, count, split in [("gsm8k", 1319, "test"), ("ifeval", 541, "train"), ("hellaswag", 10042, "validation")]:
        task = Mock(eval_docs=range(count))
        task.dump_config.return_value = {"test_split": split}
        tasks[name] = task
    manager.return_value.load.return_value = {"tasks": tasks}
    cfg = Config(tasks=list(tasks), num_examples=None,
                 task_sample_caps={"gsm8k": 1000, "ifeval": None, "hellaswag": 2000})
    _, _, selection, details = resolve_tasks(cfg)
    assert {name: len(ids) for name, ids in selection.items()} == {"gsm8k": 1000, "ifeval": 541, "hellaswag": 2000}
    assert selection["hellaswag"] == list(range(2000))
    assert details["ifeval"]["split"] == "train"
    assert details["hellaswag"]["subset"] is True
