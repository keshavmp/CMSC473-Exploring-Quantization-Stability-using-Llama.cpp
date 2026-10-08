import json
from unittest.mock import patch

from src.config import Config
from src.evaluate import evaluate_model

FAKE_RESULTS = {
    "results": {"gsm8k,none": {"exact_match,strict-match": 0.5}},
    "samples": [{"prompt": "hi", "completion": "there"}],
}


@patch("src.evaluate.lm_eval.simple_evaluate")
def test_evaluate_model_writes_results_and_samples(mock_eval, tmp_path):
    mock_eval.return_value = FAKE_RESULTS
    cfg = Config(project_root=tmp_path, name="demo")

    output = evaluate_model(cfg, limit=3)

    assert output == tmp_path / "results" / "demo.json"
    assert json.loads(output.read_text(encoding="utf-8")) == FAKE_RESULTS["results"]

    samples = tmp_path / "results" / "demo_samples.json"
    assert json.loads(samples.read_text(encoding="utf-8")) == FAKE_RESULTS["samples"]

    assert mock_eval.call_args.kwargs["limit"] == 3
