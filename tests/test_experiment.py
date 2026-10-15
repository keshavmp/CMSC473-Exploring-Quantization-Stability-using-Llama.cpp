from unittest.mock import Mock, patch

import pytest

from src import experiment
from src.config import Config


@patch.object(experiment, "stop_server")
@patch.object(experiment, "evaluate_model")
@patch.object(experiment, "start_server")
def test_run_experiment_stops_the_server_even_when_evaluation_fails(
    mock_start, mock_eval, mock_stop, tmp_path
):
    process, log = Mock(), Mock()
    mock_start.return_value = (process, log)
    mock_eval.side_effect = RuntimeError("evaluation blew up")

    with pytest.raises(RuntimeError):
        experiment.run_experiment(Config(name="demo", project_root=tmp_path))

    mock_stop.assert_called_once_with(process, log)


@patch.object(experiment, "run_experiment")
def test_run_experiments_records_failure_and_keeps_going(mock_run, tmp_path):
    def fake_run(cfg):
        if cfg.name == "bad":
            raise RuntimeError("server died")
        return tmp_path / "results" / f"{cfg.name}.json"

    mock_run.side_effect = fake_run

    summaries = experiment.run_experiments([Config(name="ok"), Config(name="bad")])

    assert [summary["status"] for summary in summaries] == ["ok", "failed"]
    assert summaries[0]["output"].endswith("ok.json")
    assert summaries[1]["error"] == "server died"


@patch.object(experiment, "evaluate_kl")
@patch.object(experiment, "stop_server")
@patch.object(experiment, "evaluate_model")
@patch.object(experiment, "start_server")
def test_kl_runs_after_server_stops(mock_start, mock_eval, mock_stop, mock_kl, tmp_path):
    events = []
    mock_start.return_value = (Mock(), Mock())
    mock_stop.side_effect = lambda *args: events.append("stop")
    mock_kl.side_effect = lambda cfg: events.append("kl")
    config = Config(project_root=tmp_path, kl_enabled=True)
    experiment.run_experiment(config)
    assert events == ["stop", "kl"]
    assert (config.run_dir / "config.json").exists()


@patch.object(experiment, "stop_server")
@patch.object(experiment, "evaluate_model")
@patch.object(experiment, "start_server")
def test_run_records_model_file_size(mock_start, mock_eval, mock_stop, tmp_path):
    import json
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model bytes")
    mock_start.return_value = (Mock(), Mock())
    config = Config(project_root=tmp_path, model_file=model)
    experiment.run_experiment(config)
    settings = json.loads((config.run_dir / "config.json").read_text())
    assert settings["model_size_bytes"] == len(b"model bytes")
    assert settings["model_file"] == str(model.resolve())
