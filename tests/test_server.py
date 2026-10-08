import subprocess
from unittest.mock import Mock

import pytest

from src.config import Config
from src.server import get_model_file, stop_server


def test_get_model_file_returns_the_local_gguf(tmp_path):
    model = tmp_path / "models" / "demo.gguf"
    model.parent.mkdir()
    model.touch()

    cfg = Config(project_root=tmp_path, name="demo")

    assert get_model_file(cfg) == model


def test_get_model_file_explains_how_to_fix_a_missing_model(tmp_path):
    cfg = Config(project_root=tmp_path, name="missing")

    with pytest.raises(FileNotFoundError, match="hf_repo"):
        get_model_file(cfg)


def test_stop_server_terminates_a_running_process():
    process, log = Mock(), Mock()
    process.poll.return_value = None

    stop_server(process, log)

    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=10)
    log.close.assert_called_once()


def test_stop_server_leaves_an_already_finished_process_alone():
    process, log = Mock(), Mock()
    process.poll.return_value = 0

    stop_server(process, log)

    process.terminate.assert_not_called()
    log.close.assert_called_once()


def test_stop_server_kills_a_process_that_wont_exit():
    process, log = Mock(), Mock()
    process.poll.return_value = None
    process.wait.side_effect = subprocess.TimeoutExpired(cmd="llama-server", timeout=10)

    stop_server(process, log)

    process.kill.assert_called_once()
