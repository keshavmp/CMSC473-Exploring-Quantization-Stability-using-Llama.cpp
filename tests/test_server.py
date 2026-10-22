import subprocess
from unittest.mock import Mock, patch

import pytest

from src.config import Config
from src.server import get_model_file, start_server, stop_server


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


def test_server_allocates_context_for_each_worker(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    config = Config(project_root=tmp_path, model_file=model,
                    request_workers=4, context_size=2048)
    with patch("src.server.subprocess.Popen") as popen, \
         patch("src.server.requests.get") as get, \
         patch("src.server.time.sleep"):
        popen.return_value.poll.return_value = None
        get.return_value.ok = True
        process, log = start_server(config)
        try:
            command = popen.call_args.args[0]
            assert command[command.index("--ctx-size") + 1] == "8192"
            assert command[command.index("--parallel") + 1] == "4"
            assert "--cont-batching" in command
            assert command[command.index("-lv") + 1] == "3"
        finally:
            log.close()
