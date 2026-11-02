from unittest.mock import patch

import pytest

from src.quantize import quantize_model


def make_inputs(tmp_path):
    tool = tmp_path / "llama-quantize.exe"
    tool.touch()
    source = tmp_path / "model.gguf"
    source.touch()
    return tool, source


def test_missing_quantize_tool_raises(tmp_path):
    _, source = make_inputs(tmp_path)

    with pytest.raises(FileNotFoundError):
        quantize_model(tmp_path / "nope.exe", source, tmp_path / "out.gguf")


def test_missing_source_model_raises(tmp_path):
    tool, _ = make_inputs(tmp_path)

    with pytest.raises(FileNotFoundError):
        quantize_model(tool, tmp_path / "nope.gguf", tmp_path / "out.gguf")


@patch("src.quantize.subprocess.run")
def test_builds_pure_command_with_imatrix(mock_run, tmp_path):
    tool, source = make_inputs(tmp_path)
    imatrix = tmp_path / "chat.imatrix.gguf"
    imatrix.touch()
    output = tmp_path / "models" / "chat-q4.gguf"

    quantize_model(tool, source, output, "Q4_K_M", imatrix_file=imatrix, pure=True)

    command = mock_run.call_args.args[0]
    assert command[0] == str(tool)
    assert "--pure" in command
    assert command[command.index("--imatrix") + 1] == str(imatrix)
    assert command[-3:] == [str(source), str(output), "Q4_K_M"]
    assert output.parent.is_dir()  # destination folder is created
