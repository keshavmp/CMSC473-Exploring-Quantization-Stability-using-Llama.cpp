from unittest.mock import patch

import src.imatrix as imatrix


class FakeRows:
    """Stand-in for a HF dataset: shuffle/select are no-ops, then it iterates."""

    def __init__(self, rows):
        self.rows = rows

    def shuffle(self, seed):
        return self

    def select(self, indices):
        return self

    def __iter__(self):
        return iter(self.rows)


def test_chat_format_wraps_one_message():
    assert (
        imatrix.chat_format("user", " hi ")
        == "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
    )


@patch("src.imatrix.load_dataset")
def test_get_math_examples_pairs_question_and_answer(mock_load):
    mock_load.return_value = FakeRows([{"question": "2+2?", "answer": "4"}])

    examples = imatrix.get_math_examples()

    assert examples == [
        "<|start_header_id|>user<|end_header_id|>\n\n2+2?<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n4<|eot_id|>"
    ]


@patch("src.imatrix.subprocess.run")
def test_make_imatrix_writes_text_file_and_runs_tool(mock_run, tmp_path, monkeypatch):
    monkeypatch.setattr(imatrix, "TEXT_DIR", tmp_path / "calibration")
    monkeypatch.setattr(imatrix, "IMATRIX_DIR", tmp_path / "imatrix")

    imatrix.make_imatrix("chat", ["one", "two"])

    text = (tmp_path / "calibration" / "chat.txt").read_text(encoding="utf-8")
    assert "one" in text and "two" in text

    command = mock_run.call_args.args[0]
    assert str(tmp_path / "imatrix" / "chat.imatrix.gguf") in command
    mock_run.assert_called_once()
