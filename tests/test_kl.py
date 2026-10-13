import json
import struct
from unittest.mock import patch

import pytest

from src.config import Config
from src.kl import evaluate_kl, parse_kl, prepare_texts, reference_key


OUTPUT = "Mean    KLD: 0.012345 \u00b1 0.000123\nMedian  KLD: 0.008\n95.0%   KLD: 0.04\n"


def test_parse_kl_requires_final_statistics():
    assert parse_kl(OUTPUT) == {"mean": 0.012345, "median": 0.008,
                                "p95": 0.04, "uncertainty": 0.000123}
    with pytest.raises(ValueError, match="Missing"):
        parse_kl("failed to load model")


def make_config(tmp_path):
    model = tmp_path / "source.gguf"
    tool = tmp_path / "perplexity.exe"
    model.write_bytes(b"model")
    tool.write_bytes(b"tool")
    texts = tmp_path / "texts"
    texts.mkdir()
    (texts / "math.txt").write_text("held-out math text")
    return Config(project_root=tmp_path, source_model=model, llama_perplexity=tool,
                  kl_text_dir=texts, kl_context_size=128, kl_chunks=2)


def test_reference_key_changes_with_text_and_settings(tmp_path):
    config = make_config(tmp_path)
    text = config.kl_text_dir / "math.txt"
    original = reference_key(config, text)
    assert reference_key(config.with_overrides(kl_chunks=3), text) != original
    text.write_text("different text")
    assert reference_key(config, text) != original


def fake_reference(path, config):
    context, chunks, vocab = config.kl_context_size, config.kl_chunks, 3
    size = 20 + context * chunks * 4 + chunks * (context - 1 - context // 2) * 16
    path.write_bytes(b"_logits_" + struct.pack("<Iii", context, vocab, chunks)
                     + bytes(size - 20))


@patch("src.kl.run_perplexity")
def test_evaluate_kl_reuses_reference_and_drops_stale_metrics(mock_run, tmp_path):
    config = make_config(tmp_path)
    def run(command, log):
        if "--save-all-logits" in command:
            fake_reference(Path(command[-1]), config)
        return OUTPUT
    from pathlib import Path
    mock_run.side_effect = run
    output = evaluate_kl(config)
    result = json.loads(output.read_text())
    assert result["domains"]["math"]["tokens"] == 126
    assert result["domains"]["math"]["mean"] == 0.012345
    evaluate_kl(config)
    assert mock_run.call_count == 3  # reference once, comparison twice
    mock_run.return_value = "no statistics"
    mock_run.side_effect = None
    with pytest.raises(ValueError):
        evaluate_kl(config)
    assert not output.exists()


def test_custom_text_directory_must_contain_text(tmp_path):
    config = Config(kl_text_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        prepare_texts(config)


def test_kl_rejects_budget_too_small_for_statistics(tmp_path):
    with pytest.raises(ValueError, match="100 scored tokens"):
        evaluate_kl(Config(project_root=tmp_path, kl_context_size=32, kl_chunks=2))
