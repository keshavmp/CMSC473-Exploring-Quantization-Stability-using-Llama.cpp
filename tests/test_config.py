from pathlib import Path

from src.config import Config


def test_result_paths_follow_the_run_name():
    cfg = Config(name="chat-q4")

    assert cfg.results_file.name == "chat-q4.json"
    assert cfg.samples_file.name == "chat-q4_samples.json"
    assert cfg.log_file.name == "chat-q4-server.log"
    assert cfg.base_url == "http://127.0.0.1:8080"


def test_resolve_model_file_defaults_to_models_dir():
    cfg = Config(name="uniform-q4")

    assert cfg.resolve_model_file() == cfg.models_dir / "uniform-q4.gguf"


def test_resolve_model_file_prefers_an_explicit_path():
    cfg = Config(model_file=Path("elsewhere.gguf"))

    assert cfg.resolve_model_file() == Path("elsewhere.gguf")


def test_with_overrides_returns_a_modified_copy():
    cfg = Config()
    other = cfg.with_overrides(name="chat-q4", num_examples=50)

    assert other.name == "chat-q4"
    assert other.num_examples == 50
    assert cfg.name == "uniform-q4"  # original is untouched
