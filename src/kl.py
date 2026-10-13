"""Domain-specific KL divergence against the BF16 source, using llama-perplexity."""

import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

from src.config import Config


# Match imatrix's seed and skip its first 300 training examples.
CALIBRATION_EXAMPLES = 300
TEXT_EXAMPLES = 32
SEED = 42


def prepare_texts(config: Config) -> list[Path]:
    """Use custom .txt files, or save a fixed held-out subset per domain once."""
    directory = config.kl_text_dir or config.project_root / "data" / "kl"
    if config.kl_text_dir is None:
        from datasets import load_dataset

        directory.mkdir(parents=True, exist_ok=True)
        domains = [
            ("chat", "HuggingFaceH4/ultrachat_200k", None, "train_sft"),
            ("math", "openai/gsm8k", "main", "train"),
            ("code", "google-research-datasets/mbpp", "full", "train"),
        ]
        for domain, dataset, subset, split in domains:
            path = directory / f"{domain}.txt"
            if path.exists():
                continue
            data = load_dataset(dataset, subset, split=split).shuffle(seed=SEED)
            data = data.select(range(CALIBRATION_EXAMPLES,
                                     CALIBRATION_EXAMPLES + TEXT_EXAMPLES))
            examples = []
            for row in data:
                if domain == "chat":
                    messages = row["messages"]
                else:
                    question = row["question"] if domain == "math" else row["text"]
                    answer = row["answer"] if domain == "math" else row["code"]
                    messages = [{"role": "user", "content": question},
                                {"role": "assistant", "content": answer}]
                examples.append("\n".join(f"{m['role']}: {m['content'].strip()}"
                                          for m in messages))
            path.write_text("\n\n".join(examples),
                            encoding="utf-8")
    files = sorted(directory.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No KL text files in {directory}")
    return files


def parse_kl(text: str) -> dict[str, float]:
    """Read llama.cpp's final statistics; missing output is an evaluation failure."""
    number = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    metrics = {}
    for key, label in [("mean", "Mean"), ("median", "Median"), ("p95", "95.0%")]:
        match = re.search(re.escape(label) + r"\s+KLD:\s*" + number, text)
        if match is None:
            raise ValueError(f"Missing {label} KLD in llama-perplexity output")
        metrics[key] = float(match[1])
    uncertainty = re.search(r"Mean\s+KLD:\s*" + number + r"\s*\u00b1\s*" + number, text)
    if uncertainty is None:
        raise ValueError("Missing mean KLD uncertainty")
    metrics["uncertainty"] = float(uncertainty[2])
    return metrics


def reference_key(config: Config, text_file: Path) -> str:
    """Invalidate the shared reference when inputs or evaluation settings change."""
    identity = {}
    for label, path in [("model", config.source_model), ("tool", config.llama_perplexity)]:
        stat = path.stat()
        identity[label] = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    identity.update(text=hashlib.sha256(text_file.read_bytes()).hexdigest(),
                    context=config.kl_context_size, chunks=config.kl_chunks,
                    gpu_layers=config.gpu_layers)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def run_perplexity(command: list[str], log_file: Path) -> str:
    with log_file.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return log_file.read_text(encoding="utf-8", errors="replace")


def reference_tokens(path: Path, config: Config) -> int:
    """Validate the reference header and require the requested number of chunks."""
    with path.open("rb") as file:
        header = file.read(20)
    if len(header) != 20 or header[:8] != b"_logits_":
        raise ValueError(f"Invalid KL reference: {path}")
    context, vocabulary, chunks = struct.unpack("<Iii", header[8:])
    expected = 20 + context * chunks * 4 + chunks * (context - 1 - context // 2) * (2 * (2 * ((vocabulary + 1) // 2) + 4))
    if context != config.kl_context_size or chunks != config.kl_chunks or path.stat().st_size != expected:
        raise ValueError(f"Incomplete KL reference: {path}; check text length and log")
    return chunks * (context - 1 - context // 2)


def evaluate_kl(config: Config) -> Path:
    """Evaluate each domain sequentially and save metrics plus raw tool logs."""
    if config.kl_context_size < 4 or config.kl_chunks < 1:
        raise ValueError("KL needs context >= 4 and chunks >= 1")
    if config.kl_chunks * (config.kl_context_size - 1 - config.kl_context_size // 2) < 100:
        raise ValueError("KL needs at least 100 scored tokens for final statistics")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    output = config.run_dir / "kl.json"
    output.unlink(missing_ok=True)  # never retain old KL metrics after a failed rerun
    domains = {}
    for text_file in prepare_texts(config):
        domain = text_file.stem
        cache = config.results_dir / "_kl_reference" / reference_key(config, text_file)
        cache.mkdir(parents=True, exist_ok=True)
        reference = cache / "reference.kld"
        common = [str(config.llama_perplexity), "--verbosity", "4", "--ctx-size", str(config.kl_context_size),
                  "--batch-size", str(config.kl_context_size), "--chunks", str(config.kl_chunks),
                  "--n-gpu-layers", config.gpu_layers]
        if not reference.exists():
            partial = cache / "reference.partial"
            run_perplexity(common + ["--model", str(config.source_model),
                           "--file", str(text_file), "--save-all-logits", str(partial)],
                           cache / "reference.log")
            reference_tokens(partial, config)
            partial.replace(reference)
        tokens = reference_tokens(reference, config)
        log_file = config.run_dir / f"kl-{domain}.log"
        text = run_perplexity(common + ["--model", str(config.resolve_model_file()),
                              "--kl-divergence-base", str(reference), "--kl-divergence"],
                              log_file)
        domains[domain] = dict(parse_kl(text), tokens=tokens,
                               text_file=str(text_file), reference=str(reference))
    output.write_text(json.dumps({"baseline": str(config.source_model),
                                 "direction": "BF16 || evaluated model", "units": "nats",
                                 "domains": domains}, indent=2), encoding="utf-8")
    return output
