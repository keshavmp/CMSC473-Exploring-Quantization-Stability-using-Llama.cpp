"""Context-length x weight-quantization experiments for Llama-3.2-1B on CPU.

Run from the repository root:

    python -m rohan.run_long_context --prepare            # tokenizer, essays, KL text
    python -m rohan.run_long_context --part pilot
    python -m rohan.run_long_context --part night1 --resume
    python -m rohan.run_long_context --part night1 --list # show runs and status

Experiments (see rohan/README.md):
    ruler  RULER accuracy at a target prompt length (E2, E2b, E4)
    gsm8k  The pure sweep's GSM8K questions behind L tokens of essay padding (E3)
    kl     KL divergence from BF16 with a long window (E1), via src/kl.py

Every run writes rohan/results/<experiment>/<run>/ with metrics.json (or
kl.json), samples.json, config.json, server.log, and stats.json holding time,
throughput, prompt-cache, and memory measurements.
"""
import argparse
import json
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import requests

from rohan.run_family_sweep import ROHAN_DIR, family_configs, tool
from src.config import PROJECT_ROOT, Config
from src.gguf import GGUFLM
from src.kl import evaluate_kl


TOKENIZER = "unsloth/Llama-3.2-1B-Instruct"  # Llama's own tokenizer sizes every prompt
EXPERIMENT = "Llama_Long_Context"
GENERATION_RESERVE = 512   # server context = target length + reserve
KL_TOKENS = 32768          # every KL window scores the same leading text
DATA_DIR = ROHAN_DIR / "data" / "long_context"
KL_TEXT_DIR = DATA_DIR / "kl"
GSM8K_STOP = ["Question:", "</s>", "<|im_end|>"]   # installed gsm8k.yaml
GSM8K_MAX_TOKENS = 256

# Llama-3.2-1B attention shape, for the KV-cache size llama.cpp allocates (f16).
LAYERS, KV_HEADS, HEAD_DIM = 16, 8, 64

BASE = Config(
    project_root=ROHAN_DIR,
    experiment=EXPERIMENT,
    gpu_layers="0",
    request_workers=1,         # CPU prefill is compute bound; parallel slots don't help
    request_timeout=1800,
    server_start_timeout=300,
    kl_enabled=True,
    llama_server=tool("llama-server"),
    llama_perplexity=tool("llama-perplexity"),
)


def level_models() -> dict[str, Path]:
    """BF16, pure Q2-Q8 from the family sweep, and task15's mixed Q2_K model."""
    models = {}
    for config in family_configs("llama"):
        label = "BF16" if config.name == "BF16-pure-sweep" else config.name.split("-")[1].upper()
        models[label] = config.model_file
    models["Q2_K-mixed"] = PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct-Q2_K.gguf"
    return models


@dataclass(frozen=True)
class Run:
    kind: str          # "ruler", "gsm8k", or "kl"
    level: str         # key of level_models()
    length: int        # target prompt length, padding length, or KL window
    task: str = ""     # RULER task name
    n: int = 0         # questions (unused for kl)

    @property
    def name(self) -> str:
        level, k = self.level.lower(), f"{self.length // 1024}k"
        if self.kind == "ruler":
            return f"ruler-{level}-{self.task}-{k}"
        if self.kind == "gsm8k":
            return f"gsm8k-{level}-pad{k}"
        return f"kl-{level}-w{k}"


def ruler(levels, lengths, tasks, n) -> list[Run]:
    """Length-major order, so an interrupted night leaves complete short lengths."""
    return [Run("ruler", level, length, task, n)
            for length in lengths for task in tasks for level in levels]


K4, K8, K16, K32 = 4096, 8192, 16384, 32768
HARD = ["niah_multikey_2", "ruler_vt"]
KL_WINDOWS = (K4, K8, K16)
PARTS = {
    "pilot": (ruler(["Q4", "BF16"], [K4, K16], ["niah_single_1", *HARD], 3)
              + [Run("gsm8k", "Q4", K16, n=5), Run("kl", "Q4", K4)]),
    # E2 core: does the gap to BF16 widen with length?
    "night1": ruler(["BF16", "Q4"], [K4, K8, K16], HARD, 20),
    "night2": ruler(["Q2"], [K4], HARD, 20) + ruler(["Q6", "Q3"], [K4, K8, K16], HARD, 20),
    # E3 padded GSM8K. Answers run to the 256-token cap and generation at 16K is
    # ~4.4 tok/s (pilot), so 16K uses the first 50 of the sweep's 100 questions.
    "night3": [Run("gsm8k", level, length, n=n) for length, n in ((K4, 100), (K16, 50))
               for level in ("BF16", "Q8", "Q6", "Q4", "Q3")],
    # E1 KL (Q2 first builds each BF16 reference), then E2b at 4K and 16K for Q2s.
    # No 32K window: llama-perplexity keeps n_ctx x 128K-vocab float logits
    # (~17 GB at 32K), more than this 16 GB Mac has.
    "night4": ([Run("kl", level, window) for window in KL_WINDOWS
                for level in ("Q2", "Q3", "Q4", "Q5", "Q6", "Q8")]
               + [Run("kl", "BF16", K4)]   # self-KL floor check
               + ruler(["BF16", "Q8", "Q6", "Q5", "Q4", "Q3", "Q2", "Q2_K-mixed"], [K4],
                       ["niah_single_1"], 20)
               + ruler(["Q2", "Q2_K-mixed"], [K16], ["niah_single_1"], 20)),
    # Rest of E2b at 16K (task15 bridge), then E4 32K stretch.
    "night5": (ruler(["BF16", "Q4"], [K16], ["niah_single_1"], 20)
               + ruler(["BF16", "Q4"], [K32], ["niah_multikey_2"], 10)),
}


def experiment_for(part: str) -> str:
    return f"{EXPERIMENT}-pilot" if part == "pilot" else EXPERIMENT


def run_config(run: Run, part: str) -> Config:
    models = level_models()
    return BASE.with_overrides(
        name=run.name,
        experiment=experiment_for(part),
        model_file=models[run.level],
        source_model=models["BF16"],
        context_size=run.length + GENERATION_RESERVE,
        quantization_type=run.level,
        pure_quantization=run.level.startswith("Q") and run.level != "Q2_K-mixed",
        kl_text_dir=KL_TEXT_DIR,
        kl_context_size=run.length,
        kl_chunks=max(1, KL_TOKENS // run.length),
    )


def is_done(config: Config) -> bool:
    return (config.run_dir / "stats.json").is_file()


# --- Measurement ---------------------------------------------------------------
class PeakMemory:
    """Sample resident memory every 0.1 s: one process, or all children of this one."""

    def __init__(self, pid: int | None = None):
        self.pid, self.peak_mb = pid, 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def current_mb(self) -> float:
        try:
            procs = ([psutil.Process(self.pid)] if self.pid
                     else psutil.Process().children(recursive=True))
            return sum(proc.memory_info().rss for proc in procs) / 1024**2
        except psutil.Error:
            return 0.0

    def _watch(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self.current_mb())
            self._stop.wait(0.1)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def kv_cache_mib(context: int) -> float:
    """f16 K and V for every layer; matches llama.cpp's 'llama_kv_cache: size' line."""
    return context * LAYERS * KV_HEADS * HEAD_DIM * 2 * 2 / 1024**2


def timing_stats(timings: list[dict], questions: int, seconds: float) -> dict:
    """Summarize llama-server's per-request `timings` objects."""
    processed = sum(t["prompt_n"] for t in timings)
    cached = sum(t.get("cache_n", 0) for t in timings)
    prompt_ms = sum(t["prompt_ms"] for t in timings)
    generated = sum(t["predicted_n"] for t in timings)
    generate_ms = sum(t["predicted_ms"] for t in timings)
    requests_made = max(len(timings), 1)
    return {
        "eval_seconds": seconds,
        "questions": questions,
        "seconds_per_question": seconds / max(questions, 1),
        "requests": len(timings),
        "mean_prompt_tokens": (processed + cached) / requests_made,
        "prompt_tokens_processed": processed,
        "prompt_tokens_cached": cached,
        "cache_hit_rate": cached / max(processed + cached, 1),
        "prompt_tokens_per_second": processed / (prompt_ms / 1000) if prompt_ms else None,
        "mean_time_to_first_token_s": prompt_ms / 1000 / requests_made,
        "generated_tokens": generated,
        "generation_tokens_per_second": generated / (generate_ms / 1000) if generate_ms else None,
    }


class TimedGGUFLM(GGUFLM):
    """The repository's lm-eval backend, keeping llama-server's per-request timings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timings = []

    def _post(self, endpoint, payload):
        response = super()._post(endpoint, payload)
        if isinstance(response, dict) and "timings" in response:
            self.timings.append(response["timings"])
        return response


# --- Server ------------------------------------------------------------------
def start_server(config: Config) -> tuple[subprocess.Popen, object, float]:
    """Like src/server.start_server, plus one slot, metrics, and flash attention on."""
    with socket.socket() as sock:
        if sock.connect_ex((config.host, config.port)) == 0:
            raise RuntimeError(f"Port {config.port} is in use; stop the other server first.")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    log = open(config.log_file, "w+", encoding="utf-8")
    command = [
        str(config.llama_server), "--model", str(config.model_file),
        "--n-gpu-layers", config.gpu_layers, "--ctx-size", str(config.context_size),
        "--parallel", "1", "--flash-attn", "on", "--metrics",
        # The default 8 GB RAM prompt cache stores every past prompt's KV state,
        # which inflated night1's peak memory by ~2 GB. The slot still reuses
        # its own prefix, so padded GSM8K keeps its cache hits.
        "--cache-ram", "0",
        "--host", config.host, "--port", str(config.port), "-lv", str(config.server_log_verbosity),
    ]
    start = time.monotonic()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    deadline = start + config.server_start_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.close()
            raise RuntimeError(f"llama-server stopped. Check: {config.log_file}")
        try:
            if requests.get(f"{config.base_url}/health", timeout=2).ok:
                return process, log, time.monotonic() - start
        except requests.RequestException:
            pass
        time.sleep(0.5)
    process.terminate()
    log.close()
    raise TimeoutError(f"llama-server did not start within {config.server_start_timeout} s")


def stop_server(process: subprocess.Popen, log) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    log.close()


# --- Experiments -------------------------------------------------------------
def evaluate_ruler(run: Run, config: Config) -> tuple[dict, list, dict]:
    import lm_eval

    with TimedGGUFLM(base_url=config.base_url, max_length=config.context_size,
                     request_workers=1, timeout=config.request_timeout) as model:
        start = time.monotonic()
        results = lm_eval.simple_evaluate(
            model=model, tasks=[run.task], num_fewshot=0, limit=run.n,
            random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
            fewshot_random_seed=1234, batch_size=1, log_samples=True,
            metadata={"max_seq_lengths": [run.length], "tokenizer": TOKENIZER},
        )
        stats = timing_stats(model.timings, run.n, time.monotonic() - start)
    metrics = results["results"]
    metrics["accuracy"] = metrics[run.task][f"{run.length},none"]
    return metrics, results["samples"][run.task], stats


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


def essay_text() -> str:
    """RULER's Paul Graham essay haystack, as one string."""
    from lm_eval.tasks.ruler.prepare_niah import get_haystack

    return " ".join(get_haystack(type_haystack="essay"))


def padding(length: int, tok=None) -> str:
    """The first `length` tokens of the essays, decoded back to text."""
    tok = tok or tokenizer()
    ids = tok(essay_text()[: length * 8], add_special_tokens=False)["input_ids"]
    if len(ids) < length:
        raise ValueError(f"Essay text is shorter than {length} tokens")
    return tok.decode(ids[:length])


def gsm8k_prompt(filler: str, question: str) -> str:
    """Padding, then the pure sweep's exact GSM8K prompt."""
    return f"{filler}\n\nQuestion: {question}\nAnswer:"


def evaluate_gsm8k(run: Run, config: Config) -> tuple[dict, list, dict]:
    from datasets import load_dataset

    from rohan.analyze_findings import NEW_QUESTION, last_number, normalize

    docs = load_dataset("openai/gsm8k", "main", split="test").select(range(run.n))
    filler = padding(run.length)
    session, records, timings = requests.Session(), [], []
    start = time.monotonic()
    for doc_id, doc in enumerate(docs):
        response = session.post(f"{config.base_url}/v1/completions", timeout=config.request_timeout,
                                json={"prompt": gsm8k_prompt(filler, doc["question"]),
                                      "max_tokens": GSM8K_MAX_TOKENS, "temperature": 0,
                                      "stop": GSM8K_STOP}).json()
        timings.append(response["timings"])
        text, target = response["choices"][0]["text"].strip(), normalize(doc["answer"])
        records.append({
            "doc_id": doc_id, "target": doc["answer"], "response": text,
            "exact_match": normalize(last_number(text)) == target,
            "first_answer_match": normalize(last_number(NEW_QUESTION.split(text)[0])) == target,
        })
    stats = timing_stats(timings, run.n, time.monotonic() - start)
    metrics = {"accuracy": sum(r["exact_match"] for r in records) / len(records),
               "first_answer_accuracy": sum(r["first_answer_match"] for r in records) / len(records),
               "questions": len(records)}
    return metrics, records, stats


def run_one(run: Run, part: str) -> None:
    config = run_config(run, part)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("stats.json", "metrics.json", "samples.json", "kl.json"):
        (config.run_dir / name).unlink(missing_ok=True)
    settings = asdict(config) | {"run": asdict(run), "tokenizer": TOKENIZER, "flash_attn": "on"}
    (config.run_dir / "config.json").write_text(json.dumps(settings, indent=2, default=str))
    model_mib = config.model_file.stat().st_size / 1024**2
    start = time.monotonic()

    if run.kind == "kl":
        with PeakMemory() as memory:  # llama-perplexity runs as a child process
            output = evaluate_kl(config)
        domains = json.loads(output.read_text())["domains"]
        tokens = sum(d["tokens"] for d in domains.values())
        wall = time.monotonic() - start
        # The first level per window (Q2) also pays for the BF16 reference pass.
        stats = {"kl_mean": sum(d["mean"] * d["tokens"] for d in domains.values()) / tokens,
                 "kv_cache_mib": kv_cache_mib(run.length),
                 "peak_rss_mb": memory.peak_mb, "kl_window": run.length, "kl_chunks": config.kl_chunks,
                 "scored_tokens": tokens, "processed_tokens": KL_TOKENS,
                 "processed_tokens_per_second": KL_TOKENS / wall}
    else:
        process, log, load_seconds = start_server(config)
        try:
            memory = PeakMemory(process.pid)
            baseline = memory.current_mb()
            with memory:
                evaluate = evaluate_ruler if run.kind == "ruler" else evaluate_gsm8k
                metrics, samples, stats = evaluate(run, config)
        finally:
            stop_server(process, log)
        (config.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        (config.run_dir / "samples.json").write_text(json.dumps(samples, indent=2, default=str))
        stats |= {"load_seconds": load_seconds, "baseline_rss_mb": baseline,
                  "peak_rss_mb": memory.peak_mb, "rss_increase_mb": memory.peak_mb - baseline,
                  "kv_cache_mib": kv_cache_mib(config.context_size),
                  "accuracy": metrics["accuracy"]}
    stats |= {"run": run.name, "kind": run.kind, "level": run.level, "task": run.task,
              "length": run.length, "server_context": config.context_size,
              "model_file_mib": model_mib, "cache_ram_mib": 0,
              "wall_seconds": time.monotonic() - start}
    (config.run_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))


def prepare() -> None:
    """Cache the tokenizer, RULER essays, GSM8K, and the long KL text."""
    from datasets import load_dataset

    tok = tokenizer()
    KL_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    text = padding(KL_TOKENS + 2048, tok)  # margin for llama-perplexity's own tokenization
    (KL_TEXT_DIR / "essays.txt").write_text(text, encoding="utf-8")
    load_dataset("openai/gsm8k", "main", split="test")
    print(f"KL text: {KL_TEXT_DIR / 'essays.txt'} ({len(tok(text)['input_ids'])} tokens)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=PARTS)
    parser.add_argument("--resume", action="store_true", help="skip runs with stats.json")
    parser.add_argument("--list", action="store_true", help="list the part's runs and stop")
    parser.add_argument("--prepare", action="store_true", help="cache data, then exit")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if args.part is None:
        parser.error("--part is required")

    runs = PARTS[args.part]
    if args.list:
        for run in runs:
            status = "done" if is_done(run_config(run, args.part)) else "todo"
            print(f"{status}  {run.name}  (n={run.n})")
        return
    for path in [BASE.llama_server, BASE.llama_perplexity, *level_models().values()]:
        if not Path(path).is_file():
            raise SystemExit(f"Missing: {path}")

    failed = []
    for index, run in enumerate(runs, 1):
        if args.resume and is_done(run_config(run, args.part)):
            print(f"Skipping finished run: {run.name}")
            continue
        print(f"\n=== [{index}/{len(runs)}] {run.name} ===", flush=True)
        try:
            run_one(run, args.part)
        except Exception as error:
            print(f"[{run.name}] failed: {error}", flush=True)
            failed.append(run.name)
    print(f"\nFinished {args.part}: {len(runs) - len(failed)} ok, {len(failed)} failed {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
