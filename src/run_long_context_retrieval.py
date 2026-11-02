
import argparse
import json
import socket
import subprocess
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import lm_eval
import psutil
import requests

try:
    from src.config import Config
except ModuleNotFoundError:
    from config import Config


# ============================================================
# 1. CONFIGURATION 
# ============================================================

cfg = Config()

MODEL = str(
    cfg.models_dir / "Llama-3.2-1B-Instruct-Q2_K.gguf"
)

SERVER = str(cfg.llama_server)
BASE_URL = cfg.base_url.rstrip("/")
PORT = int(cfg.port)

TASK = "niah_single_1"

STARTUP_TIMEOUT_SECONDS = 120
REQUEST_TIMEOUT_SECONDS = 900
GENERATION_RESERVE = 512

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure long-context retrieval quality and memory "
            "for llama.cpp KV-cache quantization."
        )
    )

    parser.add_argument(
        "--k-type",
        default="f16",
        choices=["f16", "q8_0", "q4_0"],
        help="Key-cache precision",
    )

    parser.add_argument(
        "--v-type",
        default="f16",
        choices=["f16", "q8_0", "q4_0"],
        help="Value-cache precision",
    )

    parser.add_argument(
        "--context",
        type=int,
        default=8192,
        choices=[4096, 8192, 16384],
        help="Target RULER input sequence length",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Maximum evaluation samples",
    )

    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to Metal; 0 means CPU-only",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON filename",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    if args.gpu_layers < 0:
        parser.error("--gpu-layers must be nonnegative")

    return args


# ============================================================
# 2. METRICS AND MEMORY HELPERS
# ============================================================

def read_metrics():
    """Read llama-server's Prometheus metrics, if available."""
    try:
        response = requests.get(
            f"{BASE_URL}/metrics",
            timeout=5,
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return None


def parse_metrics(metrics_text):
    """Parse simple Prometheus metric lines into a dictionary."""
    metrics = {}

    if not metrics_text:
        return metrics

    for line in metrics_text.splitlines():
        line = line.strip()

        if not line or line.startswith("#") or " " not in line:
            continue

        name, value = line.rsplit(None, 1)

        try:
            metrics[name] = float(value)
        except ValueError:
            continue

    return metrics


def calculate_metric_deltas(before_text, after_text):
    """Return differences for metrics present in both snapshots."""
    before = parse_metrics(before_text)
    after = parse_metrics(after_text)

    deltas = {}

    for name in before.keys() & after.keys():
        delta = after[name] - before[name]

        if delta >= 0:
            deltas[name] = delta

    return deltas


def monitor_memory():
    """Track peak RSS of the llama-server process."""
    global peak_memory_mb

    while (
        server is not None
        and server.poll() is None
        and not stop_memory_monitor.is_set()
    ):
        try:
            memory_mb = (
                server_process.memory_info().rss / (1024 * 1024)
            )
            peak_memory_mb = max(peak_memory_mb, memory_mb)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break

        stop_memory_monitor.wait(0.1)


def wait_for_server():
    """Wait until the server responds to its models endpoint."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                "llama-server exited before becoming ready "
                f"(exit code {server.returncode}). "
                "Inspect the visible server output above."
            )

        try:
            response = requests.get(
                f"{BASE_URL}/v1/models",
                timeout=2,
            )

            if response.status_code == 200:
                return

        except requests.RequestException:
            pass

        time.sleep(1)

    raise TimeoutError(
        f"llama-server did not become ready within "
        f"{STARTUP_TIMEOUT_SECONDS} seconds."
    )


# ============================================================
# 3. VALIDATE SETUP
# ============================================================

args = parse_args()

k_type = args.k_type
v_type = args.v_type
context_limit = args.context
sample_limit = args.limit
gpu_layers = args.gpu_layers

# This assignment is important: the CLI setting must reach
# the actual llama-server command.
GPU_LAYERS = gpu_layers

server_context = context_limit + GENERATION_RESERVE

if not Path(MODEL).is_file():
    raise FileNotFoundError(f"Model file not found: {MODEL}")

if not Path(SERVER).is_file():
    raise FileNotFoundError(
        f"llama-server binary not found: {SERVER}"
    )

print("\nExperiment configuration")
print(f"Model: {MODEL}")
print(f"Task: {TASK}")
print(f"K cache: {k_type}")
print(f"V cache: {v_type}")
print(f"Target input length: {context_limit}")
print(f"Server context capacity: {server_context}")
print(f"Sample limit: {sample_limit}")
print(f"GPU layers requested: {GPU_LAYERS}")
print(
    "Backend: CPU-only"
    if GPU_LAYERS == 0
    else "Backend: Metal offloading requested; verify startup logs"
)

# Do not accidentally connect to or interfere with another server.
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    if sock.connect_ex(("127.0.0.1", PORT)) == 0:
        raise RuntimeError(
            f"Port {PORT} is already in use. "
            "Stop the existing server before running this experiment."
        )


# ============================================================
# 4. START SERVER AND EVALUATE
# ============================================================

server_command = [
    SERVER,
    "-m", MODEL,
    "--port", str(PORT),
    "--ctx-size", str(server_context),
    "--parallel", "1",
    "--n-gpu-layers", str(GPU_LAYERS),
    "--cache-type-k", k_type,
    "--cache-type-v", v_type,
    "--metrics",
]

server = None
server_process = None
memory_thread = None

baseline_rss_mb = None
peak_memory_mb = 0.0
stop_memory_monitor = threading.Event()

started_at = datetime.now().astimezone().isoformat()
experiment_start = time.monotonic()

metrics_before = None
metrics_after = None
evaluation_results = None
evaluation_error = None
server_exit_code = None
evaluation_seconds = None

try:
    print("\nStarting llama-server...")
    print("Command:", " ".join(server_command))
    print("Server logs are enabled for debugging.")

    # Keep server logs visible so startup errors and Metal offloading
    # can be verified directly.
    server = subprocess.Popen(server_command)
    server_process = psutil.Process(server.pid)

    print("Waiting for server...")
    wait_for_server()
    print("Server is ready.")

    baseline_rss_mb = (
        server_process.memory_info().rss / (1024 * 1024)
    )
    peak_memory_mb = baseline_rss_mb

    memory_thread = threading.Thread(
        target=monitor_memory,
        daemon=True,
    )
    memory_thread.start()

    metrics_before = read_metrics()

    print(f"\nRunning {TASK}...")
    print(
        f"Requested sequence length: {context_limit}; "
        f"sample limit: {sample_limit}"
    )
    print(f"HTTP request timeout: {REQUEST_TIMEOUT_SECONDS} seconds")

    evaluation_start = time.monotonic()

    evaluation_results = lm_eval.simple_evaluate(
        model="gguf",
        model_args=(
            f"base_url={BASE_URL},"
            f"timeout={REQUEST_TIMEOUT_SECONDS},"
        ),
        tasks=[TASK],
        num_fewshot=0,
        batch_size=1,
        limit=sample_limit,
        log_samples=True,
        metadata={
            "max_seq_lengths": [context_limit],
            "tokenizer": "Qwen/Qwen3-0.6B",
        },
    )

    evaluation_seconds = time.monotonic() - evaluation_start
    print(f"Evaluation finished in {evaluation_seconds:.2f} seconds.")

except Exception as exc:
    evaluation_error = f"{type(exc).__name__}: {exc}"
    print(f"\nExperiment failed: {evaluation_error}")
    traceback.print_exc()

finally:
    # Capture metrics before stopping the server.
    metrics_after = read_metrics()

    stop_memory_monitor.set()

    if memory_thread is not None:
        memory_thread.join(timeout=2)

    if server is not None and server.poll() is None:
        print("\nStopping llama-server...")
        server.terminate()

        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Server did not terminate; killing it.")
            server.kill()
            server.wait()

    if server is not None:
        server_exit_code = server.returncode

    print("Server cleanup complete.")


# ============================================================
# 5. SAVE RESULTS
# ============================================================

elapsed_seconds = time.monotonic() - experiment_start

evaluation_summary = None
sample_records = None

if isinstance(evaluation_results, dict):
    evaluation_summary = dict(evaluation_results)
    sample_records = evaluation_summary.pop("samples", None)

if args.output:
    output_path = Path(args.output)
    samples_dir = output_path.parent / "samples"
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"task15_{TASK}_k-{k_type}_v-{v_type}_"
        f"ctx-{context_limit}_gpu-{GPU_LAYERS}_{timestamp}.json"
    )
    output_path = OUTPUT_DIR / "statistics" / filename
    samples_dir = OUTPUT_DIR / "samples"

sample_output_path = None

if sample_records is not None:
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_output_path = samples_dir / output_path.name

    sample_payload = {
        "started_at": started_at,
        "task": TASK,
        "model": MODEL,
        "weight_quant": "Q2_K",
        "k_type": k_type,
        "v_type": v_type,
        "target_input_length": context_limit,
        "sample_limit": sample_limit,
        "gpu_layers_requested": GPU_LAYERS,
        "samples": sample_records,
    }

    with sample_output_path.open("w", encoding="utf-8") as file:
        json.dump(sample_payload, file, indent=2, default=str)

result = {
    "started_at": started_at,
    "model": MODEL,
    "task": TASK,
    "weight_quant": "Q2_K",
    "k_type": k_type,
    "v_type": v_type,
    "target_input_length": context_limit,
    "server_context_capacity": server_context,
    "generation_reserve": GENERATION_RESERVE,
    "sample_limit": sample_limit,
    "gpu_layers_requested": GPU_LAYERS,
    "backend": (
        "cpu"
        if GPU_LAYERS == 0
        else "metal_offload_requested_verify_server_logs"
    ),
    "baseline_server_rss_mb": baseline_rss_mb,
    "peak_server_rss_mb": peak_memory_mb,
    "server_rss_increase_mb": (
        peak_memory_mb - baseline_rss_mb
        if baseline_rss_mb is not None
        else None
    ),
    "elapsed_seconds": elapsed_seconds,
    "evaluation_seconds": evaluation_seconds,
    "metrics_before": parse_metrics(metrics_before),
    "metrics_after": parse_metrics(metrics_after),
    "metric_deltas": calculate_metric_deltas(
        metrics_before,
        metrics_after,
    ),
    "evaluation": evaluation_summary,
    "sample_results_file": (
        str(sample_output_path)
        if sample_output_path is not None
        else None
    ),
    "error": evaluation_error,
    "server_exit_code": server_exit_code,
}

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(result, file, indent=2, default=str)

print(f"\nStatistics saved to: {output_path}")

if sample_output_path is not None:
    print(f"Sample results saved to: {sample_output_path}")
elif evaluation_results is not None:
    print("Warning: no sample records were returned to save separately.")

if evaluation_error:
    raise RuntimeError(
        "The experiment did not complete successfully. "
        f"See the saved statistics JSON: {output_path}"
    )

print("\nEvaluation summary:")
print(
    json.dumps(
        (evaluation_summary or {}).get("results", {}),
        indent=2,
        default=str,
    )
)
print(f"Peak server RSS: {peak_memory_mb:.2f} MB")


