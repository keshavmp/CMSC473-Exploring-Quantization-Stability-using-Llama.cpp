import json
import socket
import subprocess
import time
import threading
import lm_eval
import psutil
import requests
import argparse



# Support both package-style execution and direct execution.
# The recommended command from the repository root is:
# python -m src.run_kv_cache_experiments

try:
    from src.config import Config
except ModuleNotFoundError:
    from config import Config



cfg = Config()


# Resolve model, server executable, API address, and port
# from the shared project configuration.
MODEL = str(
    cfg.models_dir / "Llama-3.2-1B-Instruct-Q2_K.gguf"
)

SERVER = str(cfg.llama_server)
BASE_URL = cfg.base_url
PORT = int(cfg.port)

# ============================================================
# 1. CONFIGURATION
# ============================================================


 # Define experiment settings that can be changed from
# the terminal instead of hard-coding them below.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a llama.cpp KV-cache quantization experiment"
    )


    # Key and value cache precisions are independent variables.
    parser.add_argument(
        "--k-type",
        default="f16",
        choices=["f16", "q8_0", "q4_0"],
        help="KV-cache key precision",
    )
    parser.add_argument(
        "--v-type",
        default="f16",
        choices=["f16", "q8_0", "q4_0"],
        help="KV-cache value precision",
    )
     # Context capacity is held fixed when studying task type.
    parser.add_argument(
        "--context",
        type=int,
        default=4096,
        choices=[4096, 8192, 16384],
        help="Configured context capacity",
    )
     # Select exactly one benchmark task per invocation.
    parser.add_argument(
        "--task",
        default="arc_easy",
        choices=[
            "arc_easy",
            "gsm8k",
            "ifeval",
            "mmlu_college_computer_science",
            "mathqa",
        ],
        help="lm-eval task to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of evaluation samples",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    return args


args = parse_args()

k_type = args.k_type
v_type = args.v_type
context_limit = args.context

task_name = args.task
tasks = [task_name]

# CPU experiment
gpu_layers = 0

SAMPLE_LIMIT = args.limit

STARTUP_TIMEOUT_SECONDS = 120


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def get_metric(metrics_text, metric_name):
    """Return a metric value, or None if it is missing."""
    prefix = metric_name + " "


     # Find the line beginning with the requested metric name.
    for line in metrics_text.splitlines():
        if line.startswith(prefix):
            try:
                return float(line.split()[-1])
            except (ValueError, IndexError):
                return None

    # A missing or malformed metric should not crash parsing.
    return None


def read_metrics():
    """Fetch metrics from llama-server."""


     # The running server exposes measurements through /metrics.
    response = requests.get(
        f"{BASE_URL}/metrics",
        timeout=5,
    )
    response.raise_for_status()
    return response.text


def metric_delta(before_text, after_text, metric_name):
    """Calculate the difference between two cumulative metrics."""


    # Subtract the pre-evaluation reading from the post-evaluation
    # reading to isolate activity during this experiment.
    before = get_metric(before_text, metric_name)
    after = get_metric(after_text, metric_name)

    if before is None or after is None:
        return None

    delta = after - before


    # Avoid returning a negative difference
    return delta if delta >= 0 else None


# ============================================================
# 3. MEMORY MONITOR
# ============================================================

def monitor_memory():
    global peak_memory_mb


    # Sample the llama-server process's resident memory while it
    # is running, keeping track of the largest observed value.
    while (
        server.poll() is None
        and not stop_memory_monitor.is_set()
    ):
        try:
            memory_mb = (
                server_process.memory_info().rss
                / (1024 * 1024)
            )

            peak_memory_mb = max(
                peak_memory_mb,
                memory_mb,
            )

        except psutil.NoSuchProcess:
            break

         # Sample approximately every 100 milliseconds.
        stop_memory_monitor.wait(0.1)


# ============================================================
# 4. VALIDATE SETUP
# ============================================================

# Fail early if the configured model or server binary is missing.
if not MODEL or not __import__("os").path.isfile(MODEL):
    raise FileNotFoundError(f"Model file not found: {MODEL}")

if not __import__("os").path.isfile(SERVER):
    raise FileNotFoundError(
        f"llama-server binary not found: {SERVER}"
    )

print("Experiment configuration")
print(f"Model: {MODEL}")
print(f"K cache: {k_type}")
print(f"V cache: {v_type}")
print(f"Context limit: {context_limit}")
print(f"Tasks: {tasks}")
print(f"GPU layers: {gpu_layers}")
print(f"Sample limit: {SAMPLE_LIMIT}")


# Refuse to use a port that already has a server running.
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    if sock.connect_ex(("127.0.0.1", PORT)) == 0:
        raise RuntimeError(
            f"Port {PORT} is already in use. "
            "Identify and stop the existing server first."
        )


# ============================================================
# 5. SERVER SETUP
# ============================================================


# Launch llama-server as a separate operating-system process.
# Each run gets the selected K/V precision and context capacity.
server_command = [
    SERVER,
    "-m", MODEL,
    "--port", str(PORT),
    "--ctx-size", str(context_limit),
    "--n-gpu-layers", str(gpu_layers),
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


# ============================================================
# 6. RUN EXPERIMENT
# ============================================================

try:
    print("\nStarting server...")

    server = subprocess.Popen(server_command)
    server_process = psutil.Process(server.pid)

    # Wait for the server to become ready, with a timeout.
    print("Waiting for server...")
    startup_deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS

    while True:
        if server.poll() is not None:
            raise RuntimeError(
                "llama-server exited before becoming ready. "
                "Check the server path, model path, and port."
            )

        if time.monotonic() >= startup_deadline:
            raise TimeoutError(
                "llama-server did not become ready within "
                f"{STARTUP_TIMEOUT_SECONDS} seconds."
            )

        try:
            # A successful response confirms the API is available.
            response = requests.get(
                f"{BASE_URL}/v1/models",
                timeout=2,
            )

            if response.status_code == 200:
                break

        except requests.exceptions.RequestException:
             # The server may still be starting up; try again.
            pass

        time.sleep(1)

    print("Server is ready!")

    # Baseline memory after model loading.
    baseline_rss_mb = (
        server_process.memory_info().rss
        / (1024 * 1024)
    )

    peak_memory_mb = baseline_rss_mb

    # Start memory monitoring.
    memory_thread = threading.Thread(
        target=monitor_memory,
        daemon=True,
    )
    memory_thread.start()

    # Metrics before evaluation.
    before_experiment_metrics = read_metrics()

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    print("Running evaluation...")
    evaluation_start = time.perf_counter()



    # Evaluate the selected task through the running llama-server.
    # Use zero few-shot examples, a fixed sample limit, batch size 1,
    # and a fixed random seed for reproducibility where applicable.

    # Use GSM8K-specific generation settings only for math reasoning.
    model_args = (
       f"base_url={BASE_URL},timeout=600"
    )

    if task_name == "gsm8k":
        model_args += ",max_gen_toks=512"
    elif task_name == "ifeval":
         model_args += ",parallel=1"

    results = lm_eval.simple_evaluate(
        model="gguf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=0,
        limit=SAMPLE_LIMIT,
        batch_size=1,
        torch_random_seed=0,
        log_samples=True,
    )

    evaluation_wall_seconds = (
        time.perf_counter() - evaluation_start
    )

    print("Evaluation complete!")

    # Metrics after evaluation.
    after_experiment_metrics = read_metrics()

    


    # --------------------------------------------------------
    # Prompt-processing metrics
    # --------------------------------------------------------

    prompt_tokens = metric_delta(
        before_experiment_metrics,
        after_experiment_metrics,
        "llamacpp:prompt_tokens_total",
    )

    prompt_seconds = metric_delta(
        before_experiment_metrics,
        after_experiment_metrics,
        "llamacpp:prompt_seconds_total",
    )

    if (
        prompt_tokens is not None
        and prompt_seconds is not None
        and prompt_seconds > 0
    ):
        prompt_tokens_per_second = (
            prompt_tokens / prompt_seconds
        )
    else:
        prompt_tokens_per_second = None

    # Maximum sequence length observed by llama-server.
    max_observed_sequence_length = get_metric(
        after_experiment_metrics,
        "llamacpp:n_tokens_max",
    )

    # --------------------------------------------------------
    # Generation metrics
    # --------------------------------------------------------

    # The server's generation-time metric has produced implausible
    # values in previous runs. Keep token counts for diagnostics,
    # but do not report throughput until timing is validated.
    generation_tokens = metric_delta(
        before_experiment_metrics,
        after_experiment_metrics,
        "llamacpp:tokens_predicted_total",
    )

    # Diagnostic only: server timing has not been validated.
    generation_seconds_raw = metric_delta(
        before_experiment_metrics,
        after_experiment_metrics,
        "llamacpp:tokens_predicted_seconds_total",
    )

    generation_tokens_per_second = None

   

    # --------------------------------------------------------
    # Memory results
    # --------------------------------------------------------

    memory = {
        "baseline_rss_mb": baseline_rss_mb,
        "peak_rss_mb": peak_memory_mb,
        "peak_rss_delta_mb": (
            peak_memory_mb - baseline_rss_mb
        ),
        "measurement_note": (
            "RSS measures total llama-server process memory, "
            "not KV-cache-only memory."
        ),
    }

    # --------------------------------------------------------
    # Performance results
    # --------------------------------------------------------

    performance = {
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prompt_seconds": prompt_seconds,
        "generation_seconds_raw_unvalidated": generation_seconds_raw,
        "prompt_tokens_per_second": prompt_tokens_per_second,
        "generation_tokens_per_second": generation_tokens_per_second,
        "max_observed_sequence_length": max_observed_sequence_length,
        "evaluation_wall_seconds": evaluation_wall_seconds,
    }

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    

    result = {
        "model": MODEL,
        "weight_quant": "Q2_K",
        "k_type": k_type,
        "v_type": v_type,
        "context_limit": context_limit,
        "tasks": tasks,
        "backend": "cpu",
        "sample_limit": SAMPLE_LIMIT,
        "quality": results["results"],
        "performance": performance,
        "memory": memory,
    }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_dir = cfg.results_dir / "task21_kv_workloads"
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ## gsm8k debugger
    # if "samples" in results:
    #     sample_path = (
    #         results_dir
    #         / f"{k_type}_{v_type}_{task_name}_{context_limit}_samples.json"
    #     )
    #     with open(sample_path, "w", encoding="utf-8") as f:
    #         json.dump(results["samples"], f, indent=2)
    #     print(f"Saved evaluation samples to: {sample_path}")
    # else:
    #     print("WARNING: lm_eval returned no logged samples.")


    filename = (
        results_dir
        / f"{k_type}_{v_type}_{task_name}_{context_limit}.json"
    )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    print(f"Quality: {results['results']}")
    print(f"Prompt tok/s: {prompt_tokens_per_second}")
    print("Generation tok/s: not measured reliably")

    print(
        f"Evaluation wall time: "
        f"{evaluation_wall_seconds:.2f} seconds"
    )

    print(
        f"Max observed sequence length: "
        f"{max_observed_sequence_length}"
    )

    print(f"Baseline RSS: {baseline_rss_mb:.2f} MB")
    print(f"Peak RSS: {peak_memory_mb:.2f} MB")

    print(
        f"Peak RSS delta: "
        f"{peak_memory_mb - baseline_rss_mb:.2f} MB"
    )

    print(f"Saved results to: {filename}")
    print("=" * 60)

finally:
    # Stop memory monitoring.
    stop_memory_monitor.set()

    if memory_thread is not None:
        memory_thread.join(timeout=2)

    # Stop the server started by this runner.
    if server is not None and server.poll() is None:
        print("\nStopping server...")
        server.terminate()

        try:
            server.wait(timeout=10)

        except subprocess.TimeoutExpired:
            print("Server did not terminate; killing it.")
            server.kill()
            server.wait()

    print("Experiment cleanup complete.")