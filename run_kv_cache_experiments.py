import subprocess
import time
import requests
import json
import lm_eval
import psutil
import threading
from pathlib import Path

MODEL = "models/Llama-3.2-1B-Instruct-Q2_K.gguf"
SERVER = "../llama.cpp/build/bin/llama-server"

BASE_URL = "http://127.0.0.1:8080"


# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

k_type = "f16"
v_type = "f16"
context_limit = "4096"
task_name = "arc_easy"
tasks = ["arc_easy"]
gpu_layers = "0"


# ============================================================
# 2. SERVER SETUP
# ============================================================

server_command = [
    SERVER,
    "-m", MODEL,
    "--port", "8080",
    "--ctx-size", context_limit,
    "--n-gpu-layers", gpu_layers,
    "--cache-type-k", k_type,
    "--cache-type-v", v_type,
    "--metrics"
]

print("Starting server...")
print(f"K cache: {k_type}")
print(f"V cache: {v_type}")
print(f"Context Limit: {context_limit}")
print(f"Tasks: {tasks}")

server = subprocess.Popen(server_command)

server_process = psutil.Process(server.pid)

peak_memory_mb = 0


def monitor_memory():
    global peak_memory_mb

    while server.poll() is None:
        try:
            memory_mb = server_process.memory_info().rss / (1024 * 1024)

            if memory_mb > peak_memory_mb:
                peak_memory_mb = memory_mb

        except psutil.NoSuchProcess:
            break

        time.sleep(0.1)


memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

try:

    # ========================================================
    # 3. WAIT FOR SERVER TO START
    # ========================================================

    print("Waiting for server...")

    while True:

        # Check whether OUR server process has died.
        if server.poll() is not None:
            raise RuntimeError(
                "The llama-server process exited before becoming ready. "
                "Check whether port 8080 is already in use."
            )

        try:
            response = requests.get(
                f"{BASE_URL}/v1/models",
                timeout=2
            )

            if response.status_code == 200:
                break

        except requests.exceptions.ConnectionError:
            pass

        time.sleep(1)

    print("Server is ready!")

    # ========================================================
    # 4. GET INITIAL METRICS
    # ========================================================

    before_experiment_metrics = requests.get(
        f"{BASE_URL}/metrics"
    ).text

    # ========================================================
    # 5. RUN THE EVALUATION
    # ========================================================

    print("Running Evaluation")

    results = lm_eval.simple_evaluate(
        model="gguf",
        model_args=f"base_url={BASE_URL}, timeout=30",
        tasks=tasks,
        num_fewshot=0,
        limit=50,
        batch_size=1,
        torch_random_seed=None,
    )

    memory = {
        "peak_rss_mb": peak_memory_mb
    }

    # ========================================================
    # 6. GET FINAL METRICS
    # ========================================================

    after_experiment_metrics = requests.get(
        f"{BASE_URL}/metrics"
    ).text

    # ========================================================
    # 7. CALCULATE PERFORMANCE NUMBERS
    # ========================================================

    def get_metric(metrics_text, metric_name):

        for line in metrics_text.splitlines():

            if line.startswith(metric_name + " "):
                return float(line.split()[-1])

        return None

    prompt_tokens_before = get_metric(
        before_experiment_metrics,
        "llamacpp:prompt_tokens_total"
    )

    prompt_tokens_after = get_metric(
        after_experiment_metrics,
        "llamacpp:prompt_tokens_total"
    )

    generation_tokens_before = get_metric(
        before_experiment_metrics,
        "llamacpp:tokens_predicted_total"
    )

    generation_tokens_after = get_metric(
        after_experiment_metrics,
        "llamacpp:tokens_predicted_total"
    )

    prompt_tokens_per_second = get_metric(
        after_experiment_metrics,
        "llamacpp:prompt_tokens_seconds"
    )

    generation_tokens_per_second = get_metric(
        after_experiment_metrics,
        "llamacpp:predicted_tokens_seconds"
    )

    max_observed_sequence_length = get_metric(
        after_experiment_metrics,
        "llamacpp:n_tokens_max"
    )

    prompt_tokens = prompt_tokens_after - prompt_tokens_before
    generation_tokens = (
        generation_tokens_after - generation_tokens_before
    )

    performance = {
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prompt_tokens_per_second": prompt_tokens_per_second,
        "generation_tokens_per_second": generation_tokens_per_second,
        "max_observed_sequence_length": max_observed_sequence_length
    }

    # ========================================================
    # 8. COMBINE EVERYTHING INTO ONE RESULT
    # ========================================================

    result = {
        "model": MODEL,
        "weight_quant": "Q2_K",

        "k_type": k_type,
        "v_type": v_type,

        "context_limit": context_limit,
        "tasks": tasks,

        "backend": "cpu",

        "quality": results["results"],

        "performance": performance,

        "memory": memory
    }

    # ========================================================
    # 9. SAVE RESULT
    # ========================================================

    Path("results").mkdir(exist_ok=True)

    filename = (
        f"results/"
        f"{k_type}_{v_type}_{task_name}_{context_limit}.json"
    )

    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved results to {filename}")


finally:

    # ========================================================
    # 10. SHUT DOWN SERVER
    # ========================================================

    print("Stopping server...")

    server.terminate()
    server.wait()

    memory_thread.join()

    print(f"Peak memory: {peak_memory_mb:.2f} MB")

    print("Experiment complete!")