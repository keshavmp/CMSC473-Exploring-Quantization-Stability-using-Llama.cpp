"""Run per-layer KV-cache sensitivity experiments for Task 31."""

from pathlib import Path
import argparse
import platform
import subprocess
import sys
import time

from lm_eval.tasks import TaskManager, get_task_dict


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLAMA_CPP_ROOT = PROJECT_ROOT.parent / "llama.cpp"

MODEL_PATH = (
    PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct-Q2_K.gguf"
)

SCRIPT_DIR = PROJECT_ROOT / "src" / "kv_layer_sensitivity"

CAPTURE = SCRIPT_DIR / "capture_kv"
QUANTIZE = SCRIPT_DIR / "quantize_kv"
ANALYZE = SCRIPT_DIR / "analyze_kv_layer_sensitivity.py"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task31_kv_layer_sensitivity"
    / "workloads"
)

# These reflect the settings in capture_kv.cpp.
CONTEXT_SIZE = 4096
GPU_LAYERS = 99


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run layer-wise KV-cache sensitivity experiments"
    )

    parser.add_argument(
        "--task",
        default="arc_easy",
        choices=[
            "arc_easy",
            "mathqa",
            "mmlu_college_computer_science",
        ],
        help="Benchmark workload",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of workload samples",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    return args


# ============================================================
# 2. ENVIRONMENT CHECKS
# ============================================================

def check_environment():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"llama.cpp root: {LLAMA_CPP_ROOT}")
    print(f"Model: {MODEL_PATH}")
    print(f"Context size: {CONTEXT_SIZE}")
    print(f"GPU layers requested: {GPU_LAYERS}")
    print("Inference backend: Metal requested")

    required_paths = [
        MODEL_PATH,
        LLAMA_CPP_ROOT,
        CAPTURE,
        QUANTIZE,
        ANALYZE,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found: {path}"
            )

    print("\nEnvironment checks passed.")


# ============================================================
# 3. LOAD WORKLOAD PROMPTS
# ============================================================

def load_prompts(task_name, limit):
    print(f"\nLoading workload: {task_name}")

    task_manager = TaskManager()

    task_dict = get_task_dict(
        [task_name],
        task_manager=task_manager,
    )

    task = task_dict[task_name]

    # Match Task 21's zero-shot configuration.
    task.set_config(
        key="num_fewshot",
        value=0,
    )

    # Use the benchmark's evaluation split.
    if task.has_test_docs():
        docs = task.test_docs()
    elif task.has_validation_docs():
        docs = task.validation_docs()
    else:
        raise RuntimeError(
            f"No evaluation documents found for {task_name}"
        )

    prompts = []

    for index, doc in enumerate(docs):
        if index >= limit:
            break

        prompt = task.doc_to_text(doc)

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Invalid prompt for sample {index}"
            )

        prompts.append(prompt)

    if not prompts:
        raise RuntimeError("No prompts were loaded")

    print(f"Loaded {len(prompts)} prompts")

    return prompts


# ============================================================
# 4. PROCESS INDIVIDUAL SAMPLE
# ============================================================

def process_sample(prompt, sample_dir):
    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the original prompt for reproducibility.
    prompt_path = sample_dir / "prompt.txt"

    prompt_path.write_text(
        prompt,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Capture original F16 K/V tensors using Metal
    # --------------------------------------------------------

    print("Capturing F16 KV tensors...")

    subprocess.run(
        [
            str(CAPTURE),
            prompt,
            str(sample_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    # --------------------------------------------------------
    # Quantize and reconstruct using Q8_0 and Q4_0
    # --------------------------------------------------------

    print("Quantizing KV tensors...")

    subprocess.run(
        [
            str(QUANTIZE),
            str(sample_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    # --------------------------------------------------------
    # Calculate layer-wise reconstruction errors
    # --------------------------------------------------------

    print("Analyzing reconstruction errors...")

    subprocess.run(
        [
            sys.executable,
            str(ANALYZE),
            "--results-dir",
            str(sample_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print(f"Sample results saved to: {sample_dir}")


# ============================================================
# 5. RUN WORKLOAD EXPERIMENT
# ============================================================

def main():
    args = parse_args()

    check_environment()

    prompts = load_prompts(
        args.task,
        args.limit,
    )

    task_dir = RESULTS_DIR / args.task

    task_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_start = time.perf_counter()

    for index, prompt in enumerate(prompts):

        print("\n" + "=" * 60)
        print(f"Task: {args.task}")
        print(f"Sample: {index + 1}/{len(prompts)}")
        print("=" * 60)

        sample_dir = task_dir / f"sample_{index:03d}"

        sample_start = time.perf_counter()

        process_sample(
            prompt,
            sample_dir,
        )

        sample_elapsed = (
            time.perf_counter() - sample_start
        )

        print(
            f"Sample elapsed time: "
            f"{sample_elapsed:.2f} seconds"
        )

    experiment_elapsed = (
        time.perf_counter() - experiment_start
    )

    print("\n" + "=" * 60)
    print("WORKLOAD EXPERIMENT COMPLETE")
    print("=" * 60)

    print(f"Task: {args.task}")
    print(f"Samples processed: {len(prompts)}")
    print(f"GPU layers requested: {GPU_LAYERS}")
    print(f"Elapsed time: {experiment_elapsed:.2f} seconds")
    print(f"Results directory: {task_dir}")


if __name__ == "__main__":
    main()