
import os
import shutil
import subprocess
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_MODEL = (
    PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct-BF16.gguf"
)

TEXT_DIR = PROJECT_ROOT / "data" / "calibration"
IMATRIX_DIR = PROJECT_ROOT / "models" / "imatrix"

NUM_EXAMPLES = 300
SEED = 42
GPU_LAYERS = int(os.getenv("IMATRIX_GPU_LAYERS", "0"))

# Override with LLAMA_IMATRIX environment variable if needed.
LLAMA_IMATRIX = os.getenv("LLAMA_IMATRIX", "llama-imatrix")


def chat_format(role, text):
    """Format a message with Llama 3 chat tokens."""
    return (
        f"<|start_header_id|>{role}<|end_header_id|>\n\n"
        f"{text.strip()}<|eot_id|>"
    )


def get_chat_examples():
    dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
    )

    dataset = dataset.shuffle(seed=SEED)
    dataset = dataset.select(range(NUM_EXAMPLES))

    examples = []

    for row in dataset:
        messages = [
            chat_format(m["role"], m["content"])
            for m in row["messages"]
        ]
        examples.append("\n".join(messages))

    return examples


def get_math_examples():
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
    )

    dataset = dataset.shuffle(seed=SEED)
    dataset = dataset.select(range(NUM_EXAMPLES))

    examples = []

    for row in dataset:
        question = chat_format("user", row["question"])
        answer = chat_format("assistant", row["answer"])
        examples.append(question + "\n" + answer)

    return examples


def get_code_examples():
    dataset = load_dataset(
        "google-research-datasets/mbpp",
        "full",
        split="train",
    )

    dataset = dataset.shuffle(seed=SEED)
    dataset = dataset.select(range(NUM_EXAMPLES))

    examples = []

    for row in dataset:
        question = chat_format("user", row["text"])
        answer = chat_format("assistant", row["code"])
        examples.append(question + "\n" + answer)

    return examples


def save_calibration_text(task, examples):
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    text_file = TEXT_DIR / f"{task}.txt"

    content = "<|begin_of_text|>\n\n"
    content += "\n\n".join(examples)

    text_file.write_text(content, encoding="utf-8")

    print(f"Saved {task} calibration text: {text_file}")
    return text_file


def make_imatrix(task, examples):
    IMATRIX_DIR.mkdir(parents=True, exist_ok=True)

    text_file = save_calibration_text(task, examples)
    output_file = IMATRIX_DIR / f"{task}.imatrix.gguf"

    command = [
        LLAMA_IMATRIX,
        "--model", str(SOURCE_MODEL),
        "--file", str(text_file),
        "--output-file", str(output_file),
        "--output-format", "gguf",
        "--n-gpu-layers", str(GPU_LAYERS),
        "--parse-special",
        "--no-ppl",
    ]

    print(f"\nGenerating {task} importance matrix...")

    subprocess.run(command, check=True)

    if not output_file.is_file():
        raise RuntimeError(f"Missing imatrix output: {output_file}")

    size_mib = output_file.stat().st_size / (1024 ** 2)

    print(f"Saved: {output_file}")
    print(f"Size: {size_mib:.2f} MiB")

    return output_file


def main():
    if shutil.which(LLAMA_IMATRIX) is None and not Path(LLAMA_IMATRIX).is_file():
        raise FileNotFoundError(
            f"Cannot find llama-imatrix: {LLAMA_IMATRIX}"
        )

    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(
            f"Source model not found: {SOURCE_MODEL}"
        )

    tasks = {
        "chat": get_chat_examples,
        "math": get_math_examples,
        "code": get_code_examples,
    }

    for task, loader in tasks.items():
        print(f"\nLoading {task} dataset...")
        examples = loader()
        make_imatrix(task, examples)

    print("\nAll importance matrices generated.")


if __name__ == "__main__":
    main()
