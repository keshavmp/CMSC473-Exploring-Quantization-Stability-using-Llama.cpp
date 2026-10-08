import subprocess
from itertools import islice
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LLAMA_IMATRIX = Path(r"C:\llama.cpp\build\bin\Release\llama-imatrix.exe")
SOURCE_MODEL = PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct-BF16.gguf"

NUM_EXAMPLES = 300   # examples per task
SEED = 42

TEXT_DIR = PROJECT_ROOT / "data" / "calibration"
IMATRIX_DIR = PROJECT_ROOT / "models" / "imatrix"


def chat_format(role, text):
    """Wrap one message in Llama 3 chat tokens."""
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n{text.strip()}<|eot_id|>"


def get_chat_examples():
    data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    data = data.shuffle(seed=SEED).select(range(NUM_EXAMPLES))

    examples = []
    for row in data:
        messages = []
        for m in row["messages"]:
            messages.append(chat_format(m["role"], m["content"]))
        examples.append("\n".join(messages))
    return examples


def get_math_examples():
    data = load_dataset("openai/gsm8k", "main", split="train")
    data = data.shuffle(seed=SEED).select(range(NUM_EXAMPLES))

    examples = []
    for row in data:
        question = chat_format("user", row["question"])
        answer = chat_format("assistant", row["answer"])
        examples.append(question + "\n" + answer)
    return examples


def get_code_examples():
    data = load_dataset("google-research-datasets/mbpp", "full", split="train")
    data = data.shuffle(seed=SEED).select(range(NUM_EXAMPLES))

    examples = []
    for row in data:
        question = chat_format("user", row["text"])
        answer = chat_format("assistant", row["code"])
        examples.append(question + "\n" + answer)
    return examples


def make_imatrix(task, examples):
    """Save the examples as a text file, then run llama-imatrix on it."""
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    IMATRIX_DIR.mkdir(parents=True, exist_ok=True)

    text_file = TEXT_DIR / f"{task}.txt"
    text_file.write_text("<|begin_of_text|>\n\n" + "\n\n".join(examples), encoding="utf-8")

    output_file = IMATRIX_DIR / f"{task}.imatrix.gguf"
    command = [
        str(LLAMA_IMATRIX),
        "--model", str(SOURCE_MODEL),
        "--file", str(text_file),
        "--output-file", str(output_file),
        "--n-gpu-layers", "99",
        "--parse-special",
        "--no-ppl",
    ]

    print(f"Creating {task} imatrix...")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    if not LLAMA_IMATRIX.exists():
        raise FileNotFoundError(f"Not found: {LLAMA_IMATRIX}")
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(f"Not found: {SOURCE_MODEL}")

    make_imatrix("chat", get_chat_examples())
    make_imatrix("math", get_math_examples())
    make_imatrix("code", get_code_examples())

    print("Done.")