from huggingface_hub import hf_hub_download
from pathlib import Path
import json

# Use HF token only if secrets.json exists
HF_TOKEN = None

if Path("secrets.json").exists():
    with open("secrets.json", "r") as f:
        HF_TOKEN = json.load(f).get("HF_TOKEN")

REPO = "unsloth/Llama-3.2-1B-Instruct-GGUF"

MODELS = [
    "Llama-3.2-1B-Instruct-Q2_K.gguf",
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
]

Path("models").mkdir(exist_ok=True)

for model in MODELS:
    print(f"Downloading {model}")

    hf_hub_download(
        repo_id=REPO,
        filename=model,
        local_dir="models",
        token=HF_TOKEN,
    )

print("Done")