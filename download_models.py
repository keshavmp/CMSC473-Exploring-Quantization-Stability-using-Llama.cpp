"""Download the GGUF checkpoints. Run from the project folder."""

import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
if Path.cwd() != ROOT:
    raise SystemExit(f"Run this from the project folder:\n  {ROOT}")

REPO = "unsloth/Llama-3.2-1B-Instruct-GGUF"
MODELS = ["Llama-3.2-1B-Instruct-BF16.gguf"]


def hf_token():
    """Environment token, else secrets.json next to the code."""
    if token := os.environ.get("HF_TOKEN"):
        return token
    secrets = ROOT / "secrets.json"
    if not secrets.is_file():
        return None
    return json.loads(secrets.read_text(encoding="utf-8")).get("HF_TOKEN")


def main():
    models = ROOT / "models"
    models.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {models}")

    for model in MODELS:
        print(f"Downloading {model}")
        hf_hub_download(
            repo_id=REPO, filename=model, local_dir=str(models), token=hf_token()
        )

    print("Done")


if __name__ == "__main__":
    main()
