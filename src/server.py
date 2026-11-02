"""Start llama.cpp on the GPU and download the model if needed."""

import re
import subprocess
import time
from pathlib import Path

import requests

from src.config import Config


def get_model_file(config: Config) -> Path:
    """Return the local GGUF or download it from Hugging Face once."""
    model_file = config.resolve_model_file()
    if model_file.exists():
        return model_file

    if not config.hf_repo or not config.hf_filename:
        raise FileNotFoundError(
            f"Model not found: {model_file}\n"
            "Put the GGUF there, or set hf_repo and hf_filename on the Config."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("Run: pip install huggingface-hub") from error

    print(f"Downloading {config.hf_filename} from {config.hf_repo}...")
    return Path(
        hf_hub_download(
            repo_id=config.hf_repo,
            filename=config.hf_filename,
            local_dir=model_file.parent,
        )
    )


def start_server(config: Config) -> tuple[subprocess.Popen, object]:
    """Start llama.cpp for one Config, wait until ready, confirm GPU offload."""
    if not config.llama_server:
        raise FileNotFoundError(f"llama-server not found: {config.llama_server}")

    model_file = get_model_file(config)
    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    log = open(config.log_file, "w+", encoding="utf-8")

    command = [
        str(config.llama_server),
        "--model", str(model_file),
        "--n-gpu-layers", config.gpu_layers,
        "--ctx-size", str(config.context_size * config.request_workers),
        "--parallel", str(config.request_workers),
        "--cont-batching",
        "--host", config.host,
        "-lv", str(config.server_log_verbosity),
        "--port", str(config.port),
    ]

    print("Starting llama.cpp")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

    deadline = time.time() + config.server_start_timeout
    while time.time() < deadline:
        if process.poll() is not None:
            log.close()
            raise RuntimeError(f"llama-server stopped. Check: {config.log_file}")

        try:
            if requests.get(f"{config.base_url}/health", timeout=2).ok:
                break
        except requests.RequestException:
            pass

        time.sleep(0.5)
    else:
        process.terminate()
        log.close()
        raise TimeoutError(
            f"llama-server did not start within {config.server_start_timeout} seconds."
        )

    time.sleep(1)
    log.flush()
    log.seek(0)
    startup_log = log.read()

    match = re.search(r"offloaded (\d+)/(\d+) layers to GPU", startup_log)
    if match and int(match.group(1)) > 0:
        print(f"Server Running on GPU ({match.group(1)}/{match.group(2)} layers)")
    elif match:
        print("Server Running on Cpu")
    else:
        print("GPU offload was not reported in the startup log.")

    print(f"Server ready: {config.base_url}")
    return process, log


def stop_server(process: subprocess.Popen, log: object) -> None:
    """Stop llama.cpp when evaluation finishes."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    log.close()
