#!/usr/bin/env bash
# Serve a GGUF from this repo's models/ folder. Paths anchor to this script.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$DIR/${1:-models/Llama-3.2-1B-Instruct-BF16.gguf}"
SERVER="${LLAMA_SERVER:-llama-server}"

[[ -f "$MODEL" ]] || { echo "Model not found: $MODEL" >&2; exit 1; }
command -v "$SERVER" >/dev/null || { echo "llama-server not on PATH" >&2; exit 1; }

exec "$SERVER" -m "$MODEL" --port "${2:-8080}" --n-gpu-layers 0 --cache-type-k f16 --cache-type-v f16
