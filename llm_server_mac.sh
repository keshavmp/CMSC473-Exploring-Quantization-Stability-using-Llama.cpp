#!/bin/bash

MODEL="models/llama-2-7b-chat.Q4_K_M.gguf"

SERVER="../llama.cpp/build/bin/llama-server"

"$SERVER" \
    -m "$MODEL" \
    --port 8080 \
    --n-gpu-layers 0 \
    --cache-type-k f16 \
    --cache-type-v f16