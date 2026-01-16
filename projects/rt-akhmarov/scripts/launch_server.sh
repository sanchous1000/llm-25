#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${PROJECT_DIR}/models"
PORT=8080

if [ -z "$1" ]; then
    echo "Usage: ./launch_server.sh [gemma|llama|qwen]"
    exit 1
fi

case $1 in
    gemma)
        MODEL_PATH="$MODEL_DIR/gemma-3-4b-it-q4_0.gguf"
        LLM_MODEL_NAME="gemma-3-4b"
        NGL=6
        ;;
    llama)
        MODEL_PATH="$MODEL_DIR/llama-3.2-3b-instruct-q4_k_m.gguf"
        LLM_MODEL_NAME="llama-3.2-3b-instruct"
        NGL=10
        ;;
    qwen)
        MODEL_PATH="$MODEL_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        LLM_MODEL_NAME="qwen2.5-1.5b"
        NGL=20
        ;;
    *)
        echo "Unknown model: $1"
        exit 1
        ;;
esac

# pkill llama-server || true
# sleep 1 

sed -i "/^LLM_MODEL_NAME=/d" .env
echo "LLM_MODEL_NAME='$LLM_MODEL_NAME'" >> .env
echo "Using model at $MODEL_PATH"
echo "Starting $1 server on port $PORT..."
llama-server \
    -m "$MODEL_PATH" \
    --port $PORT \
    --host "localhost" \
    -ngl $NGL \
    -c 1024 \
    --api-key "sk-no-key-required" \
    --cont-batching