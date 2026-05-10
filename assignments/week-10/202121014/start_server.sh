#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="$HOME/models/deepseek-coder-v2-lite"
PORT=8000
GPU_UTIL=0.90
MAX_MODEL_LEN=32768

# Run the OpenAI-compatible vLLM server and tee logs to a local file.
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --served-model-name deepseek-coder-v2 \
  --trust-remote-code \
  2>&1 | tee vllm_server.log

