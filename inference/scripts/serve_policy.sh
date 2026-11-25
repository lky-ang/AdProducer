#!/bin/bash
# Serve the AdProducer policy (Qwen3.5-9B, SFT + RL) on one GPU with sglang (OpenAI-compatible, :8000).
# usage: scripts/serve_policy.sh /path/to/checkpoint [gpu] [port]
set -eu
CKPT=$1; GPU=${2:-0}; PORT=${3:-8000}
CUDA_VISIBLE_DEVICES=$GPU python -m sglang.launch_server --model-path "$CKPT" --served-model-name adproducer \
  --context-length 65536 --mem-fraction-static 0.85 --port $PORT --host 127.0.0.1
# if the flashinfer JIT fails on your GPU, add: --attention-backend triton --sampling-backend pytorch
