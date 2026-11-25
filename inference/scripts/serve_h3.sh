#!/bin/bash
# Serve MiniMax H3 in Ref2VA mode on two GPUs with sglang (/v1/videos API, :30011).
# usage: scripts/serve_h3.sh /path/to/H3 [port]      (weights and install: see the MiniMax-H3 repository)
set -eu
H3=$1; PORT=${2:-30011}
sglang serve --model-path "$H3" --num-gpus 2 --ulysses-degree 2 --performance-mode speed --host 0.0.0.0 \
  --port $PORT --model-variant ref2va --use-fsdp-inference --encoder-parallel fold --pin-cpu-memory false
