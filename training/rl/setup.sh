#!/bin/bash
# One-time environment setup on the training machine (inside the slime docker image, slimerl/slime:latest).
# Only code / environment: no data or checkpoints are downloaded (see README.md for where they live).
#   ADP_DIR=/root/AdProducer/training/rl bash /root/AdProducer/training/rl/setup.sh
set -ex

ADP_DIR=${ADP_DIR:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"}
SLIME_DIR=${SLIME_DIR:-/root/slime}
SLIME_COMMIT=${SLIME_COMMIT:-540ea919e45b8b929198abce008a103be9cfc779}   # the commit this code was written against

# 1) slime at the tested commit
if [ ! -d "${SLIME_DIR}/.git" ]; then
  git clone https://github.com/THUDM/slime.git "${SLIME_DIR}"
fi
( cd "${SLIME_DIR}" && git fetch -q origin && git checkout "${SLIME_COMMIT}" && pip install -e . --no-deps )

# 2) python deps + ffmpeg
pip install -r "${ADP_DIR}/requirements.txt"
command -v ffmpeg >/dev/null || (apt-get update && apt-get install -y ffmpeg)

# 3) Qwen3.5-9B with its vision tower for slime
cp "${ADP_DIR}/scripts/models/qwen3.5-9B-vl.sh" "${SLIME_DIR}/scripts/models/"
