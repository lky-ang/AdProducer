#!/bin/bash
# AdProducer RL: GRPO on slime, starting from AdProducer-SFT-9B.  Run inside the slime image after setup.sh:
#   export SLIME_DIR=/root/slime ADP_DIR=/path/to/AdProducer/training/rl
#   set -a; source $ADP_DIR/.env; set +a          # tools, MiniMax H3 servers, judge, data paths
#   bash $ADP_DIR/train_rl.sh
set -e
export PYTHONUNBUFFERED=1

SLIME_DIR=${SLIME_DIR:-/root/slime}
ADP_DIR=${ADP_DIR:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"}
cd "${SLIME_DIR}"
python3 "${ADP_DIR}/scripts/patch_slime.py" "${SLIME_DIR}"            # Qwen3.5-VL fixes for slime@540ea91
cp -n "${ADP_DIR}/scripts/models/qwen3.5-9B-vl.sh" "${SLIME_DIR}/scripts/models/" || true
source "${SLIME_DIR}/scripts/models/qwen3.5-9B-vl.sh"

SFT_CKPT=${SFT_CKPT:-/root/models/AdProducer-SFT-9B}                 # HF format (AdProducer/AdProducer-SFT-9B)
SFT_CKPT_DIST=${SFT_CKPT_DIST:-/root/models/AdProducer-SFT-9B_torch_dist}   # Megatron format (convert_hf_to_torch_dist.py)
TRAIN_JSONL=${TRAIN_JSONL:-${ADP_DIR}/data/rl_train.jsonl}
SAVE_DIR=${SAVE_DIR:-${ADP_DIR}/checkpoints/adproducer_rl}
NUM_GPUS=${NUM_GPUS:-4}

CKPT_ARGS=(--hf-checkpoint ${SFT_CKPT} --ref-load ${SFT_CKPT_DIST} --load ${SAVE_DIR} --save ${SAVE_DIR}
           --save-interval ${SAVE_INTERVAL:-1})

ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_JSONL} --input-key prompt --label-key label --metadata-key metadata
   --apply-chat-template --rollout-shuffle
   --num-rollout ${NUM_ROLLOUT:-125}            # one epoch over the 1,000 RL briefs (8 per step)
   --rollout-batch-size 8 --n-samples-per-prompt 6 --global-batch-size 48
   --rollout-max-context-len 65536 --rollout-max-response-len 61440
   --rollout-temperature 0.7 --rollout-top-p 1.0
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 2
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
   --use-dynamic-batch-size --max-tokens-per-gpu 34816 --log-probs-chunk-size 4096
)

GRPO_ARGS=(--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 --entropy-coef 0.00
           --eps-clip 0.2 --eps-clip-high 0.28)

OPTIMIZER_ARGS=(--optimizer adam --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
                --lr 1e-6 --lr-decay-style constant --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98)

SGLANG_ARGS=(--rollout-num-gpus-per-engine 2 --sglang-mem-fraction-static 0.6
             --sglang-attention-backend fa3 --sglang-mm-attention-backend fa3)

MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
           --attention-softmax-in-fp32 --attention-backend flash)

CUSTOM_ARGS=(
   --custom-generate-function-path adproducer_rl.generate     # multi-turn agent rollout with live tools
   --custom-rm-path adproducer_rl.reward_func                 # R = 0.5 R_video + 0.5 R_text
   --group-rm
   --custom-config-path ${ADP_DIR}/config.yaml
   --freeze-params-name-list visual                               # vision tower frozen, as in SFT
   --custom-rollout-log-function-path adproducer_rl.log_rollout_data
)

set +x   # the runtime environment below carries API keys
WANDB_ARGS=()
[ -n "${WANDB_API_KEY}" ] && WANDB_ARGS=(--use-wandb --wandb-project ${WANDB_PROJECT:-adproducer-rl} --wandb-group ${WANDB_GROUP:-adproducer_rl})

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats

RUNTIME_ENV_JSON="{\"env_vars\": {
  \"PYTHONPATH\": \"/root/Megatron-LM/:${SLIME_DIR}:${ADP_DIR}\",
  \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\", \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
  \"ADP_DATA_DIR\": \"${ADP_DATA_DIR:-${ADP_DIR}/data}\", \"ADP_WORK_DIR\": \"${ADP_WORK_DIR:-${ADP_DIR}/work}\",
  \"GEN_REWARD_TEXT_COEF\": \"${GEN_REWARD_TEXT_COEF:-0.5}\", \"GEN_MAX_NEW_TOKENS_PER_TURN\": \"6144\",
  \"ADP_IMAGE_MAX_PIXELS\": \"4194304\",
  \"SERPER_KEY_ID\": \"${SERPER_KEY_ID}\", \"JINA_API_KEYS\": \"${JINA_API_KEYS}\",
  \"BROWSE_SUMMARY_BASE_URL\": \"${BROWSE_SUMMARY_BASE_URL}\",
  \"BROWSE_SUMMARY_MODEL\": \"${BROWSE_SUMMARY_MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}\",
  \"BROWSE_SUMMARY_API_KEY\": \"${BROWSE_SUMMARY_API_KEY:-EMPTY}\",
  \"H3_URLS\": \"${H3_URLS}\", \"H3_URLS_DIR\": \"${H3_URLS_DIR}\", \"H3_SEED\": \"1101\", \"H3_SHORT_EDGE\": \"480\",
  \"H3_STEPS\": \"50\", \"H3_CONCURRENCY_PER_SERVER\": \"1\", \"H3_TIMEOUT_S\": \"${H3_TIMEOUT_S:-1800}\",
  \"JUDGE_BACKEND\": \"${JUDGE_BACKEND:-gemini}\", \"JUDGE_MODEL\": \"${JUDGE_MODEL:-gemini-3.1-pro-preview}\",
  \"JUDGE_API_KEY\": \"${JUDGE_API_KEY}\", \"JUDGE_BASE_URL\": \"${JUDGE_BASE_URL}\",
  \"JUDGE_AUTH_HEADER\": \"${JUDGE_AUTH_HEADER:-Authorization}\", \"WANDB_API_KEY\": \"${WANDB_API_KEY}\"
}}"

ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node ${NUM_GPUS} --rollout-num-gpus ${NUM_GPUS} --colocate \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} ${OPTIMIZER_ARGS[@]} ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} ${PERF_ARGS[@]} ${SGLANG_ARGS[@]} ${MISC_ARGS[@]} ${CUSTOM_ARGS[@]} ${EXTRA_ARGS}
