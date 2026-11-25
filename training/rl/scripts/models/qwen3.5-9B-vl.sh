# Qwen3.5-9B with its native vision tower, for slime/Megatron.
# Copy to <slime>/scripts/models/qwen3.5-9B-vl.sh.  Same two-line override that qwen3.5-35B-A3B-vl.sh applies to
# qwen3.5-35B-A3B.sh; slime_plugins/models/qwen3_5_vl.py already branches on hf_config.model_type == "qwen3_5" (dense).
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/qwen3.5-9B.sh"

MODEL_ARGS[1]="slime_plugins.models.qwen3_5_vl"
MODEL_ARGS[2]="get_qwen3_5_vl_model_provider"
