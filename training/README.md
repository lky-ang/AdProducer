# Training

AdProducer follows Gen-Searcher's two-stage recipe on Qwen3.5-9B: supervised cold start on agentic trajectories,
then agentic RL (GRPO) with a video reward.

| stage | folder | framework | output |
|---|---|---|---|
| 1. SFT | [`sft/`](sft/) | ms-swift 4.5.3, DeepSpeed ZeRO-3, 4 GPUs (80 GB), 2 epochs | [AdProducer/AdProducer-SFT-9B](https://huggingface.co/AdProducer/AdProducer-SFT-9B) |
| 2. RL | [`rl/`](rl/) | [slime](https://github.com/THUDM/slime) (GRPO), MiniMax H3 servers for rendering, Gemini 3.1 Pro judge | [AdProducer/AdProducer-RL-9B](https://huggingface.co/AdProducer/AdProducer-RL-9B) |

Training data (SFT trajectories, RL briefs, GT videos, cached images):
[AdProducer/AdProducer-Train-Data](https://huggingface.co/datasets/AdProducer/AdProducer-Train-Data).

- `sft/README.md`: environment, data and launch; `sft/train_sft.sh`: the `swift sft` command.
- `rl/README.md`: environment and services, data and checkpoint conversion, launch, reward; `rl/train_rl.sh`: the slime launch script.
- `rl/adproducer_rl.py` is the exact file used for training (slime custom rollout + reward). The evaluation copy
  in `inference/adproducer/core/` has the same agent protocol, tools and judge; it differs only in the H3 client
  (training adds a dynamic multi-server pool) and one training-log line.
