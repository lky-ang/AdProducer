# RL training (slime, GRPO)

Agentic RL for AdProducer on [slime](https://github.com/THUDM/slime), starting from
[AdProducer-SFT-9B](https://huggingface.co/AdProducer/AdProducer-SFT-9B). The policy runs the full agent loop (think,
one tool call per round among `search` / `image_search` / `browse`, then a MiniMax H3 reference-to-video prompt with
reference images); each answer is rendered by MiniMax H3 and scored by a Gemini 3.1 Pro judge. The released
[AdProducer-RL-9B](https://huggingface.co/AdProducer/AdProducer-RL-9B) was trained with this recipe.

```
training/rl/
├── adproducer_rl.py        agent protocol + tools, slime rollout (generate), reward (reward_func), data CLI
├── train_rl.sh             slime launch script
├── setup.sh                environment setup inside the slime image (slime at the pinned commit, deps, ffmpeg)
├── config.yaml             max_turns: 9
├── prompts/                agent system prompt, video / text reward prompts, text-requirement prompt
├── scripts/                slime model script for Qwen3.5-9B (vision) and runtime patches for slime
├── requirements.txt
└── .env.example            tools, MiniMax H3 servers, judge
```

## 1. Environment

**Training container.** 4 GPUs (80 GB) run training and rollout (colocated):

```bash
docker run --gpus all -it -v /path/to/AdProducer:/root/AdProducer slimerl/slime:latest bash
export ADP_DIR=/root/AdProducer/training/rl SLIME_DIR=/root/slime
bash $ADP_DIR/setup.sh                            # slime at the pinned commit, Python deps, ffmpeg
cp $ADP_DIR/.env.example $ADP_DIR/.env            # fill in the keys and service addresses
```

**Services** (addresses go into `.env`):

| service | used for | variables |
|---|---|---|
| MiniMax H3 reference-to-video server(s), sglang `--model-variant ref2va` (see `inference/scripts/serve_h3.sh`), on separate GPUs | rendering each rollout's answer | `H3_URLS` |
| Gemini 3.1 Pro (Google AI Studio, or an OpenAI-compatible gateway) | the video and text rewards | `JUDGE_*` |
| Serper, Jina Reader, an OpenAI-compatible model for page summaries | the agent's tools | `SERPER_KEY_ID`, `JINA_API_KEYS`, `BROWSE_SUMMARY_*` |

## 2. Data and checkpoint

```bash
hf download AdProducer/AdProducer-SFT-9B --local-dir /root/models/AdProducer-SFT-9B
hf download AdProducer/AdProducer-Train-Data --repo-type dataset --local-dir /root/dl \
  --include "rl_prepared/*" --include "GT-Video/gt_video.zip" --include "cached_image/*"
mkdir -p $ADP_DIR/data && cp /root/dl/rl_prepared/* $ADP_DIR/data/
unzip -q /root/dl/GT-Video/gt_video.zip -d $ADP_DIR/data                       # -> data/gt_video/
for z in /root/dl/cached_image/*.zip; do unzip -q -n $z -d $ADP_DIR/data/cached_image; done

# HF -> Megatron checkpoint for --ref-load
cd $SLIME_DIR && source scripts/models/qwen3.5-9B-vl.sh
PYTHONPATH=/root/Megatron-LM:$SLIME_DIR python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/models/AdProducer-SFT-9B --save /root/models/AdProducer-SFT-9B_torch_dist
```

`rl_prepared/rl_train.jsonl` is the slime-ready form of `rl_data.json`; `python adproducer_rl.py prepare-data --help` shows how to rebuild it.

## 3. Train

```bash
set -a; source $ADP_DIR/.env; set +a
export ADP_DATA_DIR=$ADP_DIR/data
GEN_REWARD_TEXT_COEF=1 bash $ADP_DIR/train_rl.sh     # quick check: text reward only, no H3 calls
bash $ADP_DIR/train_rl.sh                            # full training
```

Hyperparameters (in `train_rl.sh` and `config.yaml`): GRPO with group size 6, 8 briefs per step, learning rate 1e-6,
clip 0.2 / 0.28, no KL or entropy term, temperature 0.7, top-p 1.0; at most 9 turns, 6,144 new tokens per turn,
65,536-token context.

## Reward

```
R = 0.5 R_video + 0.5 R_text                       (after a format gate: unusable answers get no reward)
R_video = 0.4 visual_correctness + 0.2 text_accuracy + 0.2 selling_point + 0.1 shot_design + 0.1 aesthetics
```

`R_video` judges the rendered video against the brief, the verified facts, the official images and the ground-truth
video (`prompts/reward_video.txt`, the same prompt as PG-Score); `R_text` judges the storyboard prompt and the chosen
reference images (`prompts/reward_text.txt`). When a brief needs no readable text, the weight of `text_accuracy` is
spread over the other dimensions.
