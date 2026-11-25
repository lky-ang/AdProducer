# SFT (ms-swift)

Full-parameter fine-tune of Qwen3.5-9B (vision tower and projector frozen) on the 3,710 agentic trajectories in
`sft_data.json` of [AdProducer/AdProducer-Train-Data](https://huggingface.co/datasets/AdProducer/AdProducer-Train-Data).
Output: [AdProducer/AdProducer-SFT-9B](https://huggingface.co/AdProducer/AdProducer-SFT-9B).

## Environment

```bash
conda create -n adproducer-sft python=3.12 -y && conda activate adproducer-sft
pip install ms-swift==4.5.3 deepspeed flash-attn
```

Hardware: 4 GPUs (80 GB), DeepSpeed ZeRO-3 with sequence parallelism over the 4 GPUs.

## Data

```bash
hf download AdProducer/AdProducer-Train-Data --repo-type dataset --include "sft_data.json" --include "cached_image/*" --local-dir data
(cd data && for f in cached_image/cached_image_part*.zip; do unzip -q $f -d cached_image/; done)
```

## Train

```bash
DATA_DIR=data bash train_sft.sh                  # -> output/AdProducer-SFT-9B (OUTPUT_DIR to change)
```

Over-length samples are dropped rather than truncated; the loss covers assistant turns only, ignoring the think prefix.
Hyperparameters (in `train_sft.sh`): 2 epochs, learning rate 1e-5 with cosine decay and 5% warm-up, effective batch
size 16 (16 accumulation steps; the 4 GPUs share each sequence), weight decay 0.1, Adam β2 0.95, 65,536-token maximum length.
