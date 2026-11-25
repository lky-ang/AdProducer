#!/bin/bash
# AdProducer SFT: full-parameter fine-tune of Qwen3.5-9B with ms-swift (4 x 80 GB GPUs, DeepSpeed ZeRO-3).
#   DATA_DIR: folder with sft_data.json and the unzipped cached_image/ (AdProducer/AdProducer-Train-Data)
#   bash train_sft.sh
set -e
DATA_DIR=${DATA_DIR:-data}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}

ROOT_IMAGE_DIR=${DATA_DIR} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CELOSS_PARALLEL_SIZE=2048 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
swift sft \
    --model ${MODEL} \
    --model_type qwen3_5 \
    --tuner_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --dataset ${DATA_DIR}/sft_data.json \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --gradient_checkpointing true \
    --sequence_parallel_size 4 \
    --max_length 65536 \
    --truncation_strategy delete \
    --loss_scale ignore_think_prefix \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --deepspeed zero3 \
    --eval_strategy no \
    --save_only_model true \
    --logging_steps 1 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --output_dir ${OUTPUT_DIR:-output/AdProducer-SFT-9B}
