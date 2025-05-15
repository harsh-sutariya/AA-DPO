#!/bin/bash

export WANDB_ENTITY=llvm_dpo
export WANDB_PROJECT=dpo_with_attn_loss
export WANDB_NAME=dpo_mia_a100_2_collage_sft_debug
export data_path=/scratch/spp9399/mia/dpo_27k.json
export video_dir=/
export image_dir=/scratch/spp9399/mia/
export output_dir=/scratch/spp9399/mia/output_a100_final_sft_debug
export lr=5e-5
export cache_dir=/scratch/spp9399/mia/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nnodes 1 --nproc_per_node 1 --node_rank $SLURM_PROCID --master_addr $(hostname) --master_port 12345 ./run_dpo.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./zero2.json \
    --model_name_or_path "/scratch/spp9399/models/llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234" \
    --dpo_alpha 0.0 --beta 0.1 --gamma 1.0 \
    --version v1 \
    --data_path ${data_path} \
    --video_folder ${video_dir} \
    --image_folder ${image_dir} \
    --X "Image" --training_modal 'image' \
    --image_tower "/scratch/spp9399/models/clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 11 \
    --learning_rate ${lr} --freeze_mm_mlp_adapter True \
    --weight_decay 0. --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --cache_dir ${cache_dir} \
    --report_to wandb
