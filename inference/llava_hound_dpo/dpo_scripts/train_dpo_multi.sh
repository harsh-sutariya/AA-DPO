#!/bin/bash
set -x

export WANDB_API_KEY="b052efff5cdec4c26696ba6da2dc0f7ce13cd2b8"

wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=2
export NNODES=1
export MASTER_PORT=29517
export CPUS_PER_TASK=4

# export WANDB_MODE=disabled
export WANDB_PROJECT=llava-hound
export WANDB_NAME=dpo
export model_name_or_path=liuhaotian/llava-v1.5-7b
export data_path=/scratch/spp9399/mia/dpo_27k.json
export video_dir=/
export image_dir=/scratch/spp9399/mia/images
export output_dir=/scratch/spp9399/mia/output
export lr=5e-5
export cache_dir=/scratch/spp9399/mia/cache


SRUN_ARGS=${SRUN_ARGS:-""}
srun -p mllm \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --gres=gpu:rtx8000:$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    --time=04:00:00 \
    --mail-type=BEGIN,END \
    --mail-user="spp9399@nyu.edu" \
    --output=/scratch/spp9399/mia/output/out_%A_%a.out
    ${SRUN_ARGS} \
    /scratch/spp9399/env/retrieval_heads/run_env.sh torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} dpo_scripts/run_dpo.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed config/zero2.json \
    --model_name_or_path ${model_name_or_path} \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0 \
    --version v1 \
    --data_path ${data_path} \
    --video_folder ${video_dir} \
    --image_folder ${image_dir} \
    --X "Image" --training_modal 'image' \
    --image_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_only_model True \
    --save_total_limit 11 \
    --learning_rate ${lr} --freeze_mm_mlp_adapter True \
    --weight_decay 0. --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --cache_dir ${cache_dir} \
    --report_to wandb
