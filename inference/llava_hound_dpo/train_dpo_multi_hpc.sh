#!/bin/bash

#SBATCH --requeue
#SBATCH --job-name=dpo            # The name of the job
#SBATCH --nodes=1                     # Request 1 compute node per job instance
#SBATCH --cpus-per-task=4             # Request 1 CPU per job instance
#SBATCH --mem=128GB                     # Request 2GB of RAM per job instance
#SBATCH --time=2-00:00:00               # Request 10 mins per job instance
#SBATCH --output=/scratch/spp9399/mia/output_logs/out_%A_%a.out  # The output will be saved here. %A will be replaced by the slurm job ID, and %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --mail-user=spp9399@nyu.edu   # Email address
#SBATCH --mail-type=BEGIN,END               # Send an email when all the instances of this job are completed
#SBATCH --account=cs_ga_3033_102-2025sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:a100:2                    # requesting 2 GPU, change --nproc_per_node based on this!

module purge                          # unload all currently loaded modules in the environment

export WANDB_ENTITY=llvm_dpo
export WANDB_PROJECT=dpo_with_attn_loss
export WANDB_NAME=dpo_mia_final
export model_name_or_path=liuhaotian/llava-v1.5-7b
export data_path=/scratch/spp9399/mia/dpo_27k_attn.json
export video_dir=/
export image_dir=/scratch/spp9399/mia/
export output_dir=/scratch/spp9399/mia/output_final
export lr=5e-5
export cache_dir=/scratch/spp9399/mia/cache

/scratch/spp9399/env/mia_env/run-cuda-12.2.2.bash torchrun --nnodes 1 --nproc_per_node 2 --node_rank $SLURM_PROCID --master_addr $(hostname) --master_port 12345 ./run_dpo.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./zero2.json \
    --model_name_or_path "liuhaotian/llava-v1.5-7b" \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0 \
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
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 100 \
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
