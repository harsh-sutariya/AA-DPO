#!/bin/bash

#SBATCH --job-name=dpo_inf            # The name of the job
#SBATCH --nodes=1                     # Request 1 compute node per job instance
#SBATCH --cpus-per-task=4             # Request 1 CPU per job instance
#SBATCH --mem=64GB                     # Request 2GB of RAM per job instance
#SBATCH --time=02:00:00               # Request 10 mins per job instance
#SBATCH --account=cs_ga_3033_102-2025sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --output=/scratch/spp9399/mia_slurm/out_%A_%a.out  # The output will be saved here. %A will be replaced by the slurm job ID, and %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --mail-user=spp9399@nyu.edu   # Email address
#SBATCH --mail-type=BEGIN,END               # Send an email when all the instances of this job are completed
#SBATCH --gres=gpu:a100:2                    # requesting 2 GPU, change --nproc_per_node based on this!

module purge                          # unload all currently loaded modules in the environment

export cache_dir=/scratch/spp9399/mia/cache


export image_dir=/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/
export data_path=/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/sequence_test_images.json
export output_file=/scratch/spp9399/MIA-DPO/inference_outputs/answer_mia_loss_model.json

# --model_name_or_path "/scratch/spp9399/llava_lora_our_loss" \
# --base_model_name_or_path "liuhaotian/llava-v1.5-7b" \

/scratch/spp9399/env/mia_env/run-cuda-12.2.2.bash python3 ./my_eval.py \
    --model_name_or_path "/scratch/spp9399/llava_lora_mia_loss" \
    --base_model_name_or_path "liuhaotian/llava-v1.5-7b" \
    --version v1 \
    --data_path ${data_path} \
    --image_folder ${image_dir} \
    --X "Image" --training_modal 'image' \
    --image_tower "/scratch/spp9399/models/clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --model_max_length 4096 \
    --lazy_preprocess True \
    --temperature 0.2 \
    --answers_file ${output_file} \
    --only_base_model False


export image_dir=/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/
export data_path=/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/sequence_test_images.json
export output_file=/scratch/spp9399/MIA-DPO/inference_outputs/answer_our_loss_model.json

/scratch/spp9399/env/mia_env/run-cuda-12.2.2.bash python3 ./my_eval.py \
    --model_name_or_path "/scratch/spp9399/llava_lora_our_loss" \
    --base_model_name_or_path "liuhaotian/llava-v1.5-7b" \
    --version v1 \
    --data_path ${data_path} \
    --image_folder ${image_dir} \
    --X "Image" --training_modal 'image' \
    --image_tower "/scratch/spp9399/models/clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --model_max_length 4096 \
    --lazy_preprocess True \
    --temperature 0.2 \
    --answers_file ${output_file} \
    --only_base_model False



export image_dir=/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/
export data_path=/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/sequence_test_images.json
export output_file=/scratch/spp9399/MIA-DPO/inference_outputs/answer_base_model.json

/scratch/spp9399/env/mia_env/run-cuda-12.2.2.bash python3 ./my_eval.py \
    --model_name_or_path "/scratch/spp9399/llava_lora_mia_loss" \
    --base_model_name_or_path "liuhaotian/llava-v1.5-7b" \
    --version v1 \
    --data_path ${data_path} \
    --image_folder ${image_dir} \
    --X "Image" --training_modal 'image' \
    --image_tower "/scratch/spp9399/models/clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --model_max_length 4096 \
    --lazy_preprocess True \
    --temperature 0.2 \
    --answers_file ${output_file} \
    --only_base_model True
