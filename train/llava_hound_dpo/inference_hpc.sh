#!/bin/bash

#SBATCH --job-name=dpo_inf            # The name of the job
#SBATCH --nodes=1                     # Request 1 compute node per job instance
#SBATCH --cpus-per-task=4             # Request 1 CPU per job instance
#SBATCH --mem=128GB                     # Request 2GB of RAM per job instance
#SBATCH --time=1-00:00:00               # Request 10 mins per job instance
#SBATCH --output=/scratch/spp9399/mia/output_logs/out_%A_%a.out  # The output will be saved here. %A will be replaced by the slurm job ID, and %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --mail-user=spp9399@nyu.edu   # Email address
#SBATCH --mail-type=BEGIN,END               # Send an email when all the instances of this job are completed
#SBATCH --gres=gpu:rtx8000:4                    # requesting 2 GPU, change --nproc_per_node based on this!

module purge                          # unload all currently loaded modules in the environment

export cache_dir=/scratch/spp9399/mia/cache

/scratch/spp9399/env/mia/run_env.sh python3 ./run_llava_eval.py --model-path "/scratch/spp9399/mia/llava_lora_our_loss" --model-base "liuhaotian/llava-v1.5-7b" --data_path "/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/sequence_test_images.json" --data_image "/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/" --output_json_path "/scratch/spp9399/mia_dpo_our_loss_seq.json"
