#!/bin/bash

#SBATCH --job-name=dpo            # The name of the job
#SBATCH --nodes=1                     # Request 1 compute node per job instance
#SBATCH --cpus-per-task=1             # Request 1 CPU per job instance
#SBATCH --mem=8GB                     # Request 2GB of RAM per job instance
#SBATCH --time=00:05:00               # Request 10 mins per job instance
#SBATCH --output=/scratch/spp9399/mia/output/out_%A_%a.out  # The output will be saved here. %A will be replaced by the slurm job ID, and %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --mail-user=spp9399@nyu.edu   # Email address
#SBATCH --mail-type=BEGIN,END               # Send an email when all the instances of this job are completed
#SBATCH --gres=gpu:rtx8000:1                    # requesting 2 GPUs

module purge                          # unload all currently loaded modules in the environment

export WANDB_PROJECT=llava-hound
export WANDB_NAME=dpo
export model_name_or_path=liuhaotian/llava-v1.5-7b
export data_path=/scratch/spp9399/mia/dpo_27k.json
export video_dir=/
export image_dir=/scratch/spp9399/mia/
export output_dir=/scratch/spp9399/mia/output
export lr=5e-5
export cache_dir=/scratch/spp9399/mia/cache

/scratch/spp9399/env/mia/run_env.sh pip install bitsandbytes==0.41.0
