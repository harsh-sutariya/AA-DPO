export cache_dir=/scratch/spp9399/mia/cache

export image_dir=/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/
export data_path=/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/collage_test_images.json
export output_file=/scratch/spp9399/MIA-DPO/inference_outputs/collage_answer_mia_loss_model.json

# --model_name_or_path "/scratch/spp9399/llava_lora_our_loss" \
# --base_model_name_or_path "liuhaotian/llava-v1.5-7b" \

python3 ./my_eval.py \
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

export output_file=/scratch/spp9399/MIA-DPO/inference_outputs/collage_answer_our_loss_model.json

python3 ./my_eval.py \
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



export output_file=/scratch/spp9399/MIA-DPO/inference_outputs/collage_answer_base_model.json

python3 ./my_eval.py \
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
