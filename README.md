# Attention-Aware DPO
Large Language Vision Models(LLVMs) have illustrated significant improvements across various multimodal tasks. To enhance the usability of LLVMs, preference alignment has become a standard technique. The Direct Preference Optimization (DPO) has emerged as a de-facto preference alignment algorithm which generally improves LLVMs single image performance, and LLVMs as a direct consequence struggle in multi-image scenarios. Previous studies have illustrated that LLVMs hallucinate when prompt contains containing multiple images and reference such as "In Image1". The misalignment can be improved using three broad solutions: (i) generating scalable dataset generation pipelines; (ii) improving alignment loss; and (iii) improving alignment at inference time. In this work, we present, Attention-aware Multi-Image Augmented Direct Preference Optimization, a preference alignment approach to handle multi-image inputs. Our improved alignment loss has shown promising results, with an improvement of 8.5\% in terms of accuracy over the base model. Lastly, we also tackle improving alignment at inference time, by extending the previous studies on adaptive attention scaling at inference time to multi-image inputs and see an improvement of 10\% over the base model.

# Training LLaVA1.5
To train LLaVA1.5 you can run the below mentioned file. Please specify the wandb constants, dataset folder, and model name/folder.
```bash
./train/llava_hound_dpo/train_dpo_multi_hpc_a100.sh
```
The training data can be found [here](https://huggingface.co/datasets/shaswat123/AA-DPO). You will require train.json(Prompt and QA pair) and images.zip(Image data).

If you want SFT + Attention-Aware loss then change dpo_alpha to 0 and gamma to 1.0. If you want to DPO + Attention-Aware loss then change dpo_alpha to 1.0 and gamma to 0.0.

# Inference
For inference, you can run the below mentioned file.
```
./inference/llava_hound_dpo/my_inference_hpc.sh
```

# Inference using AdaptViz
For optimizing inference using AdaptViz, run the below mentioned file
```
./inference_optimization/llava_hound_dpo/my_inference_adaptive.sh
```

# Data Generation
To generate the necessary training data, you need to first genereate the multi-image augmentation by running the files in data_gen/gen_instruction execpt pixmo.ipynb(this is used for test data generation). To generate DPO dataset, run the files under "data_gen/gen_chosen_rejected". You can find the testing data [here](https://huggingface.co/datasets/shaswat123/AA-DPO). You will require: test_images_{pip/collage/sequence}.tar and you can find the jsons(prompt + QA pair) under data_gen/gen_instruction with 3 jsons.

# Authors
Shaswat Patel, Harsh Sutaria, Jeet Patel, Vishvesh Trivedi

Department of Computer Science

New York University
