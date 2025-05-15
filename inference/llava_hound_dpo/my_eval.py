import os
import copy
import random
from dataclasses import dataclass, field
import json
from typing import Dict, Optional, Sequence, List, Any, Tuple, Union
import torch
import numpy as np

import transformers
import tokenizers
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

from llava.constants import DEFAULT_X_TOKEN, IGNORE_INDEX
from torch.utils.data import Dataset
from utils import load_jsonl, load_json

from llava import conversation as conversation_lib
conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
from llava.model import *
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from trl.trainer.utils import DPODataCollatorWithPadding

from tqdm import tqdm
from PIL import Image

from llava.conversation import conv_templates, SeparatorStyle
from peft import PeftModel

from torch.utils.data import DataLoader
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="liuhaotian/llava-v1.5-7b")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    X: Optional[List[str]] = field(default=None)
    image_tower: Optional[str] = field(default=None)
    video_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_x_start_end: bool = field(default=False)
    mm_use_x_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    bf16: bool = field(default=False)
    model_max_length: int = field(default=4096)
    base_model_name_or_path: Optional[str] = field(default=None)
    device: str = field(default="cuda")
    torch_dtype: str = field(default='bfloat16')

@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    training_modal: Optional[str] = field(default='video')
    num_sample: Optional[int] = field(default=None)
    conv_mode: str = field(default="llava_v1")

@dataclass
class DecoderArguments:
    temperature:  float = field(default=0)
    top_p: int = field(default=None)
    num_beams: int = field(default=1)
    max_new_tokens: int = field(default=128)
    answers_file: str = field(default="./answer.json")
    only_base_model: bool = field(default=False)

def load_data(data_args):
    if 'jsonl' in data_args.data_path:
        data_list = load_jsonl(data_args.data_path)
    else: 
        data_list = load_json(data_args.data_path)
    return data_list


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, X : str = None) -> Dict:
    conv = conv_templates["llava_v1"].copy() # hard coding rn!
    conv.append_message(conv.roles[0], sources)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize conversations

    # input_ids = torch.stack([tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX[X], return_tensors='pt') for prompt in conversations], dim=0)
    
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
    return input_ids


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_X: str = None
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """

    # conv = conv_templates[args.conv_mode].copy()
    # conv.append_message(conv.roles[0], qs) # role=human
    # conv.append_message(conv.roles[1], None) # role=assistant

    X = has_X if has_X is None else has_X.upper()
    return preprocess_v1(sources, tokenizer, X=X)

class DPODataset(Dataset):
    """Dataset for inference"""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):

        super(Dataset, self).__init__()
        list_data_dict = load_data(data_args)
        if data_args.num_sample is not None:
            list_data_dict = list_data_dict[:data_args.num_sample]

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.training_modal = data_args.training_modal

    def __len__(self):
        # return 20
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if any([x.lower() in sample for x in DEFAULT_X_TOKEN.keys()]) else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if any([x.lower() in sample for x in DEFAULT_X_TOKEN.keys()]) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        '''
        {
            'id': '....'
            'prompt': '<image>\n In Image1 Is there a snowman wearing a green scarf and hat in the background?',
            'image_path': '/mnt/bn/liangkeg/data/ruohongz/dpo_data/dpo_images/LRVInstruction-000000009569.jpg',
        }
        '''
        try:
            has_X = None
            data_dict = copy.deepcopy(self.list_data_dict[i]) # inplace modification following
            import re
            match = re.search(r'In Image(\d+)', data_dict['prompt'])
            if match:
                extracted_number = match.group(1)
                if type(data_dict['image']) == list:
                    if len(data_dict["image"]) > 1:
                        data_dict['target_number'] = int(extracted_number) - 1
                        data_dict['image_type'] = "sequence"
                    else:
                        data_dict['target_number'] = int(extracted_number) - 1
                        data_dict['image_type'] = "collage"
            else:
                data_dict['target_number'] = 0
                data_dict['image_type'] = "pip"

            if type(data_dict['image'])==str:
                image_file = data_dict['image']
                image_folder = self.data_args.image_folder
                processor = self.data_args.image_processor
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                prompt = data_dict['prompt']
                prompt = prompt.replace("<image>", "").strip()
                prompt = "<image>\n" + prompt
                data_dict['prompt'] = prompt
                has_X = 'image'

            elif type(data_dict['image'])==list:
                processed_images = []
                data_dims = []
                for image_file in data_dict['image']:
                    image_folder = self.data_args.image_folder
                    processor = self.data_args.image_processor
                    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                    w, l = image.size
                    l =  l // 448
                    w = w // 448
                    data_dims.append( [l, w] )

                    if self.data_args.image_aspect_ratio == 'pad':
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                        image = image.reshape(-1, 3, image.shape[-2], image.shape[-1])
                        processed_images.append(image)
                    else:
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                        image = image.reshape(-1, 3, image.shape[-2], image.shape[-1])
                        processed_images.append(image)
                image = torch.cat(processed_images)

                prompt = data_dict['prompt']
                data_dict['prompt'] = prompt
                has_X = 'image'
                data_dict['num_of_images'] = len(processed_images)
                data_dict["image_dims"] = data_dims

            data_dict['has_X'] = has_X
            if has_X == 'image':
                data_dict['image'] = image

            return data_dict
        except Exception as e:
            print(f'Error with {e}, {self.list_data_dict[i]}')
            return self.__getitem__(random.randint(0, self.__len__()-1))

@dataclass
class DPODataCollator(DPODataCollatorWithPadding):
    def collate(self, batch):
        # first, pad everything to the same length
        # input_ids, labels = tuple([instance[key] for instance in instances]
        #                           for key in ("input_ids", "labels"))
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id)
        # labels = torch.nn.utils.rnn.pad_sequence(labels,
        #                                          batch_first=True,
        #                                          padding_value=IGNORE_INDEX)
        # input_ids = input_ids[:, :self.tokenizer.model_max_length]
        # labels = labels[:, :self.tokenizer.model_max_length]
        # batch = dict(
        #     input_ids=input_ids,
        #     labels=labels,
        #     attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        # )
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                else:
                    continue

                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        
        return padded_batch


    def tokenize_batch_element(
        self,
        prompt: str,
        has_X: str = None
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        # import pdb; pdb.set_trace()
        batch = {}
        
        prompt_data_dict = preprocess(
            prompt,
            self.tokenizer,
            has_X=has_X
        )

        batch['prompt_ids'] = prompt_data_dict

        return batch
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []
        Xs, keys = [], []
        ids = []

        num_of_images, target_images = [], []
        image_dims = []
        image_type = []
        for feature in features:
            prompt = feature["prompt"]
            has_X = feature['has_X']
            Xs.append(feature[has_X])
            keys.append(has_X)
            ids.append( feature['id'] )

            num_of_image = feature['num_of_images']
            target_image = feature['target_number']
            num_of_images.append(num_of_image)
            target_images.append(target_image)
            image_dims.append( feature['image_dims'])
            image_type.append( feature['image_type'])
            
            batch_element = self.tokenize_batch_element(prompt, has_X=has_X)
            tokenized_batch.append(batch_element)

        # return collated batch
        padded_batch =  self.collate(tokenized_batch)
        padded_batch['images'] = Xs  # we do not change the key's name.
        padded_batch['keys'] = keys
        padded_batch['id'] = ids
        padded_batch['num_of_images'] = num_of_images
        padded_batch['target_number'] = target_images
        padded_batch['image_dims'] = image_dims
        padded_batch['image_type'] = image_type

        return padded_batch

def ratio_sequence(input_ids, attention, num_of_images, target_number, layer_number): 
    indices = torch.nonzero(input_ids[0] == -200, as_tuple=False).squeeze().tolist()
    if type(indices) == int:
        indices = [indices]
    query_input_token_length = len(input_ids[0])-indices[-1]-1
    new_indices = []
    for i, item in enumerate(indices):
        if i == 0:
            new_indices.append(item)
        else:
            new_indices.append(item-i+576*i)

    indices = new_indices
    input_attention = torch.mean(attention[0][layer_number][0], dim=0)
    output_token_length = len(attention)-1
    output_token_attention = []

    for i in range(output_token_length):
        output_token_attention.append(attention[i+1][layer_number][0])

    max_tesor_size = len(output_token_attention) + len(attention[0][layer_number][0][0])
    merged_tensor = torch.zeros(len(attention[0][layer_number][0]), max_tesor_size, max_tesor_size)

    merged_tensor[:, :len(attention[0][layer_number][0][0]), :len(attention[0][layer_number][0][0])] = input_attention

    for i in range(1, len(output_token_attention)+1):
        merged_tensor[:, len(attention[0][layer_number][0][0]) + i - 1, :len(attention[0][layer_number][0][0]) + i] = output_token_attention[i-1].squeeze(1)

    merged_tensor = torch.mean(merged_tensor, dim=0)
    image_tensors = []

    for i in range(num_of_images):
        image_tensors.append(merged_tensor[-(len(output_token_attention)):, indices[i]:indices[i]+576])

    mean_image_tensors = [torch.mean(image_tensor, dim=0) for image_tensor in image_tensors]
    reshape_image_tensors = [mean_image_tensor.reshape(24, 24) for mean_image_tensor in mean_image_tensors]
    tensors = [reshape_image_tensor.cpu() for reshape_image_tensor in reshape_image_tensors]
    means = [torch.mean(tensor).item() for tensor in tensors]
    ratio = means[target_number]/sum(means)

    final_tensor = torch.cat(tensors, dim = 1)
    final_tensor_np = final_tensor.numpy()

    # plt.figure(figsize=(10, 10))
    # cax = plt.imshow(final_tensor_np, cmap='viridis', interpolation='nearest', vmin=0, vmax=0.0012)
    # cbar = plt.colorbar(cax, fraction=0.036, pad=0.04, aspect=10)
    # cbar.set_label('Intensity')
    # plt.title('Heatmap of Merged Tensor Matrix')
    # plt.savefig("/scratch/spp9399/attn_2.png", dpi=900)

    return ratio


def ratio_collage( input_ids, attention, image_dims, block_index, layer_number):
    block_index = block_index + 1
    image_dims = image_dims[0]

    indices = torch.nonzero(input_ids[0] == -200, as_tuple=False).squeeze().tolist()

    if type(indices) == int:
        indices = [indices]
    query_input_token_length = len(input_ids[0])-indices[-1]-1
    new_indices = []
    for i, item in enumerate(indices):
        if i == 0:
            new_indices.append(item)
        else:
            new_indices.append(item-i+576*i)
    indices = new_indices
    # print(indices)

    ### input_attention 输入的图文的 attention
    input_attention = torch.mean(attention[0][layer_number][0], dim=0)
    ### 输出的 tokens 的长度
    output_token_length = len(attention)-1
    ### 输出的 tokens 的 attention
    output_token_attention = []
    for i in range(output_token_length):
        output_token_attention.append(attention[i+1][layer_number][0])

    max_tesor_size = len(output_token_attention) + len(attention[0][layer_number][0][0])
    merged_tensor = torch.zeros(len(attention[0][layer_number][0]), max_tesor_size, max_tesor_size)
    merged_tensor[:, :len(attention[0][layer_number][0][0]), :len(attention[0][layer_number][0][0])] = input_attention
    for i in range(1, len(output_token_attention)+1):
        merged_tensor[:, len(attention[0][layer_number][0][0]) + i - 1, :len(attention[0][layer_number][0][0]) + i] = output_token_attention[i-1].squeeze(1)
    merged_tensor = torch.mean(merged_tensor, dim=0)
    # print(merged_tensor.shape)

    ### 获取图像的 image_attention
    image_tensors = []
    for i in range(1):
        image_tensors.append(merged_tensor[-(len(output_token_attention)):, indices[i]:indices[i]+576])
    mean_image_tensors = [torch.mean(image_tensor, dim=0) for image_tensor in image_tensors]
    reshape_image_tensors = [mean_image_tensor.reshape(24, 24) for mean_image_tensor in mean_image_tensors]
    tensors = [reshape_image_tensor.cpu() for reshape_image_tensor in reshape_image_tensors]


    final_tensor = torch.cat(tensors, dim = 1)
    final_tensor_np = final_tensor.detach().numpy()


    rows, cols = image_dims

    # 计算每个块的大小
    block_height = final_tensor_np.shape[0] // rows
    block_width = final_tensor_np.shape[1] // cols

    # 处理不能整除的情况
    block_heights = [block_height] * rows
    block_widths = [block_width] * cols

    for i in range(final_tensor_np.shape[0] % rows):
        block_heights[i] += 1

    for j in range(final_tensor_np.shape[1] % cols):
        block_widths[j] += 1

    # 累计高度和宽度，确定块的边界
    height_cumsum = np.cumsum([0] + block_heights)
    width_cumsum = np.cumsum([0] + block_widths)
    # 确定指定块的行列位置
    row_idx = (block_index - 1) // cols
    col_idx = (block_index - 1) % cols

    # 切割指定的块
    block = final_tensor_np[height_cumsum[row_idx]:height_cumsum[row_idx + 1], width_cumsum[col_idx]:width_cumsum[col_idx + 1]]

    # 计算指定块的和和整个数组的和
    block_sum = np.sum(block)
    total_sum = np.sum(final_tensor_np)

    # 计算比例
    ratio = block_sum / total_sum


    return ratio

def ratio_pip(input_ids, attention, layer_number):
    ### 找到所有 <image> 的起始位置
    indices = torch.nonzero(input_ids[0] == -200, as_tuple=False).squeeze().tolist()
    ### 输入单张图 indices 是一个 int
    if type(indices) == int:
        indices = [indices]
    ### 计算最后一张图像之后的 query 的 tokens 长度
    query_input_token_length = len(input_ids[0])-indices[-1]-1
    ### 换算插入image token之后每张图的起始位置
    new_indices = []
    for i, item in enumerate(indices):
        if i == 0:
            new_indices.append(item)
        else:
            new_indices.append(item-i+576*i)
    indices = new_indices
    # print(indices)

    ### input_attention 输入的图文的 attention
    input_attention = torch.mean(attention[0][layer_number][0], dim=0)
    ### 输出的 tokens 的长度
    output_token_length = len(attention)-1
    ### 输出的 tokens 的 attention
    output_token_attention = []
    for i in range(output_token_length):
        output_token_attention.append(attention[i+1][layer_number][0])

    ### 计算 input tokens 和 outpu tokens 组成的总的 attention map 的大小
    max_tesor_size = len(output_token_attention) + len(attention[0][layer_number][0][0])
    ### 把所有 attention merge 在一
    merged_tensor = torch.zeros(len(attention[0][layer_number][0]), max_tesor_size, max_tesor_size)
    merged_tensor[:, :len(attention[0][layer_number][0][0]), :len(attention[0][layer_number][0][0])] = input_attention
    # 依次拼接大小为torch.Size([32, 1, N])的tensor
    for i in range(1, len(output_token_attention)+1):
        merged_tensor[:, len(attention[0][layer_number][0][0]) + i - 1, :len(attention[0][layer_number][0][0]) + i] = output_token_attention[i-1].squeeze(1)
    merged_tensor = torch.mean(merged_tensor, dim=0)
    # print(merged_tensor.shape)

    ### 获取图像的 image_attention
    image_tensors = []
    for i in range(1):
        image_tensors.append(merged_tensor[-(len(output_token_attention)):, indices[i]:indices[i]+576])
    mean_image_tensors = [torch.mean(image_tensor, dim=0) for image_tensor in image_tensors]
    reshape_image_tensors = [mean_image_tensor.reshape(24, 24) for mean_image_tensor in mean_image_tensors]
    tensors = [reshape_image_tensor.cpu() for reshape_image_tensor in reshape_image_tensors]


    final_tensor = torch.cat(tensors, dim = 1)
    final_tensor_np = final_tensor.detach().numpy()
    # print(final_tensor_np.shape)

    if final_tensor_np.shape != (24, 24):
        raise ValueError("Input array must be of shape 24x24")

    # 计算中间16x16区域的索引范围
    start = (24 - 16) // 2  # 计算起始索引
    end = start + 16         # 计算结束索引

    # 提取中间16x16区域
    center_region = final_tensor_np[start:end, start:end]

    # 计算中间16x16区域的和
    center_sum = np.sum(center_region)

    # 计算整个24x24数组的和
    total_sum = np.sum(final_tensor_np)
    # 计算比例
    ratio = center_sum / total_sum

    return ratio

def calculation_attention_ratio( input_ids, attention, num_of_images, target_number, image_type, image_dims, layer_number):
    input_ids = input_ids.unsqueeze(0)

    if image_type == "sequence":
        return ratio_sequence(input_ids, attention, num_of_images, target_number, layer_number)

    elif image_type == "collage":
        return ratio_collage(input_ids, attention, image_dims, target_number, layer_number)
    elif image_type == "pip":
        return ratio_pip(input_ids, attention, layer_number)
    else:
        raise ValueError("image type can be sequence, collage, pip but not " + image_type)

def evaluate(attn_implementation):

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, DecoderArguments))
    model_args, data_args, decoder_args = parser.parse_args_into_dataclasses()
    
    """
    print("Loading base model from: ", model_args.base_model_name_or_path )
    
    model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.base_model_name_or_path,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if model_args.bf16 else torch.float16),
        )

    # Loading Peft
    if model_args.model_name_or_path:
        print("Loading Peft Model from: ", model_args.model_name_or_path )

        model = PeftModel.from_pretrained( model, model_args.model_name_or_path )

        print("Mering LoRa with Base model")
        model = model.merge_and_unload()
    else:
        print("No PEFT Model")

    model = model.to(torch.bfloat16 if model_args.bf16 else torch.float16)
    model = model.to("cuda")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.base_model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    image_tower = model.get_image_tower()
    if image_tower is None:
        model.get_model().initialize_image_modules(
            model_args=model_args
        )
        image_tower = model.get_image_tower()
    if not image_tower.is_loaded:
        # print('load image tower')
        image_tower.load_model()


    image_tower.to(dtype=torch.bfloat16 if model_args.bf16 else torch.float16, device="cuda")

    model.image_tower = image_tower 

    data_args.image_processor = image_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

    model.initialize_X_tokenizer(model_args, tokenizer=tokenizer)


    """

    disable_torch_init()

    model_path = os.path.expanduser( model_args.model_name_or_path )
    model_name = get_model_name_from_path(model_path)

    # torch_dtype=(torch.bfloat16 if model_args.bf16 else torch.float16)
    tokenizer, model, processor, context_len = load_pretrained_model( model_path,  model_args.base_model_name_or_path, model_name,  bf16 = model_args.bf16, image_tower_model_args=model_args, base_model_only = decoder_args.only_base_model, device_map = "balanced")
    data_args.image_processor = processor['image']

    eval_dataset = DPODataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)

    collator = DPODataCollator(
            tokenizer,
            label_pad_token_id=IGNORE_INDEX,
            pad_token_id=tokenizer.pad_token_id,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,  # or more, but make sure generate() handles padding correctly
        collate_fn=collator
    )


    answers_file = os.path.expanduser(decoder_args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for batch in tqdm( eval_dataloader, desc="Evaluating"):
        
        prompt_ids = batch["prompt_ids"][0].unsqueeze(0).to( model.device )
        image = batch["images"][0].to( model.device )
        keys = batch['keys']

        idx = batch["id"][0]

        num_of_images = batch['num_of_images'][0]
        target_number = batch['target_number'][0]
        image_dims = batch['image_dims'][0]
        image_type = batch['image_type'][0]
        
        stop_str = conv_templates[data_args.conv_mode].sep if conv_templates[data_args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[data_args.conv_mode].sep2

        # Decoder args!
        with torch.inference_mode():
            output_ids = model.generate(
                prompt_ids,
                images=[image.to(dtype=torch.bfloat16 if model_args.bf16 else torch.float16).unsqueeze(0), keys],
                do_sample=True if decoder_args.temperature > 0 else False,
                temperature=decoder_args.temperature,
                top_p=decoder_args.top_p,
                num_beams=decoder_args.num_beams,
                max_new_tokens=decoder_args.max_new_tokens,
                use_cache=True,
                output_attentions=True, return_dict_in_generate=True)
            

            attention = output_ids.attentions
            output_ids = output_ids.sequences

        attn_ratios = []
        for lidx in range( 32 ):
            attention_ratios = calculation_attention_ratio( prompt_ids.squeeze(), attention, num_of_images, target_number, image_type, image_dims, lidx)
            attn_ratios.append(attention_ratios)

        input_token_len = prompt_ids.shape[1]
        n_diff_input_output = (prompt_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_file.write(json.dumps({"question_id": idx,
                                   "text": outputs,
                                   "attention_ratio": [float(x.detach().cpu()) if torch.is_tensor(x) else float(x) for x in attn_ratios]
                                   }
                        ) + "\n")
        # ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    evaluate(attn_implementation="flash_attention_2")
