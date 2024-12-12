import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from decord import VideoReader, cpu
import copy
import argparse
import json

from transformers.modeling_outputs import BaseModelOutput
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.nn.utils.rnn import pad_sequence
import pdb

from LLaVA_llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA_llava.conversation import conv_templates, SeparatorStyle
from LLaVA_llava.model.builder import load_pretrained_model
from LLaVA_llava.utils import disable_torch_init
from LLaVA_llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from utils import add_diffusion_noise

all_options = ['A', 'B', 'C', 'D', 'E', 'F']


def split_list_by_lengths(lst, lengths):
    # 初始化一个空列表来存储分割后的子列表
    result = []
    # 初始化当前索引
    index = 0

    # 遍历长度列表
    for length in lengths:
        # 从当前索引开始，取指定长度的子列表
        sub_list = lst[index:index + length]
        # 将子列表添加到结果列表
        result.append(sub_list)
        # 更新当前索引
        index += length

    return result


class MLLM_Tester(nn.Module):

    def __init__(self, args):
        super().__init__()

        disable_torch_init()
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        print(model_name)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
        self.model = model
        print(model.dtype)
        self.vis_processor = image_processor
        self.tokenizer = tokenizer
        self.args = args

    def process_prompt(self, args, config, question, choice):
        if config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], choice)
        prompt = conv.get_prompt()

        conv0 = conv_templates[args.conv_mode].copy()
        conv0.append_message(conv.roles[0], question)
        conv0.append_message(conv.roles[1], None)
        prompt0 = conv0.get_prompt()

        return prompt0, prompt

    def generate_text(self, x):

        img_path = x['img_path']
        instruction = x['instruction']

        raw_image = Image.open(open(img_path, "rb"))
        image = self.vis_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'].to(self.model.device)
        image = image.half()

        if self.model.config.mm_use_im_start_end:
            instruction = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + instruction
        else:
            instruction = DEFAULT_IMAGE_TOKEN + '\n' + instruction

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.no_grad():
            self.model.eval()
            output_ids = self.model.generate(
                input_ids,
                images=image,
                do_sample=True,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                max_new_tokens=1024,
                use_cache=True)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print(outputs)
        return outputs


def build(args):
    return MLLM_Tester(args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='/data/stzhao_data/mllm_ckpt/llava-v1.6-vicuna-13b')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=str, default='0,1')
    parser.add_argument("--ds", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--use_instruction", type=str, default=False)
    parser.add_argument("--instruction_prompt", type=str, default=None)
    parser.add_argument("--negative", type=str, default=None)
    args = parser.parse_args()

    model = build(args)

    instruction = 'Analyze the image in a comprehensive and detailed manner and provide a detailed description about the image.'

    img_path_list = []
    directory = '/home/ubuntu/stzhao/LiCo/llava_error/'
    entries = os.listdir(directory)
    for file_name in entries:
        img_path = directory + file_name
        x = {'img_path': img_path, 'instruction': instruction}
        outputs = model.generate_text(x)
        x['output'] = outputs
        img_path_list.append(x)
    # img_path = '/home/ubuntu/stzhao/LiCo/llava_error/001a2c4273ef8abc8eda3675b0d8125391192a13324a2169beef559e820b4786.jpg'
    # x = {'img_path': img_path, 'instruction': instruction}

    # outputs = model.generate_text(x)
    with open('/home/ubuntu/stzhao/LiCo/llava_error_results.json', 'w') as f:
        json.dump(img_path_list, f, indent=4)



