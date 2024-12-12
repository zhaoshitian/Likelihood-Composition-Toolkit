import os
import sys
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="/home/ubuntu/stzhao/LLaVA/finetuned_llm_weight/LLaVA-Lightning-7B-vicuna-v1-1")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--conv-mode", type=str, default="llava_v1")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cuda", type=str, default=None)
parser.add_argument("--ds", type=str, default=None)
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--noise_step", type=int, default=None)
parser.add_argument("--multi_choices", type=str, default=False)
parser.add_argument("--use_instruction", type=str, default=False)
parser.add_argument("--instruction_prompt", type=str, default=None)
parser.add_argument("--negative", type=str, default=None)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

sys.path.insert(0,'/home/ubuntu/stzhao/LiCo')

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
from datasets import Dataset
import random
import numpy as np

# from llava.utils import disable_torch_init
# from llava.mm_utils import get_model_name_from_path



# from inference_engine.llava_interface import build as build_llava
# from inference_engine.internvl_llava_interface import build as build_internvl
# from inference_engine.yi_vl_interface import build as build_yivl
# from inference_engine.qwen_vl_interface import build as build_qwenvl
# from inference_engine.cogagent_interface import build as build_cogagent
from inference_engine.internlm_xcomposer_interface import build as build_internlmxcomposer
# from inference_engine.sphinx_interface import build as build_sphinx

import torch.backends.cudnn as cudnn

from PIL import Image
import math

ds_collections = {
    'mme': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/mme.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'mmbench_dev': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/mmbench_dev.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'seedbench': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/seedbench.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'pope_adversarial': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/pope_adversarial.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'pope_random': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/pope_random.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'pope_popular': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/pope_popular.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'vsr': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/vsr.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'winoground': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/winoground_all_10p_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'okvqa_val': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/okvqa_val_10p_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'vqav2_val': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/vqa_val_10p_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'vizwiz_val': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/vizwiz_val_10p_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'gqa_testdev': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/gqa_testdev_10p_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'clevr': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/clevr_core_multiple_choice_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'flower102_val': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/flower102_val_1p_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'imagenet1k_val': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/imagenet1k_val_10p_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'cifar10_test': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/cifar10_test_10p_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'aokvqa_val': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/aokvqa_val_10p_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'aokvqra_val': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/aokvqra_val_10p_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'scienceqa_test': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/scienceqa_test_10p_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
        'mmvp': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/test_data/mmvp.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
        'dev50': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/dev50/dev50_data.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
        'dev100': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/dev50/dev100_data.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
        'dev200': {
        'test_file_path': '/home/ubuntu/stzhao/LiCo/dev50/dev200_data.json',
        'record_logits_path': '/home/ubuntu/stzhao/LiCo/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/LiCo/reults',
        'max_new_length': 10,
        'prompt': {}
    },
}


all_options = ['A', 'B', 'C', 'D', 'E', 'F']

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

    
class VQADataset:

    def __init__(self, annotation_file_path):
        self.annotation_file = json.load(open(annotation_file_path, "r"))

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, idx):

        item = self.annotation_file[idx]
        x = {}
        # x['image'] = Image.open(item['image_path'])
        x['choices'] = item['choices']
        x['question'] = item['question']
        x['question_id'] = item['question_id']
        x['data_type'] = "image"
        x['data_path'] = item['image_path']
        x['gt_answers'] = item['gt_answers']

        return x
    

class DataCollator(object):

    def __init__(self):
        pass


    def __call__(self, x):


        data_batched = {}
        data_batched['choices'] = [sample['choices'] for sample in x]
        data_batched['question'] = [sample['question'] for sample in x]
        data_batched['question_id'] = [sample['question_id'] for sample in x]
        data_batched['data_path'] = [sample['data_path'] for sample in x]
        data_batched['data_type'] = [sample['data_type'] for sample in x]
        data_batched['gt_answers'] = [sample['gt_answers'] for sample in x]

        return data_batched





def setup_seeds(args):
    # seed = config.run_cfg.seed + get_rank()
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def inference(args):
    model = build(args)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    data = ds_collections[args.ds]
    annotation_file_path = data['test_file_path']
    if not os.path.exists(data['results_path']+"/directio"):
        os.makedirs(data['results_path']+"/directio")
    answers_file = data['results_path'] + f"/{str(args.ds)}.jsonl"
    annotation_file = json.load(open(annotation_file_path, "r"))
    ans_file = open(answers_file, "w")

    for item in tqdm(annotation_file):
        x = {}
        x['image'] = Image.open(item['image_path'])
        x['choices'] = item['choices']
        x['question'] = item['question']

        idx = item['question_id']
        x['data_type'] = "image"
        x['data_path'] = None

        losses = model(x)
        class_ranks = torch.argsort(losses, dim=-1).cpu()
        pred_id = all_options[class_ranks[0]]
        
        ans = {}
        ans['index'] = idx
        ans['pred'] = pred_id
        # ans['gt'] = all_options[item['choices'].index(item['gt_answers'])]
        ans['gt'] = item['gt_answers']

        ans = json.dumps(ans)
        ans_file.write(ans+"\n")

    ans_file.close()


def inference_debias(args):
    model = build(args)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    data = ds_collections[args.ds]
    annotation_file_path = data['test_file_path']
    # if not os.path.exists(data['results_path']+"/debias"):
    #     os.makedirs(data['results_path']+"/debias")
    # answers_file = os.path.join(data['record_logits_path'], model_name, args.ds) + f"/{str(args.ds)}.jsonl"
    annotation_file = json.load(open(annotation_file_path, "r"))
    # ans_file = open(answers_file, "w")

    logits_recorded = {}
    logits_recorded['logits_img'] = []
    logits_recorded['logits_ques'] = []
    logits_recorded['gt_answer'] = []
    logits_recorded['question_id'] = []
    logits_recorded['question'] = []
    logits_recorded['image_path'] = []

    for item in tqdm(annotation_file):
        x = {}
        x['image'] = Image.open(item['image_path'])
        x['choices'] = item['choices']
        x['question'] = item['question']

        idx = item['question_id']
        x['data_type'] = "image"
        x['data_path'] = None

        losses, logits_img, logits_ques = model.forward_debias(x)
        class_ranks = torch.argsort(losses, dim=-1).cpu()
        pred_id = all_options[class_ranks[0]]

        logits_recorded['logits_img'].append(logits_img)
        logits_recorded['logits_ques'].append(logits_ques)
        logits_recorded['gt_answer'].append(item['gt_answers'])
        logits_recorded['question_id'].append(idx)
        logits_recorded['question'].append(item['question'])
        logits_recorded['image_path'].append(item['image_path'])
        
        ans = {}
        ans['index'] = idx
        ans['pred'] = pred_id
        # ans['gt'] = all_options[item['choices'].index(item['gt_answers'])]
        ans['gt'] = item['gt_answers']

        ans = json.dumps(ans)
        # ans_file.write(ans+"\n")

    # ans_file.close()
    dataset = Dataset.from_dict(logits_recorded)
    save_file_path = os.path.join(data['record_logits_path'], model_name, args.ds)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    dataset.save_to_disk(save_file_path)


def inference_debias_batched_llava(args):
    model = build_llava(args)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    data = ds_collections[args.ds]
    annotation_file_path = data['test_file_path']

    vqadataset = VQADataset(annotation_file_path)
    datacollator = DataCollator()
    dataloader = DataLoader(vqadataset, batch_size=args.batchsize, shuffle=False, collate_fn=datacollator)

    logits_recorded = {}
    logits_recorded['logits_img'] = []
    logits_recorded['logits_ques'] = []
    logits_recorded['gt_answer'] = []
    logits_recorded['question_id'] = []
    logits_recorded['question'] = []
    logits_recorded['image_path'] = []

    for data_batched in tqdm(dataloader):

        if args.multi_choices == "True":
            losses, logits_img, logits_ques = model.forward_debias_batched_2(args, data_batched)
        else:
            losses, logits_img, logits_ques = model.forward_debias_batched(args, data_batched)
        logits_recorded['logits_img'].extend(logits_img)
        logits_recorded['logits_ques'].extend(logits_ques)
        logits_recorded['question_id'].extend(data_batched['question_id'])
        logits_recorded['gt_answer'].extend(data_batched['gt_answers'])
        logits_recorded['image_path'].extend(data_batched['data_path'])
        logits_recorded['question'].extend(data_batched['question'])


    dataset = Dataset.from_dict(logits_recorded)
    if args.noise_step is not None:
        model_name = model_name + f"_noisestep{args.noise_step}"
    if args.negative == "True":
        model_name = model_name + "_Negative"
    if args.use_instruction == "True":
        model_name = model_name + "_Instruction"
    if args.multi_choices == "True":
        model_name = model_name + "_MultiChoices"
    save_file_path = os.path.join(data['record_logits_path'], model_name, args.ds)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    dataset.save_to_disk(save_file_path)

def inference_debias_batched_internvl(args):
    model = build_internvl(args)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    data = ds_collections[args.ds]
    annotation_file_path = data['test_file_path']

    vqadataset = VQADataset(annotation_file_path)
    datacollator = DataCollator()
    dataloader = DataLoader(vqadataset, batch_size=args.batchsize, shuffle=False, collate_fn=datacollator)

    logits_recorded = {}
    logits_recorded['logits_img'] = []
    logits_recorded['logits_ques'] = []
    logits_recorded['gt_answer'] = []
    logits_recorded['question_id'] = []
    logits_recorded['question'] = []
    logits_recorded['image_path'] = []

    for data_batched in tqdm(dataloader):

        if args.multi_choices == "True":
            losses, logits_img, logits_ques = model.forward_debias_batched(args, data_batched)
        else:
            losses, logits_img, logits_ques = model.forward_debias_batched(args, data_batched)
        logits_recorded['logits_img'].extend(logits_img)
        logits_recorded['logits_ques'].extend(logits_ques)
        logits_recorded['question_id'].extend(data_batched['question_id'])
        logits_recorded['gt_answer'].extend(data_batched['gt_answers'])
        logits_recorded['image_path'].extend(data_batched['data_path'])
        logits_recorded['question'].extend(data_batched['question'])


    dataset = Dataset.from_dict(logits_recorded)
    if args.noise_step is not None:
        model_name = model_name + f"_noisestep{args.noise_step}"
    if args.negative == "True":
        model_name = model_name + "_Negative"
    if args.use_instruction == "True":
        model_name = model_name + "_Instruction"
    if args.multi_choices == "True":
        model_name = model_name + "_MultiChoices"
    save_file_path = os.path.join(data['record_logits_path'], model_name, args.ds)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    dataset.save_to_disk(save_file_path)

def inference_debias_batched_yivl(args):
    model = build_yivl(args)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    data = ds_collections[args.ds]
    annotation_file_path = data['test_file_path']

    vqadataset = VQADataset(annotation_file_path)
    datacollator = DataCollator()
    dataloader = DataLoader(vqadataset, batch_size=args.batchsize, shuffle=False, collate_fn=datacollator)

    logits_recorded = {}
    logits_recorded['logits_img'] = []
    logits_recorded['logits_ques'] = []
    logits_recorded['gt_answer'] = []
    logits_recorded['question_id'] = []
    logits_recorded['question'] = []
    logits_recorded['image_path'] = []

    for data_batched in tqdm(dataloader):

        if args.multi_choices == "True":
            losses, logits_img, logits_ques = model.forward_debias_batched_2(args, data_batched)
        else:
            losses, logits_img, logits_ques = model.forward_debias_batched(args, data_batched)
        logits_recorded['logits_img'].extend(logits_img)
        logits_recorded['logits_ques'].extend(logits_ques)
        logits_recorded['question_id'].extend(data_batched['question_id'])
        logits_recorded['gt_answer'].extend(data_batched['gt_answers'])
        logits_recorded['image_path'].extend(data_batched['data_path'])
        logits_recorded['question'].extend(data_batched['question'])


    dataset = Dataset.from_dict(logits_recorded)
    if args.noise_step is not None:
        model_name = model_name + f"_noisestep{args.noise_step}"
    if args.negative == "True":
        model_name = model_name + "_Negative"
    if args.use_instruction == "True":
        model_name = model_name + "_Instruction"
    if args.multi_choices == "True":
        model_name = model_name + "_MultiChoices"
    save_file_path = os.path.join(data['record_logits_path'], model_name, args.ds)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    dataset.save_to_disk(save_file_path)

def inference_debias_batched_qwenvl(args):
    model = build_qwenvl(args)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    data = ds_collections[args.ds]
    annotation_file_path = data['test_file_path']

    vqadataset = VQADataset(annotation_file_path)
    datacollator = DataCollator()
    dataloader = DataLoader(vqadataset, batch_size=args.batchsize, shuffle=False, collate_fn=datacollator)

    logits_recorded = {}
    logits_recorded['logits_img'] = []
    logits_recorded['logits_ques'] = []
    logits_recorded['gt_answer'] = []
    logits_recorded['question_id'] = []
    logits_recorded['question'] = []
    logits_recorded['image_path'] = []

    for data_batched in tqdm(dataloader):

        if args.multi_choices == "True":
            losses, logits_img, logits_ques = model.forward_debias_batched_2(args, data_batched)
        else:
            losses, logits_img, logits_ques = model.forward_debias_batched(args, data_batched)
        logits_img = [l.float() for l in logits_img]
        logits_ques = [l.float() for l in logits_ques]
        logits_recorded['logits_img'].extend(logits_img)
        logits_recorded['logits_ques'].extend(logits_ques)
        logits_recorded['question_id'].extend(data_batched['question_id'])
        logits_recorded['gt_answer'].extend(data_batched['gt_answers'])
        logits_recorded['image_path'].extend(data_batched['data_path'])
        logits_recorded['question'].extend(data_batched['question'])


    dataset = Dataset.from_dict(logits_recorded)
    if args.noise_step is not None:
        model_name = model_name + f"_noisestep{args.noise_step}"
    if args.negative == "True":
        model_name = model_name + "_Negative"
    if args.use_instruction == "True":
        model_name = model_name + "_Instruction"
    if args.multi_choices == "True":
        model_name = model_name + "_MultiChoices"
    save_file_path = os.path.join(data['record_logits_path'], model_name, args.ds)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    dataset.save_to_disk(save_file_path)

def inference_debias_batched_cogagent(args):
    model = build_cogagent(args)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    data = ds_collections[args.ds]
    annotation_file_path = data['test_file_path']

    vqadataset = VQADataset(annotation_file_path)
    datacollator = DataCollator()
    dataloader = DataLoader(vqadataset, batch_size=args.batchsize, shuffle=False, collate_fn=datacollator)

    logits_recorded = {}
    logits_recorded['logits_img'] = []
    logits_recorded['logits_ques'] = []
    logits_recorded['gt_answer'] = []
    logits_recorded['question_id'] = []
    logits_recorded['question'] = []
    logits_recorded['image_path'] = []

    for data_batched in tqdm(dataloader):

        if args.multi_choices == "True":
            losses, logits_img, logits_ques = model.forward_debias_batched_2(args, data_batched)
        else:
            losses, logits_img, logits_ques = model.forward_debias_batched(args, data_batched)
        logits_img = [l.float() for l in logits_img]
        logits_ques = [l.float() for l in logits_ques]
        logits_recorded['logits_img'].extend(logits_img)
        logits_recorded['logits_ques'].extend(logits_ques)
        logits_recorded['question_id'].extend(data_batched['question_id'])
        logits_recorded['gt_answer'].extend(data_batched['gt_answers'])
        logits_recorded['image_path'].extend(data_batched['data_path'])
        logits_recorded['question'].extend(data_batched['question'])


    dataset = Dataset.from_dict(logits_recorded)
    if args.noise_step is not None:
        model_name = model_name + f"_noisestep{args.noise_step}"
    if args.negative == "True":
        model_name = model_name + "_Negative"
    if args.use_instruction == "True":
        model_name = model_name + "_Instruction"
    if args.multi_choices == "True":
        model_name = model_name + "_MultiChoices"
    save_file_path = os.path.join(data['record_logits_path'], model_name, args.ds)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    dataset.save_to_disk(save_file_path)

def inference_debias_batched_internlmxcomposer(args):
    model = build_internlmxcomposer(args)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    data = ds_collections[args.ds]
    annotation_file_path = data['test_file_path']

    vqadataset = VQADataset(annotation_file_path)
    datacollator = DataCollator()
    dataloader = DataLoader(vqadataset, batch_size=args.batchsize, shuffle=False, collate_fn=datacollator)

    logits_recorded = {}
    logits_recorded['logits_img'] = []
    logits_recorded['logits_ques'] = []
    logits_recorded['gt_answer'] = []
    logits_recorded['question_id'] = []
    logits_recorded['question'] = []
    logits_recorded['image_path'] = []

    for data_batched in tqdm(dataloader):

        if args.multi_choices == "True":
            losses, logits_img, logits_ques = model.forward_debias_batched_2(args, data_batched)
        else:
            losses, logits_img, logits_ques = model.forward_debias_batched(args, data_batched)
        logits_img = [l.float() for l in logits_img]
        logits_ques = [l.float() for l in logits_ques]
        logits_recorded['logits_img'].extend(logits_img)
        logits_recorded['logits_ques'].extend(logits_ques)
        logits_recorded['question_id'].extend(data_batched['question_id'])
        logits_recorded['gt_answer'].extend(data_batched['gt_answers'])
        logits_recorded['image_path'].extend(data_batched['data_path'])
        logits_recorded['question'].extend(data_batched['question'])


    dataset = Dataset.from_dict(logits_recorded)
    if args.noise_step is not None:
        model_name = model_name + f"_noisestep{args.noise_step}"
    if args.negative == "True":
        model_name = model_name + "_Negative"
    if args.use_instruction == "True":
        model_name = model_name + "_Instruction"
    if args.multi_choices == "True":
        model_name = model_name + "_MultiChoices"
    save_file_path = os.path.join(data['record_logits_path'], model_name, args.ds)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    dataset.save_to_disk(save_file_path)


def inference_debias_batched_sphinx(args):
    model = build_sphinx(args)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    data = ds_collections[args.ds]
    annotation_file_path = data['test_file_path']

    vqadataset = VQADataset(annotation_file_path)
    datacollator = DataCollator()
    dataloader = DataLoader(vqadataset, batch_size=args.batchsize, shuffle=False, collate_fn=datacollator)

    logits_recorded = {}
    logits_recorded['logits_img'] = []
    logits_recorded['logits_ques'] = []
    logits_recorded['gt_answer'] = []
    logits_recorded['question_id'] = []
    logits_recorded['question'] = []
    logits_recorded['image_path'] = []

    for data_batched in tqdm(dataloader):

        if args.multi_choices == "True":
            losses, logits_img, logits_ques = model.forward_debias_batched_2(args, data_batched)
        else:
            losses, logits_img, logits_ques = model.forward_debias_batched(args, data_batched)
        logits_img = [l.float() for l in logits_img]
        logits_ques = [l.float() for l in logits_ques]
        logits_recorded['logits_img'].extend(logits_img)
        logits_recorded['logits_ques'].extend(logits_ques)
        logits_recorded['question_id'].extend(data_batched['question_id'])
        logits_recorded['gt_answer'].extend(data_batched['gt_answers'])
        logits_recorded['image_path'].extend(data_batched['data_path'])
        logits_recorded['question'].extend(data_batched['question'])


    dataset = Dataset.from_dict(logits_recorded)
    if args.noise_step is not None:
        model_name = model_name + f"_noisestep{args.noise_step}"
    if args.negative == "True":
        model_name = model_name + "_Negative"
    if args.use_instruction == "True":
        model_name = model_name + "_Instruction"
    if args.multi_choices == "True":
        model_name = model_name + "_MultiChoices"
    save_file_path = os.path.join(data['record_logits_path'], model_name, args.ds)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    dataset.save_to_disk(save_file_path)


if __name__ == "__main__":
    setup_seeds(args)

    start_time = time.time()

 
    print(args.ds)
    if "llava" in args.model_path or "LLaVA" in args.model_path:
        inference_debias_batched_llava(args)
    elif 'InternVL' in args.model_path:
        inference_debias_batched_internvl(args)
    elif "Yi-VL" in args.model_path:
        inference_debias_batched_yivl(args)
    elif "Qwen-VL" in args.model_path:
        inference_debias_batched_qwenvl(args)
    elif "cogagent" in args.model_path:
        inference_debias_batched_cogagent(args)
    elif "internlm-xcomposer" in args.model_path:
        inference_debias_batched_internlmxcomposer(args)
    elif "SPHINX" in args.model_path:
        inference_debias_batched_sphinx(args)

    end_time = time.time()
    run_time = (end_time - start_time) / 60
    print(f"time spent on evaluating {args.ds}: {run_time} minutes.")
