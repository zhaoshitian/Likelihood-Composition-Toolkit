import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
import json
import argparse
from PIL import Image
from tqdm import tqdm
import random

import torch
from transformers.models.clip.modeling_clip import CLIPModel
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import random
from datasets import load_from_disk
from torch.nn import functional as F


# clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
# processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
# tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
# tokenizer_llava = AutoTokenizer.from_pretrained("/home/ubuntu/stzhao/LLaVA/finetuned_llm_weight/LLaVA-Lightning-7B-vicuna-v1-1")

def eval(pred_list, gts_list):
    nums_correct = sum([a==b for a,b in zip(pred_list, gts_list)])
    return nums_correct/len(gts_list)

def eval_mmvp(pred_list, gts_list):
    num_correct, num_total = 0, 0
    # Continue with the processing of the JSONL file
    index, round_correct = 0, 0
    for a,b  in zip(pred_list, gts_list):
        index += 1

        if a==b:
            round_correct += 1
        if index == 2:
            index = 0
            if round_correct == 2:
                num_correct += 1
            round_correct = 0

            num_total += 1
    return num_correct/num_total

def eval_mme(question_id_list, pred_list, gts_list):
    cognition_tasks = ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
    perception_tasks = ["existence", "count", "position", "color", "posters", "celebrity", "scene", "OCR", "artwork", "landmark"]
    eval_type_dict = {'Perception': perception_tasks, 'Cognition': cognition_tasks}

    total_score = 0
    for eval_type, task_name_list in eval_type_dict.items():
        print("===========", eval_type, "===========")
        
        scores = 0
        task_score_dict = dict()
        acc_dict = dict()
        acc_plus_dict = dict()

        for task_name in task_name_list:

            task_pred_list = []
            task_gt_list = []
            for j in range(len(question_id_list)):
                if task_name in question_id_list[j]:
                    task_pred_list.append(pred_list[j])
                    task_gt_list.append(gts_list[j])
            
            img_num = len(task_gt_list) / 2
            task_score = 0
            acc_correct_num = 0
            acc_plus_correct_num = 0

            index, round_correct = 0, 0
            for pred, gt in zip(task_pred_list, task_gt_list):
                    
                index += 1
                if pred == gt:
                    acc_correct_num += 1
                    round_correct += 1
                if index == 2:
                    index = 0
                    if round_correct == 2:
                        acc_plus_correct_num += 1
                    round_correct = 0

            task_acc = acc_correct_num / (img_num*2)
            task_acc_plus = acc_plus_correct_num / img_num
            
            acc_dict[task_name] = task_acc
            acc_plus_dict[task_name] = task_acc_plus
            
            task_score = task_acc*100 + task_acc_plus*100
            task_score_dict[task_name] = task_score
            
            scores += task_score

        total_score += scores
        print("total score:", scores)
        # for task_name, score in task_score_dict.items():
        #     print("\t", task_name, " score:", score, " acc:", acc_dict[task_name], " acc_plus:", acc_plus_dict[task_name])          
    return total_score

def softmax(x):
    # 对每个元素计算指数
    exp_x = np.exp(x*10)
    # 对每个样本或特征求和
    sum_exp_x = np.sum(exp_x, axis=0, keepdims=True)
    # 计算softmax值
    return exp_x / sum_exp_x

def normalize_list_data(data_list, method):
    """
    对Python列表中的数据进行标准化处理。

    参数:
    data_list: list, 输入的数据列表。
    method: str, 标准化方法，可选值包括 'min_max', 'z_score', 'l2'。

    返回:
    np.array, 标准化后的NumPy数组。
    """
    # 将列表转换为NumPy数组
    data_array = np.array(data_list, dtype=np.float64)

    # 根据方法进行标准化
    if method == 'min_max':
        # 最小-最大缩放
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        normalized_data = (data_array - min_val) / (max_val - min_val)
    elif method == 'z_score':
        # Z分数标准化
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        normalized_data = (data_array - mean_val) / std_val
    elif method == 'l2':
        # L2范数标准化
        norm = np.linalg.norm(data_array, axis=0, keepdims=True)
        normalized_data = data_array / (norm+1e-12)
    elif method == 'softmax':
        normalized_data = softmax(data_array)
    else:
        raise ValueError("Unknown normalization method. Choose from 'min_max', 'z_score', or 'l2'.")

    return normalized_data.tolist()

# # 示例使用
# data_list = [1, 2, 3, 4, 5]

# # 最小-最大缩放
# min_max_data = normalize_list_data(data_list, 'min_max')

# # Z分数标准化
# z_score_data = normalize_list_data(data_list, 'z_score')

# # L2范数标准化
# l2_data = normalize_list_data(data_list, 'l2')

# print("Min-Max Scaled Data:", min_max_data)
# print("Z-Score Standardized Data:", z_score_data)
# print("L2 Normalized Data:", l2_data)


def load_logits(sample_nums, logit_path, normalize_method):
    if logit_path is None:
        return None, None, None, None

    logits_recorded = load_from_disk(f"{logit_path}")
    logits_img_list=logits_recorded['logits_img']
    logits_ques_list=logits_recorded['logits_ques']
    gts_list=logits_recorded['gt_answer']
    question_ids_list=logits_recorded['question_id']

    if normalize_method is not None:
        logits_img_list = [normalize_list_data(l, normalize_method) for l in logits_img_list]
        logits_ques_list = [normalize_list_data(l, normalize_method) for l in logits_ques_list]

    return logits_img_list, logits_ques_list, gts_list, question_ids_list


def debias(args, c):

    all_options = ['A', 'B', 'C', 'D', 'E', 'F']

    logits_img_list, logits_ques_list, gts_list, question_ids_list = load_logits(args.sample_nums, args.logits_recorded_base_path_1, args.normalize_method)

    debias_list = [[(1+c)*a-c*b for a,b in zip(img,ques)] for img, ques in zip(logits_img_list, logits_ques_list)]
    debias_list = [torch.tensor(item) for item in debias_list]
    logits_img_list = [torch.tensor(item) for item in logits_img_list]

    pred_answer_list = []
    pred_answer_debias_list = []

    for debias_logit, logit in zip(debias_list, logits_img_list):
        
        class_ranks = torch.argsort(debias_logit, dim=-1, descending=True).cpu()
        pred_id_debias = all_options[class_ranks[0]]
        pred_answer_debias_list.append(pred_id_debias)

        class_ranks = torch.argsort(logit, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        # random_pred_id = random.choice(all_options[:len(class_ranks)])
        pred_answer_list.append(pred_id)
        
    if args.ds == "mmvp":
        print(eval_mmvp(pred_answer_list, gts_list), eval_mmvp(pred_answer_debias_list, gts_list), c)
    elif args.ds == 'mme':
        print(eval_mme(question_ids_list, pred_answer_list, gts_list), eval_mme(question_ids_list, pred_answer_debias_list, gts_list), c)
    else:
        print(eval(pred_answer_list, gts_list), eval(pred_answer_debias_list, gts_list), c)


def contrast(args, c):

    all_options = ['A', 'B', 'C', 'D', 'E', 'F']

    logits_img_list_1, logits_ques_list_1, gts_list_1, question_ids_list_1 = load_logits(args.sample_nums, args.logits_recorded_base_path_1, args.normalize_method)
    logits_img_list_2, logits_ques_list_2, gts_list_2, question_ids_list_2 = load_logits(args.sample_nums, args.logits_recorded_base_path_2, args.normalize_method)

    contrast_list = [[(1+c)*a-c*b for a,b in zip(img_1, img_2)] for img_1, img_2 in zip(logits_img_list_1, logits_img_list_2)]
    # contrast_list = [[(1+c)*a-c*b for a,b in zip(img_1, img_2)] for img_1, img_2 in zip(logits_img_list_1, logits_ques_list_2)]
    contrast_list = [torch.tensor(item) for item in contrast_list]
    logits_img_list_1 = [torch.tensor(item) for item in logits_img_list_1]

    pred_answer_list = []
    pred_answer_contrast_list = []

    for contrast_logit, logit in zip(contrast_list, logits_img_list_1):
        class_ranks = torch.argsort(contrast_logit, dim=-1, descending=True).cpu()
        pred_id_contrast = all_options[class_ranks[0]]
        pred_answer_contrast_list.append(pred_id_contrast)

        class_ranks = torch.argsort(logit, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list.append(pred_id)

    if args.ds == "mmvp":
        print(eval_mmvp(pred_answer_list, gts_list_1), eval_mmvp(pred_answer_contrast_list, gts_list_1), c)
    elif args.ds == 'mme':
        print(eval_mme(question_ids_list_1, pred_answer_list, gts_list_1), eval_mme(question_ids_list_1, pred_answer_contrast_list, gts_list_1), c)
    else:

        nums_correct = sum([a==b for a,b in zip(pred_answer_list, gts_list_1)])
        nums_correct_contrast = sum([a==b for a,b in zip(pred_answer_contrast_list, gts_list_1)])

        print(nums_correct/len(gts_list_1), nums_correct_contrast/len(gts_list_1), c)


def ensemble(args, CA=False):

    all_options = ['A', 'B', 'C', 'D', 'E', 'F']

    logits_img_list_1, logits_ques_list_1, gts_list_1, question_ids_list_1 = load_logits(args.sample_nums, args.logits_recorded_base_path_1, args.normalize_method)
    logits_img_list_2, logits_ques_list_2, gts_list_2, question_ids_list_2 = load_logits(args.sample_nums, args.logits_recorded_base_path_2, args.normalize_method)
    logits_img_list_3, logits_ques_list_3, gts_list_3, question_ids_list_3 = load_logits(args.sample_nums, args.logits_recorded_base_path_3, args.normalize_method)
    logits_img_list_4, logits_ques_list_4, gts_list_4, question_ids_list_4 = load_logits(args.sample_nums, args.logits_recorded_base_path_4, args.normalize_method)
    logits_img_list_5, logits_ques_list_5, gts_list_5, question_ids_list_5 = load_logits(args.sample_nums, args.logits_recorded_base_path_5, args.normalize_method)
    logits_img_list_6, logits_ques_list_6, gts_list_6, question_ids_list_6 = load_logits(args.sample_nums, args.logits_recorded_base_path_6, args.normalize_method)
    # print(logits_img_list_6)

    logit_ensemble_list = []
    for logit_list in [logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5, logits_img_list_6]:
        if logit_list is not None:
            logit_ensemble_list.append(logit_list)


    if args.nums_ensemble == 6:
        # contrast_list = [[sum([l1,l2,l3,l4,l5,l6]) for l1,l2,l3,l4,l5,l6 in zip(logit_1,logit_2,logit_3,logit_4,logit_5,logit_6)] for logit_1,logit_2,logit_3,logit_4,logit_5,logit_6 in zip(logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5, logits_img_list_6)]
        # contrast_list = [[sum([l1,l2,l3,l4,l5,l6]) for l1,l2,l3,l4,l5,l6 in zip(logit_1,logit_2,logit_3,logit_4,logit_5,logit_6)] for logit_1,logit_2,logit_3,logit_4,logit_5,logit_6 in zip(logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5, logits_img_list_6)]
        contrast_list = []
        for logit_1,logit_2,logit_3,logit_4,logit_5,logit_6 in zip(logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5, logits_img_list_6):
            logit_1 = torch.tensor(logit_1)
            logit_2 = torch.tensor(logit_2)
            logit_3 = torch.tensor(logit_3)
            logit_4 = torch.tensor(logit_4)
            logit_5 = torch.tensor(logit_5)
            logit_6 = torch.tensor(logit_6)
            sample_logit = [logit_1, logit_2, logit_3, logit_4, logit_5, logit_6]
            sample_logit = torch.stack(sample_logit, dim=0)
            if CA == True:
                max_logit_list = sample_logit.max(dim=1).values
                answer_indices_list = sample_logit.max(dim=1).indices
                mask = torch.zeros_like(sample_logit)
                for i, item in enumerate(answer_indices_list):
                    mask[i][item] = 1
                # min_max_logit = max_logit_list.min(dim=0).values
            # print(max_logit_list.shape)
            # print(max_logit_list)
                # min_logit_list = sample_logit.min(dim=1).values
                # threshold = torch.sort(max_logit_list).values[2]
                # mask = torch.where(max_logit_list > threshold, 1, 0)
                # print(mask)
                # sample_logit = (sample_logit * mask).sum(dim=0)
                sample_logit = mask.sum(dim=0)
            else:
                sample_logit = sample_logit.sum(dim=0)
            # print(sample_logit.shape)
            contrast_list.append(sample_logit.tolist())
        # contrast_list = contrast_list.tolist()
        # print(contrast_list)

    elif args.nums_ensemble == 5:
        contrast_list = [[sum([l1,l2,l3,l4,l5]) for l1,l2,l3,l4,l5 in zip(logit_1,logit_2,logit_3,logit_4,logit_5)] for logit_1,logit_2,logit_3,logit_4,logit_5 in zip(logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5)]
    elif args.nums_ensemble == 4:
        contrast_list = [[sum([l1,l2,l3,l4]) for l1,l2,l3,l4 in zip(logit_1,logit_2,logit_3,logit_4)] for logit_1,logit_2,logit_3,logit_4 in zip(logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4)]
    elif args.nums_ensemble == 3:
        contrast_list = [[sum([l1,l2,l3]) for l1,l2,l3 in zip(logit_1,logit_2,logit_3)] for logit_1,logit_2,logit_3 in zip(logits_img_list_1, logits_img_list_2, logits_img_list_3)]
    elif args.nums_ensemble == 2:
        contrast_list = [[sum([l1,l2]) for l1,l2 in zip(logit_1,logit_2)] for logit_1,logit_2 in zip(logits_img_list_1, logits_img_list_2)]

    # contrast_list = [[(1-c)*a+c*b for a,b in zip(img_1, img_2)] for img_1, img_2 in zip(logits_img_list_1, logits_img_list_2)]
    # contrast_list = [[(1+c)*a-c*b for a,b in zip(img_1, img_2)] for img_1, img_2 in zip(logits_img_list_1, logits_ques_list_2)]
    contrast_list = [torch.tensor(item) for item in contrast_list]
    logits_img_list_1 = [torch.tensor(item) for item in logits_img_list_1]
    logits_img_list_2 = [torch.tensor(item) for item in logits_img_list_2]
    logits_img_list_3 = [torch.tensor(item) for item in logits_img_list_3]
    logits_img_list_4 = [torch.tensor(item) for item in logits_img_list_4]
    logits_img_list_5 = [torch.tensor(item) for item in logits_img_list_5]
    logits_img_list_6 = [torch.tensor(item) for item in logits_img_list_6]

    pred_answer_list_1 = []
    pred_answer_list_2 = []
    pred_answer_list_3 = []
    pred_answer_list_4 = []
    pred_answer_list_5 = []
    pred_answer_list_6 = []
    pred_answer_contrast_list = []

    for (contrast_logit, logit_1,logit_2,logit_3,logit_4,logit_5,logit_6) in zip(contrast_list, logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5, logits_img_list_6):
        class_ranks = torch.argsort(contrast_logit, dim=-1, descending=True).cpu()
        pred_id_contrast = all_options[class_ranks[0]]
        pred_answer_contrast_list.append(pred_id_contrast)

        class_ranks = torch.argsort(logit_1, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_1.append(pred_id)

        class_ranks = torch.argsort(logit_2, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_2.append(pred_id)

        class_ranks = torch.argsort(logit_3, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_3.append(pred_id)

        class_ranks = torch.argsort(logit_4, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_4.append(pred_id)
    
        class_ranks = torch.argsort(logit_5, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_5.append(pred_id)

        class_ranks = torch.argsort(logit_6, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_6.append(pred_id)

    if args.ds == "mmvp":
        print(eval_mmvp(pred_answer_list_1, gts_list_1), 
              eval_mmvp(pred_answer_list_2, gts_list_1), 
              eval_mmvp(pred_answer_list_3, gts_list_1), 
              eval_mmvp(pred_answer_list_4, gts_list_1), 
              eval_mmvp(pred_answer_list_5, gts_list_1), 
              eval_mmvp(pred_answer_list_6, gts_list_1), 
              eval_mmvp(pred_answer_contrast_list, gts_list_1))
    elif args.ds == 'mme':
        print(eval_mme(question_ids_list_1, pred_answer_list_1, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_2, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_3, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_4, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_5, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_6, gts_list_1),  
              eval_mme(question_ids_list_1, pred_answer_contrast_list, gts_list_1))
    else:
        print(eval(pred_answer_list_1, gts_list_1), 
              eval(pred_answer_list_2, gts_list_1), 
              eval(pred_answer_list_3, gts_list_1), 
              eval(pred_answer_list_4, gts_list_1), 
              eval(pred_answer_list_5, gts_list_1), 
              eval(pred_answer_list_6, gts_list_1), 
              eval(pred_answer_contrast_list, gts_list_1))
        

def debias_ensemble(args, c):

    all_options = ['A', 'B', 'C', 'D', 'E', 'F']

    logits_img_list_1, logits_ques_list_1, gts_list_1, question_ids_list_1 = load_logits(args.sample_nums, args.logits_recorded_base_path_1, args.normalize_method)
    logits_img_list_2, logits_ques_list_2, gts_list_2, question_ids_list_2 = load_logits(args.sample_nums, args.logits_recorded_base_path_2, args.normalize_method)
    logits_img_list_3, logits_ques_list_3, gts_list_3, question_ids_list_3 = load_logits(args.sample_nums, args.logits_recorded_base_path_3, args.normalize_method)
    logits_img_list_4, logits_ques_list_4, gts_list_4, question_ids_list_4 = load_logits(args.sample_nums, args.logits_recorded_base_path_4, args.normalize_method)
    logits_img_list_5, logits_ques_list_5, gts_list_5, question_ids_list_5 = load_logits(args.sample_nums, args.logits_recorded_base_path_5, args.normalize_method)
    logits_img_list_6, logits_ques_list_6, gts_list_6, question_ids_list_6 = load_logits(args.sample_nums, args.logits_recorded_base_path_6, args.normalize_method)

    logits_img_list_1 = [[(1+c)*a-c*b for a,b in zip(img,ques)] for img, ques in zip(logits_img_list_1, logits_ques_list_1)]
    logits_img_list_2 = [[(1+c)*a-c*b for a,b in zip(img,ques)] for img, ques in zip(logits_img_list_2, logits_ques_list_2)]
    logits_img_list_3 = [[(1+c)*a-c*b for a,b in zip(img,ques)] for img, ques in zip(logits_img_list_3, logits_ques_list_3)]
    logits_img_list_4 = [[(1+c)*a-c*b for a,b in zip(img,ques)] for img, ques in zip(logits_img_list_4, logits_ques_list_4)]
    logits_img_list_5 = [[(1+c)*a-c*b for a,b in zip(img,ques)] for img, ques in zip(logits_img_list_5, logits_ques_list_5)]
    logits_img_list_6 = [[(1+c)*a-c*b for a,b in zip(img,ques)] for img, ques in zip(logits_img_list_6, logits_ques_list_6)]

    logit_ensemble_list = []
    for logit_list in [logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5, logits_img_list_6]:
        if logit_list is not None:
            logit_ensemble_list.append(logit_list)

    if args.nums_ensemble == 6:
        contrast_list = [[sum([l1,l2,l3,l4,l5,l6]) for l1,l2,l3,l4,l5,l6 in zip(logit_1,logit_2,logit_3,logit_4,logit_5,logit_6)] for logit_1,logit_2,logit_3,logit_4,logit_5,logit_6 in zip(logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5, logits_img_list_6)]
    elif args.nums_ensemble == 5:
        contrast_list = [[sum([l1,l2,l3,l4,l5]) for l1,l2,l3,l4,l5 in zip(logit_1,logit_2,logit_3,logit_4,logit_5)] for logit_1,logit_2,logit_3,logit_4,logit_5 in zip(logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5)]
    elif args.nums_ensemble == 4:
        contrast_list = [[sum([l1,l2,l3,l4]) for l1,l2,l3,l4 in zip(logit_1,logit_2,logit_3,logit_4)] for logit_1,logit_2,logit_3,logit_4 in zip(logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4)]
    elif args.nums_ensemble == 3:
        contrast_list = [[sum([l1,l2,l3]) for l1,l2,l3 in zip(logit_1,logit_2,logit_3)] for logit_1,logit_2,logit_3 in zip(logits_img_list_1, logits_img_list_2, logits_img_list_3)]
    elif args.nums_ensemble == 2:
        contrast_list = [[sum([l1,l2]) for l1,l2 in zip(logit_1,logit_2)] for logit_1,logit_2 in zip(logits_img_list_1, logits_img_list_2)]

    # contrast_list = [[(1-c)*a+c*b for a,b in zip(img_1, img_2)] for img_1, img_2 in zip(logits_img_list_1, logits_img_list_2)]
    # contrast_list = [[(1+c)*a-c*b for a,b in zip(img_1, img_2)] for img_1, img_2 in zip(logits_img_list_1, logits_ques_list_2)]
    contrast_list = [torch.tensor(item) for item in contrast_list]
    logits_img_list_1 = [torch.tensor(item) for item in logits_img_list_1]
    logits_img_list_2 = [torch.tensor(item) for item in logits_img_list_2]
    logits_img_list_3 = [torch.tensor(item) for item in logits_img_list_3]
    logits_img_list_4 = [torch.tensor(item) for item in logits_img_list_4]
    logits_img_list_5 = [torch.tensor(item) for item in logits_img_list_5]
    logits_img_list_6 = [torch.tensor(item) for item in logits_img_list_6]

    pred_answer_list_1 = []
    pred_answer_list_2 = []
    pred_answer_list_3 = []
    pred_answer_list_4 = []
    pred_answer_list_5 = []
    pred_answer_list_6 = []
    pred_answer_contrast_list = []

    for (contrast_logit, logit_1,logit_2,logit_3,logit_4,logit_5,logit_6) in zip(contrast_list, logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5, logits_img_list_6):
        class_ranks = torch.argsort(contrast_logit, dim=-1, descending=True).cpu()
        pred_id_contrast = all_options[class_ranks[0]]
        pred_answer_contrast_list.append(pred_id_contrast)

        class_ranks = torch.argsort(logit_1, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_1.append(pred_id)

        class_ranks = torch.argsort(logit_2, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_2.append(pred_id)

        class_ranks = torch.argsort(logit_3, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_3.append(pred_id)

        class_ranks = torch.argsort(logit_4, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_4.append(pred_id)
    
        class_ranks = torch.argsort(logit_5, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_5.append(pred_id)

        class_ranks = torch.argsort(logit_6, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_6.append(pred_id)

    if args.ds == "mmvp":
        print(eval_mmvp(pred_answer_list_1, gts_list_1), 
              eval_mmvp(pred_answer_list_2, gts_list_1), 
              eval_mmvp(pred_answer_list_3, gts_list_1), 
              eval_mmvp(pred_answer_list_4, gts_list_1), 
              eval_mmvp(pred_answer_list_5, gts_list_1), 
              eval_mmvp(pred_answer_list_6, gts_list_1), 
              eval_mmvp(pred_answer_contrast_list, gts_list_1), c)
    elif args.ds == 'mme':
        print(eval_mme(question_ids_list_1, pred_answer_list_1, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_2, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_3, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_4, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_5, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_6, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_contrast_list, gts_list_1),  c)
    else:
        print(eval(pred_answer_list_1, gts_list_1), 
              eval(pred_answer_list_2, gts_list_1), 
              eval(pred_answer_list_3, gts_list_1), 
              eval(pred_answer_list_4, gts_list_1), 
              eval(pred_answer_list_5, gts_list_1), 
              eval(pred_answer_list_6, gts_list_1), 
              eval(pred_answer_contrast_list, gts_list_1), c)


def composition(args):

    all_options = ['A', 'B', 'C', 'D', 'E', 'F']

    weight = np.load(args.weight_path)

    logits_img_list_1, logits_ques_list_1, gts_list_1, question_ids_list_1 = load_logits(args.sample_nums, args.logits_recorded_base_path_1, args.normalize_method)
    logits_img_list_2, logits_ques_list_2, gts_list_2, question_ids_list_2 = load_logits(args.sample_nums, args.logits_recorded_base_path_2, args.normalize_method)
    logits_img_list_3, logits_ques_list_3, gts_list_3, question_ids_list_3 = load_logits(args.sample_nums, args.logits_recorded_base_path_3, args.normalize_method)
    logits_img_list_4, logits_ques_list_4, gts_list_4, question_ids_list_4 = load_logits(args.sample_nums, args.logits_recorded_base_path_4, args.normalize_method)
    logits_img_list_5, logits_ques_list_5, gts_list_5, question_ids_list_5 = load_logits(args.sample_nums, args.logits_recorded_base_path_5, args.normalize_method)
    logits_img_list_6, logits_ques_list_6, gts_list_6, question_ids_list_6 = load_logits(args.sample_nums, args.logits_recorded_base_path_6, args.normalize_method)

    # contrast_list = [[sum(np.array([i1,q1,i2,q2,i3,q3,i4,q4,i5,q5,i6,q6])*weight) for i1,q1,i2,q2,i3,q3,i4,q4,i5,q5,i6,q6 in zip(logit_img_1,logit_ques_1,logit_img_2,logit_ques_2,logit_img_3,logit_ques_3,logit_img_4,logit_ques_4,logit_img_5,logit_ques_5,logit_img_6,logit_ques_6)] for logit_img_1,logit_ques_1,logit_img_2,logit_ques_2,logit_img_3,logit_ques_3,logit_img_4,logit_ques_4,logit_img_5,logit_ques_5,logit_img_6,logit_ques_6 in zip(logits_img_list_1, logits_ques_list_1, logits_img_list_2, logits_ques_list_2, logits_img_list_3, logits_ques_list_3, logits_img_list_4, logits_ques_list_4, logits_img_list_5, logits_ques_list_5, logits_img_list_6, logits_ques_list_6)]
    contrast_list = [[sum(np.array([i1,i2,i3,i4,i5,i6])*weight) for i1,i2,i3,i4,i5,i6 in zip(logit_img_1,logit_img_2,logit_img_3,logit_img_4,logit_img_5,logit_img_6)] for logit_img_1,logit_img_2,logit_img_3,logit_img_4,logit_img_5,logit_img_6 in zip(logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5, logits_img_list_6)]

    # contrast_list = [[(1-c)*a+c*b for a,b in zip(img_1, img_2)] for img_1, img_2 in zip(logits_img_list_1, logits_img_list_2)]
    # contrast_list = [[(1+c)*a-c*b for a,b in zip(img_1, img_2)] for img_1, img_2 in zip(logits_img_list_1, logits_ques_list_2)]
    contrast_list = [torch.tensor(item) for item in contrast_list]
    logits_img_list_1 = [torch.tensor(item) for item in logits_img_list_1]
    logits_img_list_2 = [torch.tensor(item) for item in logits_img_list_2]
    logits_img_list_3 = [torch.tensor(item) for item in logits_img_list_3]
    logits_img_list_4 = [torch.tensor(item) for item in logits_img_list_4]
    logits_img_list_5 = [torch.tensor(item) for item in logits_img_list_5]
    logits_img_list_6 = [torch.tensor(item) for item in logits_img_list_6]

    pred_answer_list_1 = []
    pred_answer_list_2 = []
    pred_answer_list_3 = []
    pred_answer_list_4 = []
    pred_answer_list_5 = []
    pred_answer_list_6 = []
    pred_answer_contrast_list = []

    for (contrast_logit, logit_1,logit_2,logit_3,logit_4,logit_5,logit_6) in zip(contrast_list, logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5, logits_img_list_6):
        class_ranks = torch.argsort(contrast_logit, dim=-1, descending=True).cpu()
        pred_id_contrast = all_options[class_ranks[0]]
        pred_answer_contrast_list.append(pred_id_contrast)

        class_ranks = torch.argsort(logit_1, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_1.append(pred_id)

        class_ranks = torch.argsort(logit_2, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_2.append(pred_id)

        class_ranks = torch.argsort(logit_3, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_3.append(pred_id)

        class_ranks = torch.argsort(logit_4, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_4.append(pred_id)
    
        class_ranks = torch.argsort(logit_5, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_5.append(pred_id)

        class_ranks = torch.argsort(logit_6, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_6.append(pred_id)

    if args.ds == "mmvp":
        print(eval_mmvp(pred_answer_list_1, gts_list_1), 
              eval_mmvp(pred_answer_list_2, gts_list_1), 
              eval_mmvp(pred_answer_list_3, gts_list_1), 
              eval_mmvp(pred_answer_list_4, gts_list_1), 
              eval_mmvp(pred_answer_list_5, gts_list_1), 
              eval_mmvp(pred_answer_list_6, gts_list_1), 
              eval_mmvp(pred_answer_contrast_list, gts_list_1))
    elif args.ds == 'mme':
        print(eval_mme(question_ids_list_1, pred_answer_list_1, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_2, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_3, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_4, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_5, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_6, gts_list_1),  
              eval_mme(question_ids_list_1, pred_answer_contrast_list, gts_list_1))
    else:
        print(eval(pred_answer_list_1, gts_list_1), 
              eval(pred_answer_list_2, gts_list_1), 
              eval(pred_answer_list_3, gts_list_1), 
              eval(pred_answer_list_4, gts_list_1), 
              eval(pred_answer_list_5, gts_list_1), 
              eval(pred_answer_list_6, gts_list_1), 
              eval(pred_answer_contrast_list, gts_list_1))

def add_mask(logit):
    mask = torch.zeros_like(logit)
    max_indices = logit.max(dim=-1).indices
    mask[max_indices] = 1
    # return logit*mask
    return logit


def debias_highlight(args):

    all_options = ['A', 'B', 'C', 'D', 'E', 'F']

    weight = np.load(args.weight_path)

    logits_img_list_1, logits_ques_list_1, gts_list_1, question_ids_list_1 = load_logits(args.sample_nums, args.logits_recorded_base_path_1, args.normalize_method)
    logits_img_list_2, logits_ques_list_2, gts_list_2, question_ids_list_2 = load_logits(args.sample_nums, args.logits_recorded_base_path_2, args.normalize_method)
    logits_img_list_3, logits_ques_list_3, gts_list_3, question_ids_list_3 = load_logits(args.sample_nums, args.logits_recorded_base_path_3, args.normalize_method)
    logits_img_list_4, logits_ques_list_4, gts_list_4, question_ids_list_4 = load_logits(args.sample_nums, args.logits_recorded_base_path_4, args.normalize_method)
    logits_img_list_5, logits_ques_list_5, gts_list_5, question_ids_list_5 = load_logits(args.sample_nums, args.logits_recorded_base_path_5, args.normalize_method)
    logits_img_list_6, logits_ques_list_6, gts_list_6, question_ids_list_6 = load_logits(args.sample_nums, args.logits_recorded_base_path_6, args.normalize_method)

    neg_logits_img_list_1, neg_logits_ques_list_1, neg_gts_list_1, neg_question_ids_list_1 = load_logits(args.sample_nums, args.logits_recorded_base_path_7, args.normalize_method)
    neg_logits_img_list_2, neg_logits_ques_list_2, neg_gts_list_2, neg_question_ids_list_2 = load_logits(args.sample_nums, args.logits_recorded_base_path_8, args.normalize_method)
    neg_logits_img_list_3, neg_logits_ques_list_3, neg_gts_list_3, neg_question_ids_list_3 = load_logits(args.sample_nums, args.logits_recorded_base_path_9, args.normalize_method)
    neg_logits_img_list_4, neg_logits_ques_list_4, neg_gts_list_4, neg_question_ids_list_4 = load_logits(args.sample_nums, args.logits_recorded_base_path_10, args.normalize_method)
    neg_logits_img_list_5, neg_logits_ques_list_5, neg_gts_list_5, neg_question_ids_list_5 = load_logits(args.sample_nums, args.logits_recorded_base_path_11, args.normalize_method)
    neg_logits_img_list_6, neg_logits_ques_list_6, neg_gts_list_6, neg_question_ids_list_6 = load_logits(args.sample_nums, args.logits_recorded_base_path_12, args.normalize_method)

    # contrast_list = [[sum(np.array([i1,q1,i2,q2,i3,q3,i4,q4,i5,q5,i6,q6])*weight) for i1,q1,i2,q2,i3,q3,i4,q4,i5,q5,i6,q6 in zip(logit_img_1,logit_ques_1,logit_img_2,logit_ques_2,logit_img_3,logit_ques_3,logit_img_4,logit_ques_4,logit_img_5,logit_ques_5,logit_img_6,logit_ques_6)] for logit_img_1,logit_ques_1,logit_img_2,logit_ques_2,logit_img_3,logit_ques_3,logit_img_4,logit_ques_4,logit_img_5,logit_ques_5,logit_img_6,logit_ques_6 in zip(logits_img_list_1, logits_ques_list_1, logits_img_list_2, logits_ques_list_2, logits_img_list_3, logits_ques_list_3, logits_img_list_4, logits_ques_list_4, logits_img_list_5, logits_ques_list_5, logits_img_list_6, logits_ques_list_6)]
    ensemble_list = [[sum([l1,l2,l3,l4,l5,l6]) for l1,l2,l3,l4,l5,l6 in zip(logit_1,logit_2,logit_3,logit_4,logit_5,logit_6)] for logit_1,logit_2,logit_3,logit_4,logit_5,logit_6 in zip(logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5, logits_img_list_6)]
    f_list = []
    for l1,l2,l3,l4,l5,l6,q1,q2,q3,q4,q5,q6,l7,l8,l9,l10,l11,l12 in zip(logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5, logits_img_list_6,logits_ques_list_1,logits_ques_list_2,logits_ques_list_3,logits_ques_list_4,logits_ques_list_5,logits_ques_list_6,neg_logits_img_list_1, neg_logits_img_list_2, neg_logits_img_list_3, neg_logits_img_list_4, neg_logits_img_list_5, neg_logits_img_list_6):
        l1 = torch.tensor(l1)
        l2 = torch.tensor(l2)
        l3 = torch.tensor(l3)
        l4 = torch.tensor(l4)
        l5 = torch.tensor(l5)
        l6 = torch.tensor(l6)
        q1 = torch.tensor(q1)
        q2 = torch.tensor(q2)
        q3 = torch.tensor(q3)
        q4 = torch.tensor(q4)
        q5 = torch.tensor(q5)
        q6 = torch.tensor(q6)
        l7 = torch.tensor(l7)
        l8 = torch.tensor(l8)
        l9 = torch.tensor(l9)
        l10 = torch.tensor(l10)
        l11 = torch.tensor(l11)
        l12 = torch.tensor(l12)

        c = 0.5
        e = 0.1

        a1 = 1
        a2 = 1
        # f = (l1+l2+l3)*(1+c)-c*(q1+q2+q3)+(l4+l5+l6)*(1+e)-(l10+l11+l12)*e
        # f = (l1+l2)*(1+c)-c*(q2+q3)+(l3+l4+l5+l6)*(1+e)-(l9+l10+l11+l12)*e
        # f1 = (l1+l2+l3+l4+l5+l6)*(1+e) - (l7+l8+l8+l10+l11+l12)*e
        # f2 = (l1+l2+l3+l4+l5+l6)*(1+c) - (q1+q2+q3+q4+q5+q6)*c
        if args.nums_ensemble == 6:
            f1 = add_mask((l1*(1+c)-q1*c))+add_mask((l2*(1+c)-q2*c))+add_mask((l3*(1+c)-q3*c))+add_mask((l4*(1+c)-q4*c))+add_mask((l5*(1+c)-q5*c))+add_mask((l6*(1+c)-q6*c))
            f2 = add_mask((l1*(1+e)-l7*e))+add_mask((l2*(1+e)-l8*e))+add_mask((l3*(1+e)-l9*e))+add_mask((l4*(1+e)-l10*e))+add_mask((l5*(1+e)-l11*e))+add_mask((l6*(1+e)-l12*e))
            f_list.append(a1*f1+a2*f2)
        elif args.nums_ensemble == 5:
            f1 = add_mask((l1*(1+c)-q1*c))+add_mask((l2*(1+c)-q2*c))+add_mask((l3*(1+c)-q3*c))+add_mask((l4*(1+c)-q4*c))+add_mask((l5*(1+c)-q5*c))
            f2 = add_mask((l1*(1+e)-l7*e))+add_mask((l2*(1+e)-l8*e))+add_mask((l3*(1+e)-l9*e))+add_mask((l4*(1+e)-l10*e))+add_mask((l5*(1+e)-l11*e))
            f_list.append(a1*f1+a2*f2)
        elif args.nums_ensemble == 4:
            f1 = add_mask((l1*(1+c)-q1*c))+add_mask((l2*(1+c)-q2*c))+add_mask((l3*(1+c)-q3*c))+add_mask((l4*(1+c)-q4*c))
            f2 = add_mask((l1*(1+e)-l7*e))+add_mask((l2*(1+e)-l8*e))+add_mask((l3*(1+e)-l9*e))+add_mask((l4*(1+e)-l10*e))
            f_list.append(a1*f1+a2*f2)
        elif args.nums_ensemble == 3:
            f1 = add_mask((l1*(1+c)-q1*c))+add_mask((l2*(1+c)-q2*c))+add_mask((l3*(1+c)-q3*c))
            f2 = add_mask((l1*(1+e)-l7*e))+add_mask((l2*(1+e)-l8*e))+add_mask((l3*(1+e)-l9*e))
            f_list.append(a1*f1+a2*f2)
        elif args.nums_ensemble == 2:
            f1 = add_mask((l1*(1+c)-q1*c))+add_mask((l2*(1+c)-q2*c))
            f2 = add_mask((l1*(1+e)-l7*e))+add_mask((l2*(1+e)-l8*e))
            f_list.append(a1*f1+a2*f2)
        elif args.nums_ensemble == 1:
            f1 = add_mask((l1*(1+c)-q1*c))
            f2 = add_mask((l1*(1+e)-l7*e))
            f_list.append(a1*f1+a2*f2)

    # contrast_list = [[(1-c)*a+c*b for a,b in zip(img_1, img_2)] for img_1, img_2 in zip(logits_img_list_1, logits_img_list_2)]
    # contrast_list = [[(1+c)*a-c*b for a,b in zip(img_1, img_2)] for img_1, img_2 in zip(logits_img_list_1, logits_ques_list_2)]
    ensemble_list = [torch.tensor(item) for item in ensemble_list]
    logits_img_list_1 = [torch.tensor(item) for item in logits_img_list_1]
    logits_img_list_2 = [torch.tensor(item) for item in logits_img_list_2]
    logits_img_list_3 = [torch.tensor(item) for item in logits_img_list_3]
    logits_img_list_4 = [torch.tensor(item) for item in logits_img_list_4]
    logits_img_list_5 = [torch.tensor(item) for item in logits_img_list_5]
    logits_img_list_6 = [torch.tensor(item) for item in logits_img_list_6]

    pred_answer_list_1 = []
    pred_answer_list_2 = []
    pred_answer_list_3 = []
    pred_answer_list_4 = []
    pred_answer_list_5 = []
    pred_answer_list_6 = []
    pred_answer_ensemble_list = []
    pred_answer_f_list = []

    for (contrast_logit, f_logit, logit_1,logit_2,logit_3,logit_4,logit_5,logit_6) in zip(ensemble_list, f_list, logits_img_list_1, logits_img_list_2, logits_img_list_3, logits_img_list_4, logits_img_list_5, logits_img_list_6):
        class_ranks = torch.argsort(contrast_logit, dim=-1, descending=True).cpu()
        pred_id_ensemble = all_options[class_ranks[0]]
        pred_answer_ensemble_list.append(pred_id_ensemble)

        class_ranks = torch.argsort(f_logit, dim=-1, descending=True).cpu()
        pred_id_f = all_options[class_ranks[0]]
        pred_answer_f_list.append(pred_id_f)

        class_ranks = torch.argsort(logit_1, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_1.append(pred_id)

        class_ranks = torch.argsort(logit_2, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_2.append(pred_id)

        class_ranks = torch.argsort(logit_3, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_3.append(pred_id)

        class_ranks = torch.argsort(logit_4, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_4.append(pred_id)
    
        class_ranks = torch.argsort(logit_5, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_5.append(pred_id)

        class_ranks = torch.argsort(logit_6, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list_6.append(pred_id)

    if args.ds == "mmvp":
        print(eval_mmvp(pred_answer_list_1, gts_list_1), 
              eval_mmvp(pred_answer_list_2, gts_list_1), 
              eval_mmvp(pred_answer_list_3, gts_list_1), 
              eval_mmvp(pred_answer_list_4, gts_list_1), 
              eval_mmvp(pred_answer_list_5, gts_list_1), 
              eval_mmvp(pred_answer_list_6, gts_list_1), 
              eval_mmvp(pred_answer_ensemble_list, gts_list_1),
              eval_mmvp(pred_answer_f_list, gts_list_1))
    elif args.ds == 'mme':
        print(eval_mme(question_ids_list_1, pred_answer_list_1, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_2, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_3, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_4, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_5, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_list_6, gts_list_1),  
              eval_mme(question_ids_list_1, pred_answer_ensemble_list, gts_list_1),
              eval_mme(question_ids_list_1, pred_answer_f_list, gts_list_1))
    else:
        print(eval(pred_answer_list_1, gts_list_1), 
              eval(pred_answer_list_2, gts_list_1), 
              eval(pred_answer_list_3, gts_list_1), 
              eval(pred_answer_list_4, gts_list_1), 
              eval(pred_answer_list_5, gts_list_1), 
              eval(pred_answer_list_6, gts_list_1), 
              eval(pred_answer_ensemble_list, gts_list_1),
              eval(pred_answer_f_list, gts_list_1))




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str, default='instruct_blip')
    parser.add_argument('--anno_path', type=str, default='SEED-Bench.json')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--logits_recorded_base_path_1', type=str, default=None)
    parser.add_argument('--logits_recorded_base_path_2', type=str, default=None)
    parser.add_argument('--logits_recorded_base_path_3', type=str, default=None)
    parser.add_argument('--logits_recorded_base_path_4', type=str, default=None)
    parser.add_argument('--logits_recorded_base_path_5', type=str, default=None)
    parser.add_argument('--logits_recorded_base_path_6', type=str, default=None)
    parser.add_argument('--logits_recorded_base_path_7', type=str, default=None)
    parser.add_argument('--logits_recorded_base_path_8', type=str, default=None)
    parser.add_argument('--logits_recorded_base_path_9', type=str, default=None)
    parser.add_argument('--logits_recorded_base_path_10', type=str, default=None)
    parser.add_argument('--logits_recorded_base_path_11', type=str, default=None)
    parser.add_argument('--logits_recorded_base_path_12', type=str, default=None)
    parser.add_argument('--ori_ans_file_path', type=str, default=None)
    parser.add_argument('--task', type=str, default='all')
    parser.add_argument('--sample_nums', type=int, default=1)
    parser.add_argument('--top_nums_sim', type=int, default=0)
    parser.add_argument('--start_sim', type=int, default=0)
    parser.add_argument('--top_nums_tie', type=int, default=1)
    parser.add_argument('--start_tie', type=int, default=0)
    parser.add_argument('--method', type=str, default="debias")
    parser.add_argument('--ds', type=str, default=None)
    parser.add_argument('--normalize_method', type=str, default=None)
    parser.add_argument('--nums_ensemble', type=int, default=None)
    parser.add_argument('--weight_path', type=str, default=None)
    args = parser.parse_args()
    
    args = parser.parse_args()

    print(f'evaluating.. {args.model}')

    if args.method == "debias":
        print(args.logits_recorded_base_path_1.split("/", -1)[-1])
        debias(args, c=0.5)
    elif args.method == "contrast":
        print(args.logits_recorded_base_path_1.split("/", -1)[-1])
        contrast(args, c=0.5)
    elif args.method == "ensemble":
        print(args.logits_recorded_base_path_1.split("/", -1)[-1])
        ensemble(args, CA=False)
    elif args.method == "debias_ensemble":
        print(args.logits_recorded_base_path_1.split("/", -1)[-1])
        debias_ensemble(args, c=0.05)
    elif args.method == "composition":
        print(args.logits_recorded_base_path_1.split("/", -1)[-1])
        composition(args)
    elif args.method == "debias_highlight":
        print(args.logits_recorded_base_path_1.split("/", -1)[-1])
        debias_highlight(args)
    elif args.method == "all": 
        debias(args, c=10)
        contrast(args, c=10)