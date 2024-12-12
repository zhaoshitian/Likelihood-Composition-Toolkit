import os
import sys  
import json
import torch 
import argparse
import numpy as np
from PIL import Image  
from tqdm import tqdm
# from utils import model_gen, load_jsonl
from transformers import AutoModelForCausalLM, AutoTokenizer  
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torchvision

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


def __padding__(image):
    width, height = image.size
    tar = max(width, height)
    top_padding = int((tar - height)/2)
    bottom_padding = tar - height - top_padding
    left_padding = int((tar - width)/2)
    right_padding = tar - width - left_padding
    image = torchvision.transforms.functional.pad(image, [left_padding, top_padding, right_padding, bottom_padding])
    return image




class MLLM_Tester(nn.Module):

    def __init__(self, args):
        super().__init__()

        # kwargs = {"device_map": device_map}
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto', trust_remote_code=True, low_cpu_mem_usage=True).eval().half()
        self.model.tokenizer = self.tokenizer
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


    def forward(self, x):
        data_path, question, choices = x['data_path'], x['question'], x['choices']
        data_type = x['data_type']
        # print(f"data_path: {data_path}")
        if data_type == 'image':
            # preprocessing images in evaluation dimension 1-9
            if x['image'] is None:
                raw_image = Image.open(open(data_path, "rb"))
            else:
                raw_image = x['image']
            image = self.vis_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'].cuda()
            image = image.half()
            # pdb.set_trace()

        # print(f"image_shape: {image.shape}")
        bs = image.size(0)
        bs = 1
        # print(f"bs: {bs}")
        n_segments = 1
        # prepare prompt based on the input question
        # prompt = self.process_prompt(self.args, self.model.config, question)
        # prompt = [prompt] * bs


        n_cands = len(choices)

        losses = []

        with torch.no_grad():
            self.model.eval()
            for choice in choices:
                # print(f"choice: {choice}")
                prompt0, prompt = self.process_prompt(self.args, self.model.config, question, choice)
                # print(f"prompt0 {prompt0}")
                # print(f"prompt: {prompt}")
                # prompt = [prompt] * bs

                input_ids0 = tokenizer_image_token(prompt0, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                # print(f"len_input_ids: {len(input_ids[0])}")
                len_input_ids0 = len(input_ids0[0])
                target = copy.deepcopy(input_ids)
                target[0][:len_input_ids0] = -100
                labels = target
                # print(f"len_labels: {len(labels[0])}")

                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    images=image
                )

                loss = outputs[0]
                losses.append(loss)

        losses = torch.stack(losses)

        return losses


    def forward_debias(self, x):
        data_path, question, choices = x['data_path'], x['question'], x['choices']
        data_type = x['data_type']
        # print(f"data_path: {data_path}")
        if data_type == 'image':
            # preprocessing images in evaluation dimension 1-9
            if x['image'] is None:
                raw_image = Image.open(open(data_path, "rb"))
                
            else:
                raw_image = x['image']
            image = self.vis_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'].cuda()
            image = image.half()
            # pdb.set_trace()

        losses = []
        logits_img = []
        logits_ques = []

        with torch.no_grad():
            self.model.eval()

            for choice in choices:

                # image
                ori_question = question
                question_img = question
                if self.model.config.mm_use_im_start_end:
                    question_img = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question_img
                else:
                    question_img = DEFAULT_IMAGE_TOKEN + '\n' + question_img

                conv_img = conv_templates[self.args.conv_mode].copy()
                conv_img.append_message(conv_img.roles[0], question_img)
                conv_img.append_message(conv_img.roles[1], choice)
                prompt_img_stage2 = conv_img.get_prompt()

                conv0_img = conv_templates[self.args.conv_mode].copy()
                conv0_img.append_message(conv0_img.roles[0], question_img)
                conv0_img.append_message(conv0_img.roles[1], None)
                prompt_img_stage2_0 = conv0_img.get_prompt()


                input_ids0_img = tokenizer_image_token(prompt_img_stage2_0, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                input_ids_img = tokenizer_image_token(prompt_img_stage2, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                # print(f"len_input_ids: {len(input_ids_img[0])}")
                len_input_ids0_img = len(input_ids0_img[0])
                target = copy.deepcopy(input_ids_img)
                target[0][:len_input_ids0_img] = -100
                labels = target
                # print(f"len_labels: {len(labels[0])}")

                outputs_img = self.model(
                    input_ids=input_ids_img,
                    labels=labels,
                    images=image
                )

                loss_img = outputs_img[0]
                losses.append(loss_img)
                logit_img = torch.exp(-loss_img)

                logits_img.append(logit_img)


                # question
                conv_ques = conv_templates[self.args.conv_mode].copy()
                conv_ques.append_message(conv_ques.roles[0], ori_question)
                conv_ques.append_message(conv_ques.roles[1], choice)
                prompt_ques_stage2 = conv_ques.get_prompt()
                # print(prompt_stage2)

                conv0_ques = conv_templates[self.args.conv_mode].copy()
                conv0_ques.append_message(conv0_ques.roles[0], ori_question)
                conv0_ques.append_message(conv0_ques.roles[1], None)
                prompt_ques_stage2_0 = conv0_ques.get_prompt()


                input_ids0_ques = tokenizer_image_token(prompt_ques_stage2_0, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                input_ids_ques = tokenizer_image_token(prompt_ques_stage2, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                # print(f"len_input_ids: {len(input_ids_ques[0])}")
                len_input_ids0_ques = len(input_ids0_ques[0])
                target = copy.deepcopy(input_ids_ques)
                target[0][:len_input_ids0_ques] = -100
                labels = target
                # print(f"len_labels: {len(labels[0])}")

                outputs_ques = self.model(
                    input_ids=input_ids_ques,
                    labels=labels,
                    images=None
                )

                loss_ques = outputs_ques[0]
                logit_ques = torch.exp(-loss_ques)

                logits_ques.append(logit_ques)

        losses = torch.stack(losses)
        logits_img = torch.stack(logits_img)
        logits_ques = torch.stack(logits_ques)

        return losses, logits_img, logits_ques


    def forward_debias_batched(self, args, x, need_bos=True, padding=False):
        data_path_batch, question_batch, choices_batch = x['data_path'], x['question'], x['choices']
        origin_len = len(choices_batch)
        data_type = x['data_type']

        image_batch = data_path_batch

        num_choices_list = []

        prompt_img = '[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{}'
        prompt_ques = '[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{}'
        if args.use_instruction == "True":
            prompt_img = prompt_img + args.instruction_prompt
            prompt_ques = prompt_ques + args.instruction_prompt

        logits_img_list = []
        logits_ques_list = []
        loss_list = []

        for i, (question, choices, image) in enumerate(zip(question_batch, choices_batch, image_batch)):

            num_choices_list.append(len(choices))

            for choice in choices:
###############################################################################
                # img
                prompt_img = prompt_img.format(question, choice)
                text = prompt_img
                images = image
                pt1 = 0
                embeds = []
                im_mask = []
                images = [images]
                images_loc = [0]
                for i, pts in enumerate(images_loc + [len(text)]):
                    subtext = text[pt1:pts]
                    if need_bos or len(subtext) > 0:
                        text_embeds = self.model.encode_text(subtext, add_special_tokens=need_bos)
                        embeds.append(text_embeds)
                        im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
                        need_bos = False
                    if i < len(images):
                        try:
                            raw_image = Image.open(images[i]).convert('RGB')
                        except:
                            raw_image = images[i].convert('RGB')
                        if padding:
                            raw_image = __padding__(raw_image)
                        raw_image = self.model.vis_processor(raw_image).unsqueeze(0).half().cuda()
                        image_embeds = self.model.encode_img(raw_image)
                        embeds.append(image_embeds.to(text_embeds.device))
                        im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
                    pt1 = pts
                embeds = torch.cat(embeds, dim=1)
                im_mask = torch.cat(im_mask, dim=1)
                im_mask = im_mask.bool()

                outputs = self.model(inputs_embeds=embeds, im_mask=im_mask)
                img_logits = outputs['logits']

                answer_input_ids = self.model.tokenizer(choice, return_tensors='pt', add_special_tokens=False).input_ids[0].to(self.model.device)
                len_answer_token = len(answer_input_ids)
                labels = torch.cat((torch.tensor([-100]*(img_logits.shape[1]-len_answer_token)).to(self.model.device), answer_input_ids), dim=0)

                loss = None
                if labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = img_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    label_shape = shift_labels.shape
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss(reduce=False)
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    loss = loss.view(label_shape)
                    nums_answer_token = torch.count_nonzero(loss, dim=-1)
                    loss = loss.sum(dim=-1)
                    loss = loss / nums_answer_token

                    # print(loss)
                    logits_img_list.append(torch.exp(-loss))
                    loss_list.append(loss)

                # ques
                prompt_ques = prompt_ques.format(question, choice)
                text = prompt_ques
                images = image
                pt1 = 0
                embeds = []
                im_mask = []
                images = []
                images_loc = [0]
                for i, pts in enumerate(images_loc + [len(text)]):
                    subtext = text[pt1:pts]
                    if need_bos or len(subtext) > 0:
                        text_embeds = self.model.encode_text(subtext, add_special_tokens=need_bos)
                        embeds.append(text_embeds)
                        im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
                        need_bos = False
                    if i < len(images):
                        try:
                            raw_image = Image.open(images[i]).convert('RGB')
                        except:
                            raw_image = images[i].convert('RGB')
                        if padding:
                            raw_image = __padding__(raw_image)
                        raw_image = self.model.vis_processor(raw_image).unsqueeze(0).half().cuda()
                        image_embeds = self.model.encode_img(raw_image)
                        embeds.append(image_embeds.to(text_embeds.device))
                        im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
                    pt1 = pts
                embeds = torch.cat(embeds, dim=1)
                im_mask = torch.cat(im_mask, dim=1)
                im_mask = im_mask.bool()

                outputs = self.model(inputs_embeds=embeds, im_mask=im_mask)
                ques_logits = outputs['logits']

                answer_input_ids = self.model.tokenizer(choice, return_tensors='pt', add_special_tokens=False).input_ids[0].to(self.model.device)
                len_answer_token = len(answer_input_ids)
                labels = torch.cat((torch.tensor([-100]*(ques_logits.shape[1]-len_answer_token)).to(self.model.device), answer_input_ids), dim=0)

                loss = None
                if labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = ques_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    label_shape = shift_labels.shape
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss(reduce=False)
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    loss = loss.view(label_shape)
                    nums_answer_token = torch.count_nonzero(loss, dim=-1)
                    loss = loss.sum(dim=-1)
                    loss = loss / nums_answer_token
                    logits_ques_list.append(torch.exp(-loss))


#################################################            
        logits_img_list = torch.tensor(logits_img_list)
        logits_ques_list = torch.tensor(logits_ques_list)
        loss_list = torch.tensor(loss_list)
        loss_img = split_list_by_lengths(loss_list, num_choices_list)
        logit_img = split_list_by_lengths(logits_img_list, num_choices_list)
        logit_ques = split_list_by_lengths(logits_ques_list, num_choices_list)

        return loss_img, logit_img, logit_ques

    def forward_debias_batched_2(self, args, x, need_bos=True, padding=False):
        data_path_batch, question_batch, choices_batch = x['data_path'], x['question'], x['choices']
        origin_len = len(choices_batch)
        data_type = x['data_type']

        image_batch = data_path_batch

        num_choices_list = []

        prompt_img = '[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{}'
        prompt_ques = '[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{}'
        if args.use_instruction == "True":
            prompt_img = prompt_img + args.instruction_prompt
            prompt_ques = prompt_ques + args.instruction_prompt
            
        logits_img_list = []
        logits_ques_list = []
        loss_list = []

        for i, (question, choices, image) in enumerate(zip(question_batch, choices_batch, image_batch)):

            num_choices_list.append(len(choices))

            choice_list = []
            for i, c in enumerate(choices):
                choice_list.append('{}. {}'.format(all_options[i], c))
            choice_txt = '\n'.join(choice_list)

            question = question + '\n' + choice_txt

            for j, choice in enumerate(choices):
###############################################################################
                # img
                
                prompt_img = prompt_img.format(str(question), all_options[j])
                text = prompt_img
                images = image
                pt1 = 0
                embeds = []
                im_mask = []
                images = [images]
                images_loc = [0]
                for i, pts in enumerate(images_loc + [len(text)]):
                    subtext = text[pt1:pts]
                    if need_bos or len(subtext) > 0:
                        text_embeds = self.model.encode_text(subtext, add_special_tokens=need_bos)
                        embeds.append(text_embeds)
                        im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
                        need_bos = False
                    if i < len(images):
                        try:
                            raw_image = Image.open(images[i]).convert('RGB')
                        except:
                            raw_image = images[i].convert('RGB')
                        if padding:
                            raw_image = __padding__(raw_image)
                        raw_image = self.model.vis_processor(raw_image).unsqueeze(0).half().cuda()
                        image_embeds = self.model.encode_img(raw_image)
                        embeds.append(image_embeds.to(text_embeds.device))
                        im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
                    pt1 = pts
                embeds = torch.cat(embeds, dim=1)
                im_mask = torch.cat(im_mask, dim=1)
                im_mask = im_mask.bool()

                outputs = self.model(inputs_embeds=embeds, im_mask=im_mask)
                img_logits = outputs['logits']

                answer_input_ids = self.model.tokenizer(all_options[j], return_tensors='pt', add_special_tokens=False).input_ids[0].to(self.model.device)
                len_answer_token = len(answer_input_ids)
                labels = torch.cat((torch.tensor([-100]*(img_logits.shape[1]-len_answer_token)).to(self.model.device), answer_input_ids), dim=0)

                loss = None
                if labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = img_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    label_shape = shift_labels.shape
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss(reduce=False)
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    loss = loss.view(label_shape)
                    nums_answer_token = torch.count_nonzero(loss, dim=-1)
                    loss = loss.sum(dim=-1)
                    loss = loss / nums_answer_token
                    logits_img_list.append(torch.exp(-loss))
                    loss_list.append(loss)

                # ques
                prompt_ques = prompt_ques.format(question, all_options[j])
                text = prompt_ques
                images = image
                pt1 = 0
                embeds = []
                im_mask = []
                images = []
                images_loc = [0]
                for i, pts in enumerate(images_loc + [len(text)]):
                    subtext = text[pt1:pts]
                    if need_bos or len(subtext) > 0:
                        text_embeds = self.model.encode_text(subtext, add_special_tokens=need_bos)
                        embeds.append(text_embeds)
                        im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
                        need_bos = False
                    if i < len(images):
                        try:
                            raw_image = Image.open(images[i]).convert('RGB')
                        except:
                            raw_image = images[i].convert('RGB')
                        if padding:
                            raw_image = __padding__(raw_image)
                        raw_image = self.model.vis_processor(raw_image).unsqueeze(0).half().cuda()
                        image_embeds = self.model.encode_img(raw_image)
                        embeds.append(image_embeds.to(text_embeds.device))
                        im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
                    pt1 = pts
                embeds = torch.cat(embeds, dim=1)
                im_mask = torch.cat(im_mask, dim=1)
                im_mask = im_mask.bool()

                outputs = self.model(inputs_embeds=embeds, im_mask=im_mask)
                ques_logits = outputs['logits']

                answer_input_ids = self.model.tokenizer(all_options[j], return_tensors='pt', add_special_tokens=False).input_ids[0].to(self.model.device)
                len_answer_token = len(answer_input_ids)
                labels = torch.cat((torch.tensor([-100]*(ques_logits.shape[1]-len_answer_token)).to(self.model.device), answer_input_ids), dim=0)

                loss = None
                if labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = ques_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    label_shape = shift_labels.shape
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss(reduce=False)
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    loss = loss.view(label_shape)
                    nums_answer_token = torch.count_nonzero(loss, dim=-1)
                    loss = loss.sum(dim=-1)
                    loss = loss / nums_answer_token
                    logits_ques_list.append(torch.exp(-loss))


#################################################            
        logits_img_list = torch.tensor(logits_img_list)
        logits_ques_list = torch.tensor(logits_ques_list)
        loss_list = torch.tensor(loss_list)
        loss_img = split_list_by_lengths(loss_list, num_choices_list)
        logit_img = split_list_by_lengths(logits_img_list, num_choices_list)
        logit_ques = split_list_by_lengths(logits_ques_list, num_choices_list)

        return loss_img, logit_img, logit_ques



def build(args):
    return MLLM_Tester(args)