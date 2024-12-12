import argparse
import itertools
import json
import os
from functools import partial
import copy

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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

        # torch.distributed.init_process_group(
        #     backend='nccl',
        #     world_size=int(os.getenv('WORLD_SIZE', '1')),
        #     rank=int(os.getenv('RANK', '0')),
        # )

        # torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map='cuda', trust_remote_code=True).eval()

        tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                                trust_remote_code=True)

        self.model = model
        self.tokenizer = tokenizer

        # disable_torch_init()
        # model_path = os.path.expanduser(args.model_path)
        # model_name = get_model_name_from_path(model_path)
        # print(model_name)
        # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
        # self.model = model
        # print(model.dtype)
        # self.vis_processor = image_processor
        # self.tokenizer = tokenizer
        # self.args = args

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


    def forward_debias_batched(self, args, x):
        data_path_batch, question_batch, choices_batch = x['data_path'], x['question'], x['choices']
        origin_len = len(choices_batch)
        data_type = x['data_type']

        sample_image_batch = []
        sample_image_label_batch = []
        sample_image_input_id_batch = []
        
        sample_question_label_batch = []
        sample_question_input_id_batch = []

        num_choices_list = []

        image_batch = data_path_batch

        prompt = '<img>{}</img>Question: {}\nAnswer: {}'
        prompt_0 = '<img>{}</img>Question: {}\nAnswer:'

        for i, (question, choices, image) in enumerate(zip(question_batch, choices_batch, image_batch)):

            num_choices_list.append(len(choices))

            for choice in choices:

                # image
                # ori_question = question
                # question_img = question

                if args.negative == "True":
                    prompt_img = '<img>{}</img>Question: {}'+args.instruction_prompt+'\nAnswer: {}'
                    prompt_img_0 = '<img>{}</img>Question: {}'+args.instruction_prompt+'\nAnswer:'
                else:
                    prompt_img = '<img>{}</img>Question: {}\nAnswer: {}'
                    prompt_img_0 = '<img>{}</img>Question: {}\nAnswer:'


                prompt_img_stage2 = prompt_img.format(image, question, choice)
                prompt_img_stage2_0 = prompt_img_0.format(image, question)

                input_ids_img = self.tokenizer(prompt_img_stage2).input_ids
                input_ids0_img = self.tokenizer(prompt_img_stage2_0).input_ids

                # print(f"len_input_ids: {len(input_ids_img[0])}")
                len_input_ids0_img = len(input_ids0_img)
                target = copy.deepcopy(input_ids_img)
                # print(target)
                for l in range(len_input_ids0_img):
                    target[l] = -100
                # target[:len_input_ids0_img] = [-100]*len_input_ids0_img
                labels = target
                # print(f"len_labels: {len(labels[0])}")

                sample_image_input_id_batch.append(input_ids_img)
                sample_image_label_batch.append(labels)
                sample_image_batch.append(image[0])


                # question

                prompt_ques = 'Question: {}\nAnswer: {}'
                prompt_ques_0 = 'Question: {}\nAnswer:'

                if args.negative == "True":
                    prompt_img = 'Question: {}'+args.instruction_prompt+'\nAnswer: {}'
                    prompt_img_0 = 'Question: {}'+args.instruction_prompt+'\nAnswer:'
                else:
                    prompt_img = 'Question: {}\nAnswer: {}'
                    prompt_img_0 = 'Question: {}\nAnswer:'

                prompt_ques_stage2 = prompt_ques.format(question, choice)
                prompt_ques_stage2_0 = prompt_ques_0.format(question)

                input_ids_ques = self.tokenizer(prompt_ques_stage2).input_ids
                input_ids0_ques = self.tokenizer(prompt_ques_stage2_0).input_ids

                # print(f"len_input_ids: {len(input_ids_ques[0])}")
                len_input_ids0_ques = len(input_ids0_ques)
                target = copy.deepcopy(input_ids_ques)
                for l in range(len_input_ids0_ques):
                    target[l] = -100
                # target[:len_input_ids0_ques] = [-100]*len_input_ids0_ques
                labels = target
                # print(f"len_labels: {len(labels[0])}")

                sample_question_input_id_batch.append(input_ids_ques)
                sample_question_label_batch.append(labels)

        pad_token_id=self.tokenizer.eod_id

#         sample_image_batch = torch.stack(sample_image_batch, dim=0)
        # print(sample_image_input_id_batch)
        sample_image_input_id_batch = [torch.tensor(s, dtype=torch.long).cuda() for s in sample_image_input_id_batch]
        # print(sample_image_input_id_batch)
        sample_image_input_id_batch = pad_sequence(sample_image_input_id_batch, batch_first=True, padding_value=pad_token_id)
        sample_image_attention_mask_batch = [1 - input_tokens.eq(pad_token_id).float() for input_tokens in sample_image_input_id_batch]
        sample_image_attention_mask_batch = torch.stack(sample_image_attention_mask_batch, dim=0)
        sample_image_label_batch = [torch.tensor(s, dtype=torch.long).cuda() for s in sample_image_label_batch]
        sample_image_label_batch = pad_sequence(sample_image_label_batch, batch_first=True, padding_value=-100)

        sample_question_input_id_batch = [torch.tensor(s, dtype=torch.long).cuda() for s in sample_question_input_id_batch]
        sample_question_input_id_batch = pad_sequence(sample_question_input_id_batch, batch_first=True, padding_value=pad_token_id)
        sample_question_attention_mask_batch = [1 - input_tokens.eq(pad_token_id).float() for input_tokens in sample_question_input_id_batch]
        sample_question_attention_mask_batch = torch.stack(sample_question_attention_mask_batch, dim=0)
        sample_question_label_batch = [torch.tensor(s, dtype=torch.long).cuda() for s in sample_question_label_batch]
        sample_question_label_batch = pad_sequence(sample_question_label_batch, batch_first=True, padding_value=-100)

        losses = []
        logits_img = []
        logits_ques = []

        with torch.no_grad():
            self.model.eval()

            # image
            outputs_img = self.model(
                input_ids=sample_image_input_id_batch[:, :].cuda(),
                attention_mask=sample_image_attention_mask_batch[:, :].cuda(),
                return_dict=True,
            )

            # print(outputs_img.logits.shape)
            # print(outputs_img.keys())

            vocab_size = outputs_img.logits.shape[-1]

            logits_img = outputs_img.logits

            loss_img = None
            if sample_image_label_batch is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits_img[..., :-1, :].contiguous()
                shift_labels = sample_image_label_batch[..., 1:].contiguous()
                label_shape = shift_labels.shape
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduce=False)
                shift_logits = shift_logits.view(-1, vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss_img = loss_fct(shift_logits, shift_labels)
                loss_img = loss_img.view(label_shape)
                nums_answer_token = torch.count_nonzero(loss_img, dim=-1)
                loss_img = loss_img.sum(dim=-1)
                loss_img = loss_img / nums_answer_token

            logit_img = torch.exp(-loss_img)

            # question
            outputs_ques = self.model(
                input_ids=sample_question_input_id_batch[:, :].cuda(),
                attention_mask=sample_question_attention_mask_batch[:, :].cuda(),
                return_dict=True,
            )

            logits_ques = outputs_ques.logits

            loss_ques = None
            if sample_question_label_batch is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits_ques[..., :-1, :].contiguous()
                shift_labels = sample_question_label_batch[..., 1:].contiguous()
                label_shape = shift_labels.shape
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduce=False)
                shift_logits = shift_logits.view(-1, vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss_ques = loss_fct(shift_logits, shift_labels)
                loss_ques = loss_ques.view(label_shape)
                nums_answer_token = torch.count_nonzero(loss_ques, dim=-1)
                loss_ques = loss_ques.sum(dim=-1)
                loss_ques = loss_ques / nums_answer_token

            logit_ques = torch.exp(-loss_ques)


        # losses = torch.stack(losses)
        # print(losses.shape)
        # logits_img = torch.stack(logit_img)
        # logits_ques = torch.stack(logit_ques)
            
        loss_img = split_list_by_lengths(loss_img, num_choices_list)
        logit_img = split_list_by_lengths(logit_img, num_choices_list)
        logit_ques = split_list_by_lengths(logit_ques, num_choices_list)

        return loss_img, logit_img, logit_ques

    def forward_debias_batched_2(self, args, x):
        data_path_batch, question_batch, choices_batch = x['data_path'], x['question'], x['choices']
        origin_len = len(choices_batch)
        data_type = x['data_type']

        sample_image_batch = []
        sample_image_label_batch = []
        sample_image_input_id_batch = []
        
        sample_question_label_batch = []
        sample_question_input_id_batch = []

        num_choices_list = []

        image_batch = data_path_batch

        prompt = '<img>{}</img>Question: {}\nOptions: {}\nAnswer: {}'
        prompt_0 = '<img>{}</img>Question: {}\nOptions: {}\nAnswer:'

        for i, (question, choices, image) in enumerate(zip(question_batch, choices_batch, image_batch)):

            num_choices_list.append(len(choices))

            choice_list = []
            for i, c in enumerate(choices):
                choice_list.append('{}. {}'.format(all_options[i], c))
            choice_txt = '\n'.join(choice_list)

            for j, choice in enumerate(choices):

                # image
                # ori_question = question
                # question_img = question


                if args.negative == "True":
                    prompt_img = '<img>{}</img>Question: {}\nOptions: {}'+args.instruction_prompt+'\nAnswer: {}'
                    prompt_img_0 = '<img>{}</img>Question: {}\nOptions: {}'+args.instruction_prompt+'\nAnswer:'
                else:
                    prompt_img = '<img>{}</img>Question: {}\nOptions: {}\nAnswer: {}'
                    prompt_img_0 = '<img>{}</img>Question: {}\nOptions: {}\nAnswer:'

                prompt_img_stage2 = prompt_img.format(image, question, choice_txt, all_options[j])
                prompt_img_stage2_0 = prompt_img_0.format(image, question, choice_txt)

                input_ids_img = self.tokenizer(prompt_img_stage2).input_ids
                input_ids0_img = self.tokenizer(prompt_img_stage2_0).input_ids

                # print(f"len_input_ids: {len(input_ids_img[0])}")
                len_input_ids0_img = len(input_ids0_img)
                target = copy.deepcopy(input_ids_img)
                # print(target)
                for l in range(len_input_ids0_img):
                    target[l] = -100
                # target[:len_input_ids0_img] = [-100]*len_input_ids0_img
                labels = target
                # print(f"len_labels: {len(labels[0])}")

                sample_image_input_id_batch.append(input_ids_img)
                sample_image_label_batch.append(labels)
                sample_image_batch.append(image[0])


                # question
                if args.negative == "True":
                    prompt_ques = 'Question: {}\nOptions: {}'+args.instruction_prompt+'\nAnswer: {}'
                    prompt_ques_0 = 'Question: {}\nOptions: {}'+args.instruction_prompt+'\nAnswer:'
                else:
                    prompt_ques = 'Question: {}\nOptions: {}\nAnswer: {}'
                    prompt_ques_0 = 'Question: {}\nOptions: {}\nAnswer:'

                prompt_ques_stage2 = prompt_ques.format(question, choice_txt, all_options[j])
                prompt_ques_stage2_0 = prompt_ques_0.format(question, choice_txt)

                input_ids_ques = self.tokenizer(prompt_ques_stage2).input_ids
                input_ids0_ques = self.tokenizer(prompt_ques_stage2_0).input_ids

                # print(f"len_input_ids: {len(input_ids_ques[0])}")
                len_input_ids0_ques = len(input_ids0_ques)
                target = copy.deepcopy(input_ids_ques)
                for l in range(len_input_ids0_ques):
                    target[l] = -100
                # target[:len_input_ids0_ques] = [-100]*len_input_ids0_ques
                labels = target
                # print(f"len_labels: {len(labels[0])}")

                sample_question_input_id_batch.append(input_ids_ques)
                sample_question_label_batch.append(labels)

        pad_token_id=self.tokenizer.eod_id

#         sample_image_batch = torch.stack(sample_image_batch, dim=0)
        # print(sample_image_input_id_batch)
        sample_image_input_id_batch = [torch.tensor(s, dtype=torch.long).cuda() for s in sample_image_input_id_batch]
        # print(sample_image_input_id_batch)
        sample_image_input_id_batch = pad_sequence(sample_image_input_id_batch, batch_first=True, padding_value=pad_token_id)
        sample_image_attention_mask_batch = [1 - input_tokens.eq(pad_token_id).float() for input_tokens in sample_image_input_id_batch]
        sample_image_attention_mask_batch = torch.stack(sample_image_attention_mask_batch, dim=0)
        sample_image_label_batch = [torch.tensor(s, dtype=torch.long).cuda() for s in sample_image_label_batch]
        sample_image_label_batch = pad_sequence(sample_image_label_batch, batch_first=True, padding_value=-100)

        sample_question_input_id_batch = [torch.tensor(s, dtype=torch.long).cuda() for s in sample_question_input_id_batch]
        sample_question_input_id_batch = pad_sequence(sample_question_input_id_batch, batch_first=True, padding_value=pad_token_id)
        sample_question_attention_mask_batch = [1 - input_tokens.eq(pad_token_id).float() for input_tokens in sample_question_input_id_batch]
        sample_question_attention_mask_batch = torch.stack(sample_question_attention_mask_batch, dim=0)
        sample_question_label_batch = [torch.tensor(s, dtype=torch.long).cuda() for s in sample_question_label_batch]
        sample_question_label_batch = pad_sequence(sample_question_label_batch, batch_first=True, padding_value=-100)

        losses = []
        logits_img = []
        logits_ques = []

        with torch.no_grad():
            self.model.eval()

            # image
            outputs_img = self.model(
                input_ids=sample_image_input_id_batch[:, :],
                attention_mask=sample_image_attention_mask_batch[:, :],
                return_dict=True,
            )

            # print(outputs_img.logits.shape)
            # print(outputs_img.keys())

            vocab_size = outputs_img.logits.shape[-1]

            logits_img = outputs_img.logits

            loss_img = None
            if sample_image_label_batch is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits_img[..., :-1, :].contiguous()
                shift_labels = sample_image_label_batch[..., 1:].contiguous()
                label_shape = shift_labels.shape
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduce=False)
                shift_logits = shift_logits.view(-1, vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss_img = loss_fct(shift_logits, shift_labels)
                loss_img = loss_img.view(label_shape)
                nums_answer_token = torch.count_nonzero(loss_img, dim=-1)
                loss_img = loss_img.sum(dim=-1)
                loss_img = loss_img / nums_answer_token

            logit_img = torch.exp(-loss_img)

            # question
            outputs_ques = self.model(
                input_ids=sample_question_input_id_batch[:, :],
                attention_mask=sample_question_attention_mask_batch[:, :],
                return_dict=True,
            )

            logits_ques = outputs_ques.logits

            loss_ques = None
            if sample_question_label_batch is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits_ques[..., :-1, :].contiguous()
                shift_labels = sample_question_label_batch[..., 1:].contiguous()
                label_shape = shift_labels.shape
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduce=False)
                shift_logits = shift_logits.view(-1, vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss_ques = loss_fct(shift_logits, shift_labels)
                loss_ques = loss_ques.view(label_shape)
                nums_answer_token = torch.count_nonzero(loss_ques, dim=-1)
                loss_ques = loss_ques.sum(dim=-1)
                loss_ques = loss_ques / nums_answer_token

            logit_ques = torch.exp(-loss_ques)


        # losses = torch.stack(losses)
        # print(losses.shape)
        # logits_img = torch.stack(logit_img)
        # logits_ques = torch.stack(logit_ques)
            
        loss_img = split_list_by_lengths(loss_img, num_choices_list)
        logit_img = split_list_by_lengths(logit_img, num_choices_list)
        logit_ques = split_list_by_lengths(logit_ques, num_choices_list)

        return loss_img, logit_img, logit_ques


def build(args):
    return MLLM_Tester(args)