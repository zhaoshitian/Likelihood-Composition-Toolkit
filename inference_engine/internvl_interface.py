import argparse
import os
import re

import torch
from torch import nn
from InternVL.internvl_chat.internvl.train.dataset import build_transform, expand2square
from PIL import Image
from tqdm import tqdm
from transformers import LlamaTokenizer
from InternVL.internvl_chat.internvl.model.internvl_chat import InternVLChatModel
from InternVL.internvl_chat.internvl.conversation import get_conv_template

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

def load_image(image_file, input_size=224, pad2square=False):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size, pad2square=pad2square)
    image = transform(image)
    return image


def post_processing(response):
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    response = re.sub(pattern, '', response)
    return response

class MLLM_Tester(nn.Module):

    def __init__(self, args):
        super().__init__()

        prompt = 'Answer the question using a single word or phrase.'
        self.IMG_START_TOKEN='<img>'
        self.IMG_END_TOKEN='</img>'
        self.IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
        tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
        model = InternVLChatModel.from_pretrained(
            args.model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map='auto').cuda().eval()
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = self.model.config.force_image_size or self.model.config.vision_config.image_size
        self.pad2square = self.model.config.pad2square
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

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
        # print(f"data_path: {data_path}")
        if data_type[0] == 'image':
            # preprocessing images in evaluation dimension 1-9
            if args.noise_step is None:
                raw_image_batch = [load_image(dp, self.image_size, self.pad2square).unsqueeze(0).cuda().to(torch.bfloat16) for dp in data_path_batch]
                image_batch = raw_image_batch
            else:
                raw_image_batch = [Image.open(open(dp, "rb")) for dp in data_path_batch]
                image_batch = [add_diffusion_noise(self.vis_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'], args.noise_step).cuda() for raw_image in raw_image_batch]
            image_batch = torch.stack(image_batch, dim=0)
            image_batch = image_batch.half()
            # pdb.set_trace()


        sample_image_batch = []
        sample_image_label_batch = []
        sample_image_input_id_batch = []
        
        sample_question_label_batch = []
        sample_question_input_id_batch = []

        num_choices_list = []

        for i, (question, choices, image) in enumerate(zip(question_batch, choices_batch, image_batch)):

            num_choices_list.append(len(choices))

            for choice in choices:

                # image
                template = get_conv_template(self.model.template)
                image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.num_image_token + self.IMG_END_TOKEN
                template.append_message(template.roles[0], image_tokens + '\n' + question)
                template.append_message(template.roles[1], None)
                query = template.get_prompt()
                model_inputs = self.tokenizer(query, return_tensors='pt')
                input_ids = model_inputs['input_ids'].cuda()
                attention_mask = model_inputs['attention_mask'].cuda()
                oprint(input_ids.shape)

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


                input_ids0_img = tokenizer_image_token(prompt_img_stage2_0, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
                input_ids_img = tokenizer_image_token(prompt_img_stage2, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

                # print(f"len_input_ids: {len(input_ids_img[0])}")
                len_input_ids0_img = len(input_ids0_img)
                target = copy.deepcopy(input_ids_img)
                target[:len_input_ids0_img] = -100
                labels = target
                # print(f"len_labels: {len(labels[0])}")

                sample_image_input_id_batch.append(input_ids_img)
                sample_image_label_batch.append(labels)
                sample_image_batch.append(image[0])


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


                input_ids0_ques = tokenizer_image_token(prompt_ques_stage2_0, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
                input_ids_ques = tokenizer_image_token(prompt_ques_stage2, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

                # print(f"len_input_ids: {len(input_ids_ques[0])}")
                len_input_ids0_ques = len(input_ids0_ques)
                target = copy.deepcopy(input_ids_ques)
                target[:len_input_ids0_ques] = -100
                labels = target
                # print(f"len_labels: {len(labels[0])}")

                sample_question_input_id_batch.append(input_ids_ques)
                sample_question_label_batch.append(labels)


        sample_image_batch = torch.stack(sample_image_batch, dim=0)
        sample_image_input_id_batch = pad_sequence(sample_image_input_id_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        sample_image_label_batch = pad_sequence(sample_image_label_batch, batch_first=True, padding_value=-100)

        sample_question_input_id_batch = pad_sequence(sample_question_input_id_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        sample_question_label_batch = pad_sequence(sample_question_label_batch, batch_first=True, padding_value=-100)

        losses = []
        logits_img = []
        logits_ques = []

        with torch.no_grad():
            self.model.eval()

            # image
            outputs_img = self.model(
                input_ids=sample_image_input_id_batch,
                labels=sample_image_label_batch,
                images=sample_image_batch
            )

            loss_img = outputs_img['loss']
            logit_img = torch.exp(-loss_img)

            # question
            outputs_ques = self.model(
                input_ids=sample_question_input_id_batch,
                labels=sample_question_label_batch,
                images=None
            )

            loss_ques = outputs_ques['loss']
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
        # print(f"data_path: {data_path}")
        if data_type[0] == 'image':
            # preprocessing images in evaluation dimension 1-9
            if args.noise_step is None:
                raw_image_batch = [Image.open(open(dp, "rb")) for dp in data_path_batch]
                image_batch = [self.vis_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'].cuda() for raw_image in raw_image_batch]
            else:
                raw_image_batch = [Image.open(open(dp, "rb")) for dp in data_path_batch]
                image_batch = [add_diffusion_noise(self.vis_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'], args.noise_step).cuda() for raw_image in raw_image_batch]
            image_batch = torch.stack(image_batch, dim=0)
            image_batch = image_batch.half()
            # pdb.set_trace()


        sample_image_batch = []
        sample_image_label_batch = []
        sample_image_input_id_batch = []
        
        sample_question_label_batch = []
        sample_question_input_id_batch = []

        num_choices_list = []

        for i, (question, choices, image) in enumerate(zip(question_batch, choices_batch, image_batch)):

            num_choices_list.append(len(choices))

            choice_list = []
            for i, c in enumerate(choices):
                choice_list.append('{}. {}'.format(all_options[i], c))
            choice_txt = '\n'.join(choice_list)

            for j, choice in enumerate(choices):

                # image
                ori_question = question
                question_img = question
                if self.model.config.mm_use_im_start_end:
                    question_img = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question_img + '\n' + choice_txt
                else:
                    question_img = DEFAULT_IMAGE_TOKEN + '\n' + question_img + '\n' + choice_txt

                conv_img = conv_templates[self.args.conv_mode].copy()
                conv_img.append_message(conv_img.roles[0], question_img)
                conv_img.append_message(conv_img.roles[1], all_options[j])
                prompt_img_stage2 = conv_img.get_prompt()

                conv0_img = conv_templates[self.args.conv_mode].copy()
                conv0_img.append_message(conv0_img.roles[0], question_img)
                conv0_img.append_message(conv0_img.roles[1], None)
                prompt_img_stage2_0 = conv0_img.get_prompt()


                input_ids0_img = tokenizer_image_token(prompt_img_stage2_0, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
                input_ids_img = tokenizer_image_token(prompt_img_stage2, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

                # print(f"len_input_ids: {len(input_ids_img[0])}")
                len_input_ids0_img = len(input_ids0_img)
                target = copy.deepcopy(input_ids_img)
                target[:len_input_ids0_img] = -100
                labels = target
                # print(f"len_labels: {len(labels[0])}")

                sample_image_input_id_batch.append(input_ids_img)
                sample_image_label_batch.append(labels)
                sample_image_batch.append(image[0])


                # question
                ori_question = ori_question + '\n' + choice_txt
                conv_ques = conv_templates[self.args.conv_mode].copy()
                conv_ques.append_message(conv_ques.roles[0], ori_question)
                conv_ques.append_message(conv_ques.roles[1], all_options[j])
                prompt_ques_stage2 = conv_ques.get_prompt()
                # print(prompt_stage2)

                conv0_ques = conv_templates[self.args.conv_mode].copy()
                conv0_ques.append_message(conv0_ques.roles[0], ori_question)
                conv0_ques.append_message(conv0_ques.roles[1], None)
                prompt_ques_stage2_0 = conv0_ques.get_prompt()


                input_ids0_ques = tokenizer_image_token(prompt_ques_stage2_0, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
                input_ids_ques = tokenizer_image_token(prompt_ques_stage2, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

                # print(f"len_input_ids: {len(input_ids_ques[0])}")
                len_input_ids0_ques = len(input_ids0_ques)
                target = copy.deepcopy(input_ids_ques)
                target[:len_input_ids0_ques] = -100
                labels = target
                # print(f"len_labels: {len(labels[0])}")

                sample_question_input_id_batch.append(input_ids_ques)
                sample_question_label_batch.append(labels)


        sample_image_batch = torch.stack(sample_image_batch, dim=0)
        sample_image_input_id_batch = pad_sequence(sample_image_input_id_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        sample_image_label_batch = pad_sequence(sample_image_label_batch, batch_first=True, padding_value=-100)

        sample_question_input_id_batch = pad_sequence(sample_question_input_id_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        sample_question_label_batch = pad_sequence(sample_question_label_batch, batch_first=True, padding_value=-100)

        losses = []
        logits_img = []
        logits_ques = []

        with torch.no_grad():
            self.model.eval()

            # image
            outputs_img = self.model(
                input_ids=sample_image_input_id_batch,
                labels=sample_image_label_batch,
                images=sample_image_batch,
                output_hidden_states=True
            )

            loss_img = outputs_img['loss']
            logit_img = torch.exp(-loss_img)

            # question
            outputs_ques = self.model(
                input_ids=sample_question_input_id_batch,
                labels=sample_question_label_batch,
                images=None,
                output_hidden_states=True
            )

            loss_ques = outputs_ques['loss']
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