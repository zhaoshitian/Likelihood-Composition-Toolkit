import os
import torch
import argparse
from functools import partial
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from sat.helpers import print_rank0
from PIL import Image
from torch import nn
import torch
from torch.nn import CrossEntropyLoss
from CogVLM.utils.models import FineTuneTrainCogAgentModel, FineTuneTestCogAgentModel
from CogVLM.utils.utils import llama2_text_processor, llama2_text_processor_inference, get_image_processor, llama2_tokenizer
from transformers import AutoModelForCausalLM, LlamaTokenizer

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

        self.torch_type = torch.float16
        tokenizer = LlamaTokenizer.from_pretrained('/data/stzhao_data/mllm_ckpt/vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        self.tokenizer = tokenizer
        self.model = model

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
                raw_image_batch = [Image.open(open(dp, "rb")).convert('RGB') for dp in data_path_batch]
                image_batch = raw_image_batch
            else:
                raw_image_batch = [Image.open(open(dp, "rb")) for dp in data_path_batch]
                image_batch = [add_diffusion_noise(self.vis_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'], args.noise_step).cuda() for raw_image in raw_image_batch]
            # image_batch = torch.stack(image_batch, dim=0)
            # image_batch = image_batch.half()
            # pdb.set_trace()


        sample_image_batch = []
        sample_image_label_batch = []
        sample_image_input_id_batch = []
        sample_image_token_type_ids_batch = []
        sample_image_attention_mask_batch = []
        sample_image_cross_images_batch = []
        
        sample_question_label_batch = []
        sample_question_input_id_batch = []
        sample_question_token_type_ids_batch = []
        sample_question_attention_mask_batch = []
        sample_question_cross_images_batch = []

        num_choices_list = []

        loss_list = []
        logits_img_list = []
        logits_ques_list = []

        for i, (question, choices, image) in enumerate(zip(question_batch, choices_batch, image_batch)):

            num_choices_list.append(len(choices))

            for choice in choices:


                # img
                history = []
                if args.use_instruction == "True":
                    query = question + args.instruction_prompt
                else:
                    query = question
                input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=query, answer=choice, history=history, images=[image], template_version='vqa')
                DEVICE = self.model.device

                inputs = {
                    'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
                    'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
                    'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
                    'images': [[input_by_model['images'][0].to(self.torch_type).to(DEVICE)]] if image is not None else None,
                }
                if 'cross_images' in input_by_model and input_by_model['cross_images']:
                    inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.torch_type).to(DEVICE)]]


                answer_input_ids = self.tokenizer(choice)['input_ids']
                labels = torch.tensor([-100]*(inputs['input_ids'][0].shape[-1]-len(answer_input_ids)) + answer_input_ids)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logit_img = outputs['logits']
                loss = None
                if labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = logit_img[..., :-1, :].contiguous()
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

                    loss_list.append(loss)
                    logits_img_list.append(torch.exp(-loss))


                # sample_image_batch.append(inputs['images'])
                # sample_image_input_id_batch.append(inputs['input_ids'])
                # sample_image_label_batch.append(label)
                # sample_image_token_type_ids_batch.append(inputs['token_type_ids'])
                # sample_image_attention_mask_batch.append(inputs['attention_mask'])
                # sample_image_cross_images_batch.append(inputs['cross_images'])


                #question
                # text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT: {}" 
                # history = []
                # query = question
                # query = text_only_template.format(query, choice)
                # input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=query, history=history, template_version='base')
                # DEVICE = self.model.device

                # inputs = {
                #     'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
                #     'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
                #     'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
                #     'images': [],
                # }
                # if 'cross_images' in input_by_model and input_by_model['cross_images']:
                #     inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(self.torch_type)]]

                # print(inputs)

                # answer_input_ids = self.tokenizer(choice, add_special_tokens=False)['input_ids']
                # labels = torch.tensor([-100]*(inputs['input_ids'][0].shape[-1]-len(answer_input_ids)) + answer_input_ids)

                # outputs = self.model(**inputs)
                # logit_ques = outputs['logits']
                # print(outputs.keys())

                # loss = None
                # if labels is not None:
                #     # Shift so that tokens < n predict n
                #     shift_logits = logit_ques[..., :-1, :].contiguous()
                #     shift_labels = labels[..., 1:].contiguous()
                #     label_shape = shift_labels.shape
                #     # Flatten the tokens
                #     loss_fct = CrossEntropyLoss(reduce=False)
                #     shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                #     shift_labels = shift_labels.view(-1)
                #     # Enable model parallelism
                #     shift_labels = shift_labels.to(shift_logits.device)
                #     loss = loss_fct(shift_logits, shift_labels)
                #     loss = loss.view(label_shape)
                #     nums_answer_token = torch.count_nonzero(loss, dim=-1)
                #     loss = loss.sum(dim=-1)
                #     loss = loss / nums_answer_token

                #     logits_ques_list.append(torch.exp(-loss))

                # sample_question_input_id_batch.append(inputs['input_ids'])
                # sample_question_label_batch.append(label)
                # sample_question_token_type_ids_batch.append(inputs['token_type_ids'])
                # sample_question_attention_mask_batch.append(inputs['attention_mask'])
                # sample_question_cross_images_batch.append(inputs['cross_images'])

        loss_list = torch.tensor(loss_list)
        logits_img_list = torch.tensor(logits_img_list)
        logits_ques_list = logits_img_list
            
        loss_img = split_list_by_lengths(loss_list, num_choices_list)
        logit_img = split_list_by_lengths(logits_img_list, num_choices_list)
        logit_ques = split_list_by_lengths(logits_ques_list, num_choices_list)

        return loss_img, logit_img, logit_ques

    def forward_debias_batched_2(self, args, x):
        data_path_batch, question_batch, choices_batch = x['data_path'], x['question'], x['choices']
        origin_len = len(choices_batch)
        data_type = x['data_type']
        # print(f"data_path: {data_path}")
        if data_type[0] == 'image':
            # preprocessing images in evaluation dimension 1-9
            if args.noise_step is None:
                raw_image_batch = [Image.open(open(dp, "rb")).convert('RGB') for dp in data_path_batch]
                image_batch = raw_image_batch
            else:
                raw_image_batch = [Image.open(open(dp, "rb")) for dp in data_path_batch]
                image_batch = [add_diffusion_noise(self.vis_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'], args.noise_step).cuda() for raw_image in raw_image_batch]
            # image_batch = torch.stack(image_batch, dim=0)
            # image_batch = image_batch.half()
            # pdb.set_trace()


        sample_image_batch = []
        sample_image_label_batch = []
        sample_image_input_id_batch = []
        sample_image_token_type_ids_batch = []
        sample_image_attention_mask_batch = []
        sample_image_cross_images_batch = []
        
        sample_question_label_batch = []
        sample_question_input_id_batch = []
        sample_question_token_type_ids_batch = []
        sample_question_attention_mask_batch = []
        sample_question_cross_images_batch = []

        num_choices_list = []

        loss_list = []
        logits_img_list = []
        logits_ques_list = []

        for i, (question, choices, image) in enumerate(zip(question_batch, choices_batch, image_batch)):

            choice_list = []
            for i, c in enumerate(choices):
                choice_list.append('{}. {}'.format(all_options[i], c))
            choice_txt = '\n'.join(choice_list)

            question = question + '\n' + choice_txt

            num_choices_list.append(len(choices))

            for j, choice in enumerate(choices):


                # img
                history = []

                if args.use_instruction == "True":
                    query = question + args.instruction_prompt
                else:
                    query = question
                input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=query, answer=all_options[j], history=history, images=[image], template_version='vqa')
                DEVICE = self.model.device

                inputs = {
                    'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
                    'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
                    'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
                    'images': [[input_by_model['images'][0].to(self.torch_type).to(DEVICE)]] if image is not None else None,
                }
                if 'cross_images' in input_by_model and input_by_model['cross_images']:
                    inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.torch_type).to(DEVICE)]]


                answer_input_ids = self.tokenizer(all_options[j])['input_ids']
                labels = torch.tensor([-100]*(inputs['input_ids'][0].shape[-1]-len(answer_input_ids)) + answer_input_ids)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logit_img = outputs['logits']
                loss = None
                if labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = logit_img[..., :-1, :].contiguous()
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

                    loss_list.append(loss)
                    logits_img_list.append(torch.exp(-loss))


                # sample_image_batch.append(inputs['images'])
                # sample_image_input_id_batch.append(inputs['input_ids'])
                # sample_image_label_batch.append(label)
                # sample_image_token_type_ids_batch.append(inputs['token_type_ids'])
                # sample_image_attention_mask_batch.append(inputs['attention_mask'])
                # sample_image_cross_images_batch.append(inputs['cross_images'])


                #question
                # text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT: {}" 
                # history = []
                # query = question
                # query = text_only_template.format(query, choice)
                # input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=query, history=history, template_version='base')
                # DEVICE = self.model.device

                # inputs = {
                #     'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
                #     'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
                #     'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
                #     'images': [],
                # }
                # if 'cross_images' in input_by_model and input_by_model['cross_images']:
                #     inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(self.torch_type)]]

                # print(inputs)

                # answer_input_ids = self.tokenizer(choice, add_special_tokens=False)['input_ids']
                # labels = torch.tensor([-100]*(inputs['input_ids'][0].shape[-1]-len(answer_input_ids)) + answer_input_ids)

                # outputs = self.model(**inputs)
                # logit_ques = outputs['logits']
                # print(outputs.keys())

                # loss = None
                # if labels is not None:
                #     # Shift so that tokens < n predict n
                #     shift_logits = logit_ques[..., :-1, :].contiguous()
                #     shift_labels = labels[..., 1:].contiguous()
                #     label_shape = shift_labels.shape
                #     # Flatten the tokens
                #     loss_fct = CrossEntropyLoss(reduce=False)
                #     shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                #     shift_labels = shift_labels.view(-1)
                #     # Enable model parallelism
                #     shift_labels = shift_labels.to(shift_logits.device)
                #     loss = loss_fct(shift_logits, shift_labels)
                #     loss = loss.view(label_shape)
                #     nums_answer_token = torch.count_nonzero(loss, dim=-1)
                #     loss = loss.sum(dim=-1)
                #     loss = loss / nums_answer_token

                #     logits_ques_list.append(torch.exp(-loss))

                # sample_question_input_id_batch.append(inputs['input_ids'])
                # sample_question_label_batch.append(label)
                # sample_question_token_type_ids_batch.append(inputs['token_type_ids'])
                # sample_question_attention_mask_batch.append(inputs['attention_mask'])
                # sample_question_cross_images_batch.append(inputs['cross_images'])

        loss_list = torch.tensor(loss_list)
        logits_img_list = torch.tensor(logits_img_list)
        logits_ques_list = logits_img_list
            
        loss_img = split_list_by_lengths(loss_list, num_choices_list)
        logit_img = split_list_by_lengths(logits_img_list, num_choices_list)
        logit_ques = split_list_by_lengths(logits_ques_list, num_choices_list)

        return loss_img, logit_img, logit_ques



def build(args):
    return MLLM_Tester(args)