path1='/data/stzhao_data/mllm_ckpt/llava_v1.5_7B'
path2='/data/stzhao_data/mllm_ckpt/LLaVA-Lightning-7B-vicuna-v1-1'
path3='/data/stzhao_data/mllm_ckpt/LLaVA-Vucina-13B'
path4='/data/stzhao_data/mllm_ckpt/llava-v1.6-vicuna-7b'
path5='/data/stzhao_data/mllm_ckpt/llava-v1.6-vicuna-13b'
path6='/data/stzhao_data/mllm_ckpt/llava-v1.5-13b'

path7='/data/stzhao_data/mllm_ckpt/Yi-VL-6B'
path8='/data/stzhao_data/mllm_ckpt/Yi-VL-34B'

path9='/data/stzhao_data/mllm_ckpt/Qwen-VL-Chat'

path10='/data/stzhao_data/mllm_ckpt/InternVL/InternVL-Chat-ViT-6B-Vicuna-7B'
path11='/data/stzhao_data/mllm_ckpt/InternVL/InternVL-Chat-ViT-6B-Vicuna-13B'
path12='/data/stzhao_data/mllm_ckpt/InternVL/InternVL-Chat-ViT-6B-Vicuna-13B-448px'
path13='/data/stzhao_data/mllm_ckpt/InternVL/InternVL-Chat-Chinese-V1-2'

path14='/data/stzhao_data/mllm_ckpt/cogagent-vqa-hf'

path15='/data/stzhao_data/mllm_ckpt/internlm-xcomposer2-7b'
path16='/data/stzhao_data/mllm_ckpt/internlm-xcomposer2-vl-7b'

path17='/data/stzhao_data/mllm_ckpt/SPHINX/SPHINX-MoE-1k'

for ds in mmvp mmbench_dev okvqa_val vqav2_val vizwiz_val gqa_testdev winoground clevr scienceqa_test seedbench
do
for modelpath in ${path1} ${path2} ${path3} ${path4} ${path5} ${path6}
do
python /home/ubuntu/stzhao/LiCo/inference.py \
    --model-path ${modelpath} \
    --ds ${ds} \
    --conv-mode llava_v1 \
    --cuda "0,1,2,3" \
    --batchsize 1 \
    --multi_choices True \
    --use_instruction True \
    --negative True \
    --instruction_prompt "\nGive me the wrong answer."
done
done

# for ds in pope_popular pope_random
# do
# for modelpath in ${path1} ${path2} ${path3} ${path4} ${path5} ${path6}
# do
# python /home/ubuntu/stzhao/LiCo/inference.py \
#     --model-path ${modelpath} \
#     --ds ${ds} \
#     --conv-mode llava_v1 \
#     --cuda "0,1,2,3" \
#     --batchsize 1 \
#     --multi_choices False \
#     --use_instruction True \
#     --negative True \
#     --instruction_prompt "\nGive me the wrong answer."
# done
# done

# for ds in mmvp mmbench_dev okvqa_val vqav2_val vizwiz_val gqa_testdev winoground clevr scienceqa_test seedbench
# do
# for modelpath in ${path9}
# do
# python /home/ubuntu/stzhao/LiCo/inference.py \
#     --model-path ${modelpath} \
#     --ds ${ds} \
#     --conv-mode llava_v1 \
#     --cuda "0,1,2,3" \
#     --batchsize 1 \
#     --multi_choices True \
#     --use_instruction True \
#     --negative True \
#     --instruction_prompt "\nGive me the wrong answer."
# done
# done

# for ds in mmvp mmbench_dev okvqa_val vqav2_val vizwiz_val gqa_testdev winoground clevr scienceqa_test seedbench
# do
# for modelpath in ${path9}
# do
# python /home/ubuntu/stzhao/LiCo/inference.py \
#     --model-path ${modelpath} \
#     --ds ${ds} \
#     --conv-mode llava_v1 \
#     --cuda "0,1,2,3" \
#     --batchsize 1 \
#     --multi_choices True \
#     --use_instruction False \
#     --negative False \
#     --instruction_prompt "\nGive me the wrong answer."
# done
# done

# mmvp pope_adversarial pope_random pope_popular mme vsr okvqa_val vqav2_val vizwiz_val gqa_testdev winoground mmbench_dev clevr scienceqa_test

# llava   conv-mode: llava_v1
# yi  conv-model mm_default
# internvl vicuna_v1

# llava env: llava    pip install transformers==4.31  
# Yi-VL env: Yi   pip install transformers==4.36.0
# InternVL env: pip install transformers==4.36.2

# \nAnswer with the option's letter from the given choices directly.
# mmvp mmbench_dev okvqa_val vqav2_val vizwiz_val gqa_testdev winoground clevr scienceqa_test seedbench

# \nAnswer the question using a single word or phrase.
# mme vsr pope_adversarial pope_random pope_popular

# \nGive me the wrong answer.