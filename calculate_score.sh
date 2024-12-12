for ds in vqav2_val vizwiz_val gqa_testdev
do
python likelihood_composition.py \
    --logits_recorded_base_path_1 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.6-vicuna-7b_MultiChoices/${ds} \
    --logits_recorded_base_path_2 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.6-vicuna-7b_Negative_Instruction_MultiChoices/${ds} \
    --logits_recorded_base_path_3 /home/ubuntu/stzhao/LiCo/logits_recorded/Yi-VL-34B/${ds} \
    --logits_recorded_base_path_4 /home/ubuntu/stzhao/LiCo/logits_recorded/InternVL-Chat-ViT-6B-Vicuna-13B-448px/${ds} \
    --logits_recorded_base_path_5 /home/ubuntu/stzhao/LiCo/logits_recorded/Qwen-VL-Chat/${ds} \
    --logits_recorded_base_path_6 /home/ubuntu/stzhao/LiCo/logits_recorded/LLaVA-Lightning-7B-vicuna-v1-1_MultiChoices/${ds} \
    --logits_recorded_base_path_7 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.6-vicuna-13b_Negative_Instruction_MultiChoices/${ds} \
    --logits_recorded_base_path_8 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.6-vicuna-7b_Negative_Instruction_MultiChoices/${ds} \
    --logits_recorded_base_path_9 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.5-13b_Negative_Instruction_MultiChoices/${ds} \
    --logits_recorded_base_path_10 /home/ubuntu/stzhao/LiCo/logits_recorded/llava_v1.5_7B_Negative_Instruction_MultiChoices/${ds} \
    --logits_recorded_base_path_11 /home/ubuntu/stzhao/LiCo/logits_recorded/LLaVA-Vucina-13B_Negative_Instruction_MultiChoices/${ds} \
    --logits_recorded_base_path_12 /home/ubuntu/stzhao/LiCo/logits_recorded/LLaVA-Lightning-7B-vicuna-v1-1_Negative_Instruction_MultiChoices/${ds} \
    --ds ${ds} \
    --normalize_method softmax \
    --nums_ensemble 5 \
    --weight_path /home/ubuntu/stzhao/LiCo/dev50/weight/soup3_dev50.npy \
    --method contrast
done

# mme mmvp mmbench_dev pope_adversarial pope_random pope_popular scienceqa_test vsr okvqa_val vqav2_val vizwiz_val gqa_testdev winoground clevr dev50
# mme vsr pope_adversarial pope_popular pope_random
# mmvp mmbench_dev okvqa_val vqav2_val vizwiz_val gqa_testdev
# ensemble debias contrast

    # --logits_recorded_base_path_1 /home/ubuntu/stzhao/LiCo/logits_recorded/LLaVA-Lightning-7B-vicuna-v1-1_MultiChoices/${ds} \
    # --logits_recorded_base_path_2 /home/ubuntu/stzhao/LiCo/logits_recorded/LLaVA-Vucina-13B_MultiChoices/${ds} \
    # --logits_recorded_base_path_3 /home/ubuntu/stzhao/LiCo/logits_recorded/llava_v1.5_7B_MultiChoices/${ds} \
    # --logits_recorded_base_path_4 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.5-13b_MultiChoices/${ds} \
    # --logits_recorded_base_path_5 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.6-vicuna-7b_MultiChoices/${ds} \
    # --logits_recorded_base_path_6 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.6-vicuna-13b_MultiChoices/${ds} \
    # --logits_recorded_base_path_7 /home/ubuntu/stzhao/LiCo/logits_recorded/LLaVA-Lightning-7B-vicuna-v1-1_Negative_Instruction_MultiChoices/${ds} \
    # --logits_recorded_base_path_8 /home/ubuntu/stzhao/LiCo/logits_recorded/LLaVA-Vucina-13B_Negative_Instruction_MultiChoices/${ds} \
    # --logits_recorded_base_path_9 /home/ubuntu/stzhao/LiCo/logits_recorded/llava_v1.5_7B_Negative_Instruction_MultiChoices/${ds} \
    # --logits_recorded_base_path_10 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.5-13b_Negative_Instruction_MultiChoices/${ds} \
    # --logits_recorded_base_path_11 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.6-vicuna-7b_Negative_Instruction_MultiChoices/${ds} \
    # --logits_recorded_base_path_12 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.6-vicuna-13b_Negative_Instruction_MultiChoices/${ds} \

    
    # --logits_recorded_base_path_1 /home/ubuntu/stzhao/LiCo/logits_recorded/LLaVA-Lightning-7B-vicuna-v1-1/${ds} \
    # --logits_recorded_base_path_2 /home/ubuntu/stzhao/LiCo/logits_recorded/LLaVA-Vucina-13B/${ds} \
    # --logits_recorded_base_path_3 /home/ubuntu/stzhao/LiCo/logits_recorded/llava_v1.5_7B/${ds} \
    # --logits_recorded_base_path_4 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.5-13b/${ds} \
    # --logits_recorded_base_path_5 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.6-vicuna-7b/${ds} \
    # --logits_recorded_base_path_6 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.6-vicuna-13b/${ds} \
    # --logits_recorded_base_path_7 /home/ubuntu/stzhao/LiCo/logits_recorded/LLaVA-Lightning-7B-vicuna-v1-1_Negative_Instruction/${ds} \
    # --logits_recorded_base_path_8 /home/ubuntu/stzhao/LiCo/logits_recorded/LLaVA-Vucina-13B_Negative_Instruction/${ds} \
    # --logits_recorded_base_path_9 /home/ubuntu/stzhao/LiCo/logits_recorded/llava_v1.5_7B_Negative_Instruction/${ds} \
    # --logits_recorded_base_path_10 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.5-13b_Negative_Instruction/${ds} \
    # --logits_recorded_base_path_11 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.6-vicuna-7b_Negative_Instruction/${ds} \
    # --logits_recorded_base_path_12 /home/ubuntu/stzhao/LiCo/logits_recorded/llava-v1.6-vicuna-13b_Negative_Instruction/${ds} \