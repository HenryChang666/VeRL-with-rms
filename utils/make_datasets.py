import argparse
import os
import numpy as np
from collections import Counter
import re
from jsonkit import read_jsonl, write_jsonl
import random
import os

def extend_to_next_256(lst):
    current_length = len(lst)
    # 计算下一个256的倍数
    next_multiple = ((current_length + 255) // 256) * 256  # 确保向上取整到256的倍数
    if current_length == next_multiple:
        return lst  # 已经是256的倍数，无需扩展
    
    # 需要添加的元素数量
    num_to_add = next_multiple - current_length
    
    if not lst:
        raise ValueError("输入列表为空，无法进行扩展。")
    
    # 随机选择元素进行复制
    # 使用random.choices可以一次性选择多个元素，有放回
    additional_elements = random.choices(lst, k=num_to_add)
    extended_lst = lst + additional_elements
    return extended_lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input path", default="")
    parser.add_argument("--output", type=str, help="output path", default="")
    parser.add_argument("--prob", type=float, default=0.79, help="number of samples per task")
    args = parser.parse_args()
    
    # args.input = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangxiaoyun15/program/voc_explore/.tmp/DeepSeek-R1-Distill-Qwen-14B/train_data_distill_r1_110k_filtered_MATH_Exam_gpt4o_consis_len4205_source_r1_format_stage_99.json"
    # args.output = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangxiaoyun15/program/voc_explore/data/train/DeepSeek-R1-Distill-Qwen-14B/train_data_distill_r1_110k_filtered_MATH_Exam_gpt4o_consis_len4205_source_r1_format_stage_99.json"

    inputs = list(read_jsonl(args.input))
    outputs = []
    outputs_less_probs = []
    cnt = Counter()
    for input in inputs:
        target = input["target"]
        
        def extract_answer_text(text):
            pattern = r'<answer>(.*?)</answer>'
            matches = re.findall(pattern, text, re.DOTALL)  # re.DOTALL 允许 . 匹配换行符
            return matches
        
        count = 0
        acc = 0
        for i in input["generated"]: 
            count += 1
            generated = "<think>\n" + i
            third_pattern = r'^<think>.*?</think>.*?<answer>.*?</answer>.*?$' 
            if re.match(third_pattern, generated, re.DOTALL):
                answers = extract_answer_text(generated)
                if len(answers) == 0:
                    print("answers not in prompt")
                    continue
        
                answer = "否" if "否" in answers[0] else ("是" if "是" in answers[0] else None)
                if answer == target:
                    acc += 1
                
            else:
                print("answers not in prompt")
                continue
            
        if count == 0:
            print("count is 0")
            continue
        
        # print(count, acc, acc / count)
        
        cnt[acc] += 1
        if acc / count <= args.prob:
           outputs.append({"input": input["input"], "target": target, "pass@10": acc })
        else:
            outputs_less_probs.append({"input": input["input"], "target": target, "pass@10": acc})
    
    print("outputs - 有难度", len(outputs))
    
    
    file_name, file_post = os.path.splitext(os.path.basename(args.output))
    
    output_1 = os.path.dirname(args.output) + '/' + file_name + "_hard." + file_post
    write_jsonl(outputs, output_1)
    
    # 创建一个字典，将 outputs_less_probs 中的每个项按照 pass@10 的值进行分组
    grouped_outputs = {}
    for item in outputs_less_probs:
        pass_value = item["pass@10"]
        if pass_value not in grouped_outputs:
            grouped_outputs[pass_value] = []
        grouped_outputs[pass_value].append(item)   
    
    sample_size = len(outputs) // 8
    
    # 从每个分组中随机选择样本，并追加到列表 outputs 中
    for group in grouped_outputs.values():
        num_samples = min(sample_size, len(group))
        sampled_items = random.sample(group, num_samples)
        outputs.extend(sampled_items)
    
    print("outputs - 加入难度小的样本", len(outputs))
    
    output_2 = os.path.dirname(args.output) + '/' + file_name + "_hard_add_simpler." + file_post
    write_jsonl(outputs, output_2)
    
    outputs = extend_to_next_256(outputs)
    print(cnt)
    
    # write_jsonl(outputs, args.output)
    
    output_3 = os.path.dirname(args.output) + '/' + file_name + "_hard_add_simpler_extend256." + file_post
    write_jsonl(outputs, output_3)
    
    print("outputs - 拓展成256的倍数后", len(outputs))
    
    # ##### 低于probs的数据需要 sample len(outputs) 的数据
    # sample_size = len(outputs) // 8
    # filtered_counts = Counter({k: cnt[k] for k in [10, 9, 8] if k in cnt}) # 难度较低的数据cnt
    # total_samples = sum(filtered_counts.values())
    # print(filtered_counts, "total_samples", total_samples)
    # probs = {k: v / total_samples for k, v in filtered_counts.items()}
    # inverse_probs = {k: 1 / v for k, v in probs.items()} # # 计算反转后的概率分布, 使得难度越低越难采样到
    # total_inverse = sum(inverse_probs.values())
    # normalized_inverse_probs = {k: v / total_inverse for k, v in inverse_probs.items()}
    # # 将反转后的概率分布转换为两个列表，一个是次数列表，一个是对应的概率列表
    # times = list(normalized_inverse_probs.keys())
    # probabilities = list(normalized_inverse_probs.values())
    # # 从反转后的概率分布中进行不重复的加权抽样
    # sampled_values = np.random.choice(times, size=sample_size, p=probabilities)
    
    # for value in sampled_values:
    #     if value in grouped_outputs and grouped_outputs[value]:
    #         random_index = random.randint(0, len(grouped_outputs[value]) - 1)
    #         outputs.append(grouped_outputs[value].pop(random_index))
            
    # ########
    # print("outputs - 加入难度小的样本", len(outputs))
    
    
    
