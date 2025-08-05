import argparse
import os
import numpy as np
from collections import Counter
import re
from jsonkit import read_jsonl, write_jsonl
import random

def extend_to_next_256(lst):
    """扩展列表到下一个256的倍数"""
    current_length = len(lst)
    next_multiple = ((current_length + 255) // 256) * 256
    if current_length == next_multiple:
        return lst
    
    num_to_add = next_multiple - current_length
    if not lst:
        raise ValueError("输入列表为空，无法进行扩展。")
    
    additional_elements = random.choices(lst, k=num_to_add)
    return lst + additional_elements

def target_parse(target):
    """解析目标为int, float或non-numeric"""
    try:
        return int(target), "int"
    except ValueError:
        try:
            return float(target), "float"
        except ValueError:
            return None, "non-numeric"

def extract_answer_text(text):
    """从文本中提取答案"""
    pattern = r'\\boxed{(.*?)}'
    return re.findall(pattern, text, re.DOTALL)

def split_options(text):
    """根据多种分隔符分割选项"""
    separators = [',', '，', ' ', ';']
    for sep in separators:
        if sep in text:
            return set(text.split(sep))
    return {text}

def calculate_accuracy(input_data, target, prob):
    """计算生成结果的准确性"""
    count, acc = 0, 0
    for generated_text in input_data["generated"]:
        count += 1
        generated = "<think>\n" + generated_text
        answers = extract_answer_text(generated)
        if not answers:
            continue
        
        answer = answers[-1]
        _, target_type = target_parse(target)
        
        if target_type in ["int", "float"]:
            if answer == target:
                acc += 1
        else:
            target_set, answer_set = split_options(target), split_options(answer)
            if (len(target_set) > 1 or len(answer_set) > 1) and target_set == answer_set:
                acc += 1
            elif target == answer:
                acc += 1

    return count, acc

def main(args):
    inputs = list(read_jsonl(args.input))
    outputs = []
    cnt = Counter()
    
    for input_data in inputs:
        target = input_data["target"]
        count, acc = calculate_accuracy(input_data, target, args.prob)
        
        if count == 0:
            print("count is 0")
            continue
        
        print(count, acc, acc / count)
        cnt[acc] += 1
        if acc / count <= args.prob:
            outputs.append({"input": input_data["input"], "target": target})
    
    outputs = extend_to_next_256(outputs)
    print(cnt)
    print(len(outputs))
    write_jsonl(outputs, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input path", default="")
    parser.add_argument("--output", type=str, help="output path", default="")
    parser.add_argument("--prob", type=float, default=0.79, help="probability threshold")
    args = parser.parse_args()
    
    args.input = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangxiaoyun15/program/voc_explore/.tmp/DeepSeek-R1-Distill-Qwen-14B/train_data_distill_r1_110k_filtered_MATH_Exam_gpt4o_consis_len4205_source_r1_format_stage_99.json"
    args.output = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangxiaoyun15/program/voc_explore/data/train/DeepSeek-R1-Distill-Qwen-14B/train_data_distill_r1_110k_filtered_MATH_Exam_gpt4o_consis_len4205_source_r1_format_stage_99.json"
    
    main(args)

