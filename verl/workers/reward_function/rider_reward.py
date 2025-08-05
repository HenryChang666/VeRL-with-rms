'''
需要实现两个函数，
rm_preprocess 对 reward 前处理，
rm_postprocess 对 reward tensor 负责后处理。
'''

import re
import ast
import json

LONGCAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ content }}{% elif message['role'] == 'assistant' %}{{ content }}{% endif %}{% endfor %}"

def is_numeric_value(value):
    try:
        num = float(value)
        return True
    except (ValueError, TypeError):
        return False

def check_dict_values_are_numeric_and_0_or_1(input_dict):
    for value in input_dict.values():
        if not is_numeric_value(value):
            return False
        num = float(value)
        if num not in [0, 1]:
            return False
    return True

def calculate_average_of_dict_values(input_dict):
    total = 0
    count = 0
    for value in input_dict.values():
        if is_numeric_value(value):
            total += float(value)
            count += 1
    if count == 0:
        return 0
    return total / count

def fix_json_str(s):
    # 1. 替换全角引号为英文引号
    s = s.replace('“', '"').replace('”', '"')
    s = s.replace("\\", '')
    # 2. 用正则替换所有key为标准格式（只保留key的内容）
    def key_replacer(match):
        key = match.group(2)
        key = key.strip().replace('"', '')  # 去除key内部多余引号
        return f'{match.group(1)}"{key}":'
    s = re.sub(r'([{,]\s*)([^"\':,{}][^:,{]*?)\s*:', key_replacer, s)
    s = re.sub(r'^{\s*([^"\':,{}][^:,{]*?)\s*:', lambda m: '{"' + m.group(1).strip().replace('"', '') + '":', s)
    return s

def rm_preprocess(text_prompts, text_responses, tgt_tokenizer):
    def mk_vllm_input(text_prompt, text_response):
        rparts = text_response.rsplit('}', 1)
        response = rparts[0] + '}'
        lparts = response.split('{', 1)
        response = '{' + lparts[1]
        text_response = json.loads(response.strip())['Response']
        # 用于 Judge 的 Prompt 预先写好在了数据集构造阶段，每条数据 Response 都有与之对应的 Prompt
        prompt = text_prompt.replace('{response}', text_response)
        message = [{"role": "user", "content": prompt}]
        if tgt_tokenizer.chat_template == LONGCAT_TEMPLATE or tgt_tokenizer.chat_template is None:
            raw_prompt = 'User: \n' + prompt + '\nAssistant: \n'
            return raw_prompt
        return tgt_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    mk_vllm_inputs = []
    for text_prompt, text_response in zip(text_prompts, text_responses):
        mk_vllm_inputs.append(mk_vllm_input(text_prompt, text_response))
    # breakpoint()
    return mk_vllm_inputs


def rm_postprocess(results):
    keywords_0 = ['方案/信息错误', '信息准确性']
    keywords_1 = ['有害性']
    def mk_reward_output(string):
        think_index = string.find("</think>")
        if think_index != -1:
            string = string[think_index + len("</think>"):]
        # 提取大括号包裹的内容
        pattern = r'\{[^{}]*\}'
        json_strs = re.findall(pattern, string, re.DOTALL)
        if json_strs:
            json_str = json_strs[-1].strip()
            # 尝试直接解析
            try:
                json_data = json.loads(json_str)
            except Exception:
                # 修正非标准JSON
                json_str_fixed = fix_json_str(json_str)
                try:
                    json_data = json.loads(json_str_fixed)
                except Exception as e:
                    print("解析失败的json字符串：", json_str)
                    print("修正后：", json_str_fixed)
                    print("异常信息：", e)
                    return -1
        try:
            for key in json_data.keys():
                if any(keyword in key for keyword in keywords_0):
                    return 0
                if any(keyword in key for keyword in keywords_1):
                    return -1
            # breakpoint()
            if check_dict_values_are_numeric_and_0_or_1(json_data):
                return calculate_average_of_dict_values(json_data)
            else:
                return 0
        except:
            print("No dictionary found in the string.")
            return 0
    rewards = []
    # breakpoint()
    for result in results:
        rewards.append(mk_reward_output(result))
    return rewards

if __name__ == "__main__":
    the_case = ['它在表述上也直接指向了骑手的问题，没有出现提前结束服务的意图。\\{\"维度一\": 1, \"维度二\": 1, \"维度三\": 1, \"维度四\": 1, \"维度五\": 1\\}']
    # the_case[0] = the_case[0].replace("\\", '')
    print(rm_postprocess(the_case))