'''
需要实现两个函数，
rm_preprocess 对 reward 前处理，
rm_postprocess 对 reward tensor 负责后处理。
'''

import re
import ast
import json
from verl.workers.reward_function.consistency_check_prompt import prompt_v1

LONGCAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ content }}{% elif message['role'] == 'assistant' %}{{ content }}{% endif %}{% endfor %}"

SolutionList_strs = """{"name":"外呼商家-骑手是否返餐","describe":"联系商家，确认骑手是否完成返餐"}
{"name":"外呼商家-骑手申诉原因是否属实","describe":"联系商家，询问骑手表述的申诉原因是否属实"}
{"name":"外呼商家-核实商家是否营业","describe":"联系商家，询问商家是否处于营业状态"}
{"name":"外呼用户-是否收到餐品","describe":"联系用户，询问是否已经收到餐品"}
{"name":"外呼用户-是否需要配送","describe":"联系用户，询问是否需要配送餐品"}
{"name":"外呼用户-确认是否拒收","describe":"联系用户，询问是否拒绝收餐"}
{"name":"外呼下单人-是否需要配送","describe":"联系下单人，询问是否需要配送餐品"}
{"name":"申诉通过","describe":"在骑手符合申诉条件时，通过骑手的申诉请求"}
{"name":"申诉驳回","describe":"在骑手不符合申诉条件时，驳回骑手的申诉请求"}
{"name":"执行补款","describe":"给骑手提供补款，如果涉及具体的金额，需要在名称中写明，格式为“执行补款-xx元”"}
{"name":"跟进二次申诉结果","describe":"骑手进线申诉违规，需要跟进到有二次申诉结果"}
{"name":"跟进返餐","describe":"骑手申请配送费补贴，其他条件都满足只差未返餐，跟进骑手最终返餐情况"}
{"name":"跟进系统补款","describe":"满足所有补款要求，系统正常应该给补款但有可能失败，需要跟进到系统补款完成"}
{"name":"稍后外呼……","describe":"承诺稍后联系用户或者商家，具体场景参考上述外呼方案"}"""
SolutionList = SolutionList_strs.split('\n')

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

def parse_fangan_func(ResponseSolution):
  try:
    if ResponseSolution == "无":
        ResponseSolutionName = "无"
    elif ResponseSolution.startswith("执行补款"):
        ResponseSolutionName = [i for i in SolutionList if "执行补款" in i][0]
    elif ResponseSolution.startswith("跟进返餐"):
        ResponseSolutionName = [i for i in SolutionList if "跟进返餐" in i][0]
    elif ResponseSolution.startswith("跟进系统补款"):
        ResponseSolutionName = [i for i in SolutionList if "跟进系统补款" in i][0]
    else:
        ResponseSolutionName = [i for i in SolutionList if ResponseSolution in i][0]
  except:
    ResponseSolutionName = "无"
  
  return ResponseSolutionName

def rm_preprocess(text_prompts=[], text_responses=[], tgt_tokenizer=None):
    def mk_vllm_input(text_response):
        rparts = text_response.rsplit('}', 1)
        response = rparts[0] + '}'
        lparts = response.split('{', 1)
        response = '{' + lparts[1]
        text_response = json.loads(response.strip())['Response']
        text_thought = json.loads(response.strip())['Thought']
        text_response_solution = json.loads(response.strip())['ResponseSolution']
        # 用于 Judge 的 Prompt 预先写好在了数据集构造阶段，每条数据 Response 都有与之对应的 Prompt
        
        prompt = prompt_v1.replace('{Response}', text_response)
        prompt = prompt.replace('{Thought}', text_thought)
        prompt = prompt.replace('{ResponseSolution}', str(text_response_solution))
        solution_list = text_response_solution.split("；")
        ResponseSolutionName = ""
        # breakpoint()
        for i, solution in enumerate(solution_list):
            ResponseSolutionName_i = parse_fangan_func(solution)
            ResponseSolutionName += f"{ResponseSolutionName_i}\n"
        prompt = prompt.replace('{ResponseSolutionName}', ResponseSolutionName)
        message = [{"role": "user", "content": prompt}]
        if tgt_tokenizer.chat_template == LONGCAT_TEMPLATE or tgt_tokenizer.chat_template is None:
            raw_prompt = 'User: \n' + prompt + '\nAssistant: \n'
            return raw_prompt
        return tgt_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    mk_vllm_inputs = []
    for text_response in text_responses:
        mk_vllm_inputs.append(mk_vllm_input(text_response))
    return mk_vllm_inputs

def fix_json_str(s):
    # 1. 替换全角引号为英文引号
    s = s.replace('“', '"').replace('”', '"')
    # 2. 用正则替换所有key为标准格式（只保留key的内容）
    def key_replacer(match):
        key = match.group(2)
        key = key.strip().replace('"', '')  # 去除key内部多余引号
        return f'{match.group(1)}"{key}":'
    s = re.sub(r'([{,]\s*)([^"\':,{}][^:,{]*?)\s*:', key_replacer, s)
    s = re.sub(r'^{\s*([^"\':,{}][^:,{]*?)\s*:', lambda m: '{"' + m.group(1).strip().replace('"', '') + '":', s)
    return s

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
                    return 0
        # breakpoint()
        try:
            if check_dict_values_are_numeric_and_0_or_1(json_data):
                return calculate_average_of_dict_values(json_data)
            else:
                return 0
        except Exception as e:
            print("No dictionary found in the string.")
            return 0
    rewards = []
    for result in results:
        rewards.append(mk_reward_output(result))
    return rewards

if __name__ == "__main__":
    the_case = ['<think>\n1. Thought核心意图：告知骑手可以返餐后取消订单，并告知申诉途径。  \n   关键行动：指示骑手进行返餐和取消订单，以及告知申诉途径。\n2. Response确实告知骑手进行返餐和取消订单，并提到了申诉途径，因此在核心意图上是保持一致的。\n3. ResponseSolution为“无”，这意味着没有指定任何核心行动或行为对象，因此Response与ResponseSolution之间不存在一致性要求。\n</think>\n{"Thought与Response一致性": 1, "Response与ResponseSolution一致性": 0}']
    rm_postprocess(the_case)
    