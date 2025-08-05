import re
import json
import requests
import time
import ast


# 语义相似度
def compute_similarity(sent1, sent2, try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url="http://localhost:5032/get_bge_sim", json={"sent1": sent1, "sent2": sent2}, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert "sim" in response, f"sim not in {response}"
            return response.get("sim")
        except requests.RequestException as e:
            print(f"Request error, please check: {e}")
        except Exception as e:
            print(f"Unexpected error, please check: {e}")
        time.sleep(1)
    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")

# 回复分数计算
def calc_response_score(pred_response, tar_response):
    # 格式检验
    # breakpoint()
    # print("Start calculating sim...")
    if not pred_response.startswith('客服:'):
        return 0.0
    
    # sim = compute_similarity(pred_response, tar_response)
    # return 0.7 * sim + 0.3
    return 1

# 总分
def compute_score(solution_str, ground_truth) -> float:

    # # breakpoint()
    # if solution_str.count("Assistant: \n") != 1:
    #     return 0.0

    # input_str = query.split('Assistant: \n')[0] + 'Assistant: \n'
    target = json.loads(ground_truth)
    # pattern = r"【对话历史】如下：\n(.*?)\n\nAssistant:"
    # match = re.search(pattern, input_str, re.DOTALL)
    # context = match.group(1).strip()
    try:
        rparts = solution_str.rsplit('}', 1)
        query = rparts[0] + '}'
        lparts = query.split('{', 1)
        query = '{' + lparts[1]
    except:
        print("!!!!!!!!!!!! 输出不合法 !!!!!!!!!!!!!", solution_str)
        # 输出不合法
        return 0.0

    try:
        # json格式
        # breakpoint()
        pred_js = json.loads(solution_str.split('Assistant: \n')[-1].strip())
        # 字段缺失【mod】
        if any([x not in pred_js for x in ['RelevantConstraintRuleNumber', 'RelevantKnowledgeNumber', 'Response', 'ResponseSolution', 'DialogueAgreeSolution']]):
            return 0.0
        # 知识编号为列表【mod】
        if type(pred_js['RelevantConstraintRuleNumber']) != list or type(pred_js['RelevantKnowledgeNumber']) != list:
            try:
                RelevantConstraintRuleNumber_lst = ast.literal_eval(pred_js['RelevantConstraintRuleNumber'])
                RelevantKnowledgeNumber_lst = ast.literal_eval(pred_js['RelevantKnowledgeNumber'])
                if type(RelevantConstraintRuleNumber_lst) != list or type(RelevantKnowledgeNumber_lst) != list:
                    return 0.0
            except:
                return 0.0
        
        cot_format_score = 1
        # 不同类型数据特殊奖励（0/1）COT【mod】
        if (target['Thought'] == "" and pred_js['Thought'] != "") or (target['Thought'] != "" and pred_js['Thought'] == ""):
            cot_format_score = 0.1
        elif len(pred_js['Thought']) >= 1300:
            cot_format_score = 0.1
        
        # 话术分数（0-1）,限定为 格式分数 zrl
        response_score = calc_response_score(pred_js['Response'], target['Response'])

        # 回复方案分数(0.1 / 1)
        is_solution = pred_js['ResponseSolution'] == target['ResponseSolution'] and pred_js['DialogueAgreeSolution'] == target['DialogueAgreeSolution']
        # 知识编号正确性(0.1 / 1)【mod】
        is_rule_know = pred_js['RelevantConstraintRuleNumber'] == target['RelevantConstraintRuleNumber'] and pred_js['RelevantKnowledgeNumber'] == target['RelevantKnowledgeNumber']
        solution_score = 1.0 if (is_solution and is_rule_know) else 0.1
        return cot_format_score * response_score * solution_score
    except:
        # 输出不合法
        return 0.0


