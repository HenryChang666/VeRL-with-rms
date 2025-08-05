import re
import json
import requests
import time


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
    sim = compute_similarity(pred_response, tar_response)
    return 0.7 * sim + 0.3

# 总分
def compute_score_bge(solution_str, ground_truth) -> float:
    rparts = solution_str.rsplit('}', 1)
    query = rparts[0] + '}'
    lparts = query.split('{', 1)
    query = '{' + lparts[1]
    target = json.loads(ground_truth)

    try:
        # json格式
        # breakpoint()
        pred_js = json.loads(query.split('Assistant: \n')[-1].strip())
        # 字段缺失【mod】
        if any([x not in pred_js for x in ['RelevantConstraintRuleNumber', 'RelevantKnowledgeNumber', 'Response', 'ResponseSolution', 'DialogueAgreeSolution']]):
            return 0.0
        # 知识编号为列表【mod】
        if type(pred_js['RelevantConstraintRuleNumber']) != list or type(pred_js['RelevantKnowledgeNumber']) != list:
            return 0.0
        
        # 话术分数（0-1）,限定为 格式分数 zrl
        response_score = calc_response_score(pred_js['Response'], target['Response'])
        return response_score
    except:
        # 输出不合法
        return 0.0


