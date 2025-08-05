import argparse
import json

from flask import Flask, request, jsonify
from openrlhf.utils.logging_utils import init_logger
app = Flask(__name__)
logger = init_logger(__name__)

# 全局变量，存储输入和标签的对应关系
input2target = {}
input2context = {}

# bge模型加载
from transformers import AutoTokenizer, AutoModel
import torch
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhanghaoxing/model_hub/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhanghaoxing/model_hub/bge-large-zh-v1.5')
model.eval()


def load_data(data_path):
    with open(data_path, "r") as fd:
        for line in fd:
            js = json.loads(line)
            input2target[js['input']] = js['target']
            # input2context[js['input']] = js['context']
            input2context[js['input']] = js['input']
            
# 重复判断
def check_duplicate(pred_content, history_contents):
    def jarccard_similarity(str1, str2):
        set1 = set(str1)
        set2 = set(str2)
        return len(set1 & set2) / len(set1 | set2)
    duplicate = False
    over_85_num = 0
    over_95_num = 0
    for content in history_contents:
        if jarccard_similarity(pred_content, content) > 0.85:
            over_85_num += 1
        if jarccard_similarity(pred_content, content) > 0.95:
            over_95_num += 1
    if over_85_num >= 2 or over_95_num >= 1:
        duplicate = True
    return duplicate

# 风险判断
def check_risk(pred_content):
    # TODO 待实现
    return False
    
# 应安抚未安抚
def check_comfort(pred_content, history_contents):
    # TODO 待实现
    return False

# 答非所问
def check_answer(pred_content, tar_content):
    # TODO 待实现
    return False

# 编造信息
def check_fake(pred_content):
    # TODO 待实现
    return False

# 语义相似度
def compute_similarity(sent1, sent2):
    encoded_input = tokenizer([sent1, sent2], padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    sim = (sentence_embeddings[0]*sentence_embeddings[1]).sum().item()
    if sim <=0.45:
        return 0.1
    elif sim >= 0.85:
        return 1.0
    else:
        # 0.5=>0.2; 0.8=>0.9
        return 0.0163*150.4**sim

# 回复分数计算
def calc_response_score(pred_response, tar_response, context):
    # 格式检验
    if not pred_response.startswith('客服:'):
        return 0.0
    pred_content = pred_response.split('客服:', 1)[1].strip()
    tar_content = tar_response.split('客服:', 1)[1].strip()
    history_contents = []
    for line in context.split('\n'):
        if line.startswith('客服:'):
            history_contents.append(line.split('客服:', 1)[1].strip())
    # 重复判断
    duplicate = check_duplicate(pred_content, history_contents)
    # 风险判断
    risk = check_risk(pred_content)
    # 应安抚未安抚
    comfort = check_comfort(pred_content, history_contents)
    # 答非所问
    answer = check_answer(pred_content, tar_content)
    # 编造信息
    fake = check_fake(pred_content)
    # 语义相似度
    sim = compute_similarity(pred_content, tar_content)
    return sim * (1-duplicate*0.5)


def reward_fn(query:str):
    split_token = "候选分类，输出：\n"
    
    if query.count(split_token) != 1:
        logger.info("REWARD PROCESS ERROR: query+response without Assistant !")
        return 0.0
    input = query.split(split_token)[0] + split_token
    if input not in input2target:
        logger.info("REWARD PROCESS ERROR: prompt not found in dataset")
        return 0.0
    print('------target-----', input2target[input])
    target = json.loads(input2target[input])
    context = input2context[input]
    try:
        pred_js = json.loads(query.split(split_token)[-1].strip())
        
        # 字段缺失
        if any([x not in pred_js for x in ['thought', 'yes_no']]):
            return 0.0
        # 话术分数（0-1）
        # response_score = calc_response_score(pred_js['Response'], target['Response'], context)
        # 回复方案分数(0.2 / 1)
        solution_score = 1.0 if (pred_js['yes_no'] == target['yes_no']) else 0.2  # # 格式正确即可得分0.2
        return solution_score
        # return solution_score + 0.2 * response_score
    except:
        # 输出不合法
        return 0.0
        
        
    
@app.route('/get_reward', methods=['POST'])
def get_reward():
    data = request.json
    queries = data.get('query')
    
    scores = [reward_fn(query) for query in queries]
    
    return jsonify({"rewards": scores})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="data path containing the queries and target responses")
    args = parser.parse_args()
    load_data(args.data_path)
    
    app.run(host='0.0.0.0', port=5019)
