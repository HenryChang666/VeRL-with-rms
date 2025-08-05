import argparse
import json
import re

from flask import Flask, request, jsonify
app = Flask(__name__)

# 全局变量，存储输入和标签的对应关系
input2target = {}
input2context = {}



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

def extract_answer_text(text):
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL)  # re.DOTALL 允许 . 匹配换行符
    return matches


def reward_fn(query:str):

    format_reward = -0.2
    answer_reward = -1.0

    split_token = "<｜Assistant｜><think>\n"
    
    if query.count(split_token) != 1:
        return format_reward + answer_reward
    input = query.split(split_token)[0] + split_token


    if input not in input2target:
        return format_reward + answer_reward
    
    target = input2target[input]

    try:
        generated = "<think>\n" + query.split(split_token)[-1].strip()
        best_pattern = r'^<think>.*?</think>.*?<answer>.*?</answer>.*?$'   


        if re.match(best_pattern, generated, re.DOTALL):
            format_reward = abs(format_reward)
            answers = extract_answer_text(generated)
            if len(answers) == 0:
                return format_reward + answer_reward
            
            answer = "否" if "否" in answers[0] else ("是" if "是" in answers[0] else None)

            if answer == target:
                answer_reward = abs(answer_reward)

            return answer_reward + format_reward

        return format_reward + answer_reward
        
    except:
        # 输出不合法
        return format_reward + answer_reward
        
        
    
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
