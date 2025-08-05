import argparse
import json

from flask import Flask, request, jsonify
from openrlhf.utils.logging_utils import init_logger
app = Flask(__name__)
logger = init_logger(__name__)

# 全局变量，存储输入和标签的对应关系
input2target = {}

def load_data(data_path):
    with open(data_path, "r") as fd:
        for line in fd:
            js = json.loads(line)
            input2target[js['input']] = js['target']

def reward_fn(query:str):
    if query.count("Assistant: \n") != 1:
        logger.info("REWARD PROCESS ERROR: query+response without Assistant !")
        return 0.0
    input = query.split('Assistant: \n')[0] + 'Assistant: \n'
    if input not in input2target:
        logger.info("REWARD PROCESS ERROR: prompt not found in dataset")
        return 0.0
    target = json.loads(input2target[input])
    try:
        pred = json.loads(query.split('Assistant: \n')[-1].strip())
        if pred['command'] == target['command'] and (pred['args'] == target['args'] or pred['command'] in ('inquire', 'task_failed')):
            return 1.0
        else:
            return 0.1
    except:
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
