import requests
import json

# 定义请求的 URL
url = 'http://localhost:5019/get_reward'


query = []

with open("/home/hadoop-mtai/users/zhangxiaoyun15/program/OpenRLHF4VOCnew/data/voc/cot/test.json", "r") as fd:
    for line in fd:
        js = json.loads(line)
        query.append(js['input'] + "\n</think>xxxxx<answer>hello</answer>xxxxxxx")


# 准备请求数据
payload = {
    "query": query
}

# 设置请求头
headers = {
    'Content-Type': 'application/json'
}

# 发送 POST 请求
response = requests.post(url, headers=headers, data=json.dumps(payload))

# 检查响应状态码
if response.status_code == 200:
    print("Success:", response.json())
else:
    print("Error:", response.status_code, response.text)
