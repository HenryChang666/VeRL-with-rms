from jsonkit import read_jsonl, write_jsonl
import random

result = []
f = read_jsonl('/home/hadoop-mtai/users/zhangxiaoyun15/program/voc_explore/data/test/testdata_paotui_9k_r1_formatted.json')
for i, line in enumerate(f):
    rand = random.random()
    if rand < 0.1:
        result.append(line)

write_jsonl(result, '/home/hadoop-mtai/users/zhangxiaoyun15/program/voc_explore/data/test/testdata_paotui_1k_r1_formatted.json')