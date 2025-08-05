import numpy as np
from collections import Counter

# 给定的次数分布
cnt = Counter({10: 1216, 0: 898, 9: 491, 8: 271, 7: 231, 1: 230, 5: 187, 6: 183, 4: 171, 3: 169, 2: 158})
##### 低于probs的数据需要 sample len(outputs) 的数据
sample_size = 100
filtered_counts = Counter({k: cnt[k] for k in [10, 9, 8] if k in cnt}) # 难度较低的数据cnt
total_samples = sum(filtered_counts.values())
print(filtered_counts, "total_samples", total_samples)
probs = {k: v / total_samples for k, v in filtered_counts.items()}
inverse_probs = {k: 1 / v for k, v in probs.items()} # # 计算反转后的概率分布, 使得难度越低越难采样到
total_inverse = sum(inverse_probs.values())
normalized_inverse_probs = {k: v / total_inverse for k, v in inverse_probs.items()}
# 将反转后的概率分布转换为两个列表，一个是次数列表，一个是对应的概率列表
times = list(normalized_inverse_probs.keys())
probabilities = list(normalized_inverse_probs.values())
# 从反转后的概率分布中进行不重复的加权抽样
sampled_values = np.random.choice(times, size=sample_size, p=probabilities)
print(sampled_values)