# 类型1：高精度需求（如安全相关）
safety_config = {
    'tp': 3.0,  # 鼓励正确识别
    'fp': -0.5,  # 允许少量误报
    'fn': -5.0,  # 严格惩罚漏报
    'tn': 0.1
}

# 类型2：高召回需求（如推荐系统）
recall_config = {
    'tp': 2.0,
    'fp': -2.0,  # 严格限制误推荐
    'fn': -1.0,  # 允许少量漏推荐
    'tn': 0.5
}

# 类型3：平衡型配置
balanced_config = {
    'tp': 2.0,
    'fp': -1.0,
    'fn': -1.0,
    'tn': 0.3
}

# 可以为每个事件定制化rewards
class_configs = {
    'A': {'tp': 2.0, 'fp': -1.5, 'fn': -2.0, 'tn': 0.3},
    'B': {'tp': 2.2, 'fp': -1.8, 'fn': -2.5, 'tn': 0.2},
    'C': {'tp': 1.8, 'fp': -2.0, 'fn': -3.0, 'tn': 0.4}
    }

# 动态权重调整（基于验证集表现）
def update_config_based_on_f1(class_name, current_f1):
    # F1低于阈值时增强惩罚
    if current_f1 < 0.7:
        class_configs[class_name]['fn'] *= 1.5
        class_configs[class_name]['fp'] *= 1.2
    # F1较高时保持平衡
    else:
        class_configs[class_name]['tp'] += 0.1
        class_configs[class_name]['tn'] += 0.05


def per_classifier_reward(y_true, y_pred, class_config):
    """
    单样本单分类器奖励计算
    :param y_true: 实际是否属于该类别 (True/False)
    :param y_pred: 模型预测结果 (True/False)
    :param class_config: 该分类器的权重配置
    """
    if y_true and y_pred:  # TP [y_true=是 ｜ y_pred=是] 2
        return class_config['tp']
    elif not y_true and y_pred:  # FP [y_true=否 ｜ y_pred=是] -1
        return class_config['fp']
    elif y_true and not y_pred:  # FN [y_true=是 ｜ y_pred=否] -1
        return class_config['fn']
    else:  # TN
        return class_config['tn'] # # TN [y_true=否 ｜ y_pred=否] 0.3

# F1 = 2TP/(2TP + FP + FN)
# TP+FN=support

sample1_reward = per_classifier_reward(
    y_true=True,  # "属于A"
    y_pred=True,  # 预测为A
    class_config=class_configs['A']
)  # 返回 2.0 (TP)

print(sample1_reward)

sample3_reward = per_classifier_reward(
    y_true=False,  # "不属于C"
    y_pred=False,  # 预测不为C
    class_config=class_configs['C']
)  # 返回 0.4 (TN)

print(sample3_reward)

# F1 = 2TP/(2TP + FP + FN)