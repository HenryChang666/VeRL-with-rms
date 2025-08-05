import re

def extract_answer_text(text):
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL)  # re.DOTALL 允许 . 匹配换行符
    return matches

def compute_score(solution_str, ground_truth) -> float:
    format_reward = -0.2  # 默认格式错误
    penalty_answer_reward = -1.
    answer_reward = 0
    query = solution_str
    target = ground_truth

    try:
        generated = "<think>\n" + query
        best_pattern = r'^<think>.*?</think>.*?<answer>.*?</answer>.*?$'
        # 检查格式正确性
        if re.match(best_pattern, generated, re.DOTALL):
            format_reward = abs(format_reward)
            # 提取答案并计算答案奖励
            answers = extract_answer_text(generated)
            if len(answers) == 0:
                return format_reward + penalty_answer_reward # # 无法提取答案但是格式正确
            else:
                answer_text = answers[0].strip().lower()
                answer = "否" if "否" in answer_text else ("是" if "是" in answer_text else None)

            # 根据四个条件设置奖励
            # TP： [y_true=是 ｜ y_pred=是] 设置为rew=2
            # FN： [y_true=是 ｜ y_pred=否] -2.   # （惩罚漏报）
            # FP： [y_true=否 ｜ y_pred=是] -1.5   # （惩罚误报）
            # TN： [y_true=否 ｜ y_pred=否] 0.1
            if target == "是":
                answer_reward = 2 if answer == "是" else -2
            else:
                answer_reward = -1.5 if answer == "是" else 0.1
                
            return format_reward + answer_reward
            
        else:
            return format_reward + penalty_answer_reward

    except:
        return format_reward + penalty_answer_reward



if __name__ == "__main__":
    prompt = """
    ### 输出要求
- 请一步步推理和思考后给出判断。初步思考结束后可以进行假设、反思和评估验证当前思考过程和答案，以得出更准确的结论。思考过程放在<think></think>内部，例如：<think>你的假设、反思、评估验证的思考过程</think>。
- 判断是否属于“咨询反馈问题解决进度”事件，答案请放置于<answer></answer>内部，例如：<answer>是/否</answer>。
Assistant:
<think>"""
    solution_str = """首先，我需要分析当前对话内容，判断是否属于“咨询反馈问题解决进度”这一分类。用户首先要求联系人工客服，客服确认订单后，用户表示没收到货，并多次要求打电话沟通，因为文字说不清楚。随后客服尝试联系骑手，用户拒绝并坚持电话沟通，最终客服同意回电。
根据事件解释，“咨询反馈问题解决进度”涉及用户询问之前反馈的问题的解决进展。例如用户提到之前的投诉或问题处理状态。但在这个对话中，用户的主要诉求是当前的配送问题，即未收到货物，并且希望通过电话解决沟通问题，而不是追踪之前的问题处理进度。
可能发生的场景中，如用户提到之前未解决的问题或客服提及之前的反馈，但这里用户是首次进线（或至少没有明确说明之前有过反馈），并且关注的是当前订单的即时配送问题，如“没收到货”，而不是历史问题。客服回应是尝试联系骑手处理当前问题，属于首次处理而非进度追踪。
注意事项中提到，首次进线不属于该分类（e项），且用户当前的问题属于首次反馈，没有涉及历史问题的追踪或催促。用户没有提到之前有反馈过相同问题，也没有涉及之前的处理进度更新，而是直接处理当前的问题。同时，用户坚持电话沟通，可能是因为当前问题沟通不顺畅，但并未提及之前的问题处理情况。
另外，用户的行为更多是催促当前问题的解决（如要求立即电话联系），但根据注意事项d项，用户对当前问题的催促不属于该分类。此外，没有迹象表明用户之前已经反馈过相同问题，现在再次进线咨询处理进展，而是首次进线处理未收到货的情况。
因此，对话符合注意事项中的排除条件，特别是首次进线（e项）和当前问题催促（d项），而事件解释和可能场景均未满足，因此不命中该分类。
</think>
<answer>否</answer></s>"""
    ground_truth = "是"
    score = compute_score(solution_str, ground_truth)
    print(score)