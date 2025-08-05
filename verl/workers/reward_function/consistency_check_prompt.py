prompt_v1 = """# 任务描述
你是智能客服对话一致性评测专家。你的任务是：判断大模型A生成的客服回复，在两个层面是否保持一致性。
请严格按照以下要求执行：

# 一致性判定标准
1. **Thought与Response一致性**  
    - 核查Response是否忠实执行了Thought中的**核心意图**和**关键行动**（如：方案、补偿、安抚、核实等）。
    - 允许表达方式不同，但关键行动（如“外呼商家”“补款”）、处理方案需完全一致。
    - 不评估业务合理性，仅看是否按照Thought执行。
    - 典型不一致情形：Thought要求补款，Response未提及补款或拒绝补款。

2. **Response与ResponseSolution一致性**  
    - 检查Response是否**明确体现**ResponseSolution中指定的**核心行为**（如“外呼商家”“执行补款”），且行为对象（商家/用户/骑手）和目的（核实/补偿/确认）需完全对应。
    - ResponseSolution必须严格遵照**ResponseSolution解释**，不可捏造或模糊匹配。
    - 典型不一致情形：
        | **ResponseSolution**         | **一致Response示例**                          | **不一致Response示例**                     |
        |------------------------------|-----------------------------------------------|--------------------------------------------|
        | 外呼商家-骑手是否返餐         | “我会联系商家核实您是否返餐。”                | “您是否已将餐品交给商家？”（未外呼）       |
        | 外呼商家-骑手是否返餐         | “我打电话核实您是否返餐，您稍等。”            | “这边外呼商家核实返餐情况……您看可以吗？”（未实际外呼，仅询问骑手） |
        | 执行补款-7.2元               | “系统将补款7.2元，预计24小时内到账。”         | “我们会处理补款。”（未提金额）             |
        | 跟进返餐                     | “我会持续跟进您的返餐进度，请稍候。”          | “请等待返餐结果。”（未承诺跟进）           |

# 输入数据
Thought: {Thought}
Response: {Response}
ResponseSolution: {ResponseSolution}
ResponseSolution解释: {ResponseSolutionName}

# 输出要求
1. **思考过程**：请分步骤分析：
    - 第一步：提取Thought中的核心意图和关键行动。
    - 第二步：检查Response是否执行了这些行动（允许表述差异）。
    - 第三步：对比Response与ResponseSolution的核心行为、对象、目的是否完全匹配。
    - 将分析过程放在<think>...</think>标签内。
2. **最终输出**：仅输出以下JSON格式，无其他内容：
    {"Thought与Response一致性": 1/0, "Response与ResponseSolution一致性": 1/0}
    - 1表示一致，0表示不一致。
    - 如果 ResponseSolution 为"无"，则Response与ResponseSolution一致性直接输出1分。

# 严格输出规范
- 只输出<think>...</think>和最终JSON，无其他内容。
- JSON中不输出解释、多余标点或空行。

**输出示例**：  
<think>
1. Thought核心意图：外呼商家核实返餐情况。  
   Response未提及外呼商家，仅询问骑手，故不一致。  
2. ResponseSolution要求“外呼商家”，但Response未执行该行为，故不一致。  
</think>
{"Thought与Response一致性": 0, "Response与ResponseSolution一致性": 0}

请开始判断：
"""