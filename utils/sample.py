import argparse
from functools import partial
import os
import numpy as np
from vllm import LLM, SamplingParams
from parallel_kit import model_map
from jsonkit import read_jsonl, write_jsonl
import json
from tqdm import tqdm

SYSTEM_PROMPT_INSTR = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. """
SYSTEM_PROMPT_DISTILL = ""

# longcat_template = """<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>
# """

longcat_template = """User: \n{prompt}\nAssistant: \n<think>"""

def apply_chat_def(tokenizer, input_text):
    if tokenizer.special_tokens_map['eos_token'] == '<|im_end|>': # instruct model
        # SYSTEM_PROMPT = SYSTEM_PROMPT_INSTR
        SYSTEM_PROMPT = SYSTEM_PROMPT_DISTILL
        postfix = '<think>'
    elif tokenizer.special_tokens_map['bos_token'] == '<｜begin▁of▁sentence｜>': # r1 distill
        SYSTEM_PROMPT = SYSTEM_PROMPT_DISTILL
        postfix = '<think>'
    elif tokenizer.special_tokens_map['bos_token'] == '<s>': # longcat
        return longcat_template.format(prompt=input_text)
        
    input_item = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
    ]
    input_text = tokenizer.apply_chat_template(input_item, tokenize=False, add_generation_prompt=True)
    # input_text = input_text + postfix
    
    return input_text



def generate_worker(cuda_device, prompts, model_path, n, temperature, max_tokens, top_p, top_k, output_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_device)
    
    # 初始化进度条（每个worker独立）
    pbar = tqdm(
        total=len(prompts),
        desc=f"GPU {cuda_device}",
        position=int(cuda_device[0]),  # 根据GPU序号分配显示位置
        leave=False
    )

    llm = LLM(
        model=model_path,
        seed=42,
        max_model_len=max(10 * 1024, max_tokens),
        swap_space=16,
        tensor_parallel_size=len(cuda_device),
    )

    tokenizer = llm.get_tokenizer()
    stop_token_ids = [tokenizer.eos_token_id]
    print(f"SUCCESS: load llm {model_path} on cuda {cuda_device}")
    print("Temperature is", temperature)

    vllm_sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )

    test_str = "there is a apply_chat_template test"
    print(apply_chat_def(tokenizer, test_str))
    
    # 准备输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 分批处理prompts
    for batch in chunked(prompts, 100):  # 每100个为一组处理
        # 生成文本
        text_prompts = [apply_chat_def(tokenizer, item["input"]) for item in batch]
        outputs = llm.generate(text_prompts, sampling_params=vllm_sampling_params)

        # 逐条写入结果
        with open(output_path, 'a', encoding='utf-8') as f:
            for item, output in zip(batch, outputs):
                result = {
                    **item,
                    "generated": [i.text for i in output.outputs]
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()  # 确保及时写入
        
        pbar.update(len(batch))
    pbar.close()

    return len(prompts)  # 返回处理数量用于统计
    
    
    # text_prompts = [apply_chat_def(tokenizer, item["input"]) for item in prompts]

    # outputs = llm.generate(
    #     text_prompts, sampling_params=vllm_sampling_params, use_tqdm=True
    # )

    # results = []
    # for item, output in zip(prompts, outputs):
    #     result = {**item, "generated": [i.text for i in output.outputs]}
    #     results.append(result)
    # return results


def chunked(lst, size):
    """将列表分块"""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, help="prompts path")
    parser.add_argument("--out", type=str, help="output path")
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--n_sample", type=int, default=10, help="number of samples per task")
    parser.add_argument("--temperature", type=float, default=0.6, help="sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="sampling temperature")
    parser.add_argument("--top_k", type=float, default=20, help="sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max number of tokens to generate")
    parser.add_argument("--gpu_per_model", type=int, default=1, help="Number of GPUs required per model")
    args = parser.parse_args()

    prompts = list(read_jsonl(args.prompts))
    worker = partial(
        generate_worker,
        model_path=args.model,
        n=int(args.n_sample),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        output_path=args.out  # 添加输出路径参数
    ) # 
    # temperature=0.6 # qwen3
    # top_p=0.95
    # top_k=20
    # results = model_map(worker, prompts, args.gpu_per_model)
    # write_jsonl(results, args.out)
    # 执行并行处理（自动写入文件）
    total = model_map(worker, prompts, args.gpu_per_model)
    print(f"Total processed: {sum(total)} prompts")
