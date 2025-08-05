def grm_preprocess_test(text_prompts, text_responses, tgt_tokenizer):
    def mk_vllm_input(text_prompt, text_response):
        prompt = f"""Please determine whether the following "Answer" correctly answers the "Question". If correct, output only the number 1; if incorrect, output only the number 0. Do not output anything else.
Question: {text_prompt}
Answer: {text_response}"""
        message = [{"role": "user", "content": prompt}]
        return tgt_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    mk_vllm_inputs = []
    for text_prompt, text_response in zip(text_prompts, text_responses):
        mk_vllm_inputs.append(mk_vllm_input(text_prompt, text_response))
    return mk_vllm_inputs

def grm_postprocess_test(results):
    def mk_reward_output(result):
        if '0' in result:
            return 0
        elif '1' in result:
            return 1
        else:
            return 0

    rewards = []
    for result in results:
        rewards.append(mk_reward_output(result))
    return rewards