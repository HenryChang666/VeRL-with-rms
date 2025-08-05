#!/bin/bash

# 设置一些默认路径（可根据需要调整）
DEFAULT_LOCAL_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai/verl/results/checkpoint/rl/longcat_18b_V1_10_sanqing/zrl_grm_longcat_w03_add-sanqingrm_episodes_20_actor_lr_1e-6_nsamples_4"
DEFAULT_HF_MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/checkpoints/llama_factory/longcat_18b_kefu_base_sft_training_train_data_v2_4_lr-1e-5_epoch-5"
DEFAULT_TARGET_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai/verl/results/checkpoint/rl/longcat_18b_V1_10_sanqing/zrl_grm_longcat_w03_add-sanqingrm_episodes_20_actor_lr_1e-6_nsamples_4"

# 你的模型列表：模型名称及其对应的本地目录、源模型目录和输出目录
declare -A models
models=(
    ["model1"]="$DEFAULT_LOCAL_DIR/global_step_120/actor,$DEFAULT_HF_MODEL_PATH,$DEFAULT_TARGET_DIR/hf_global_step_120"
    ["model2"]="$DEFAULT_LOCAL_DIR/global_step_240/actor,$DEFAULT_HF_MODEL_PATH,$DEFAULT_TARGET_DIR/hf_global_step_240"
    ["model3"]="$DEFAULT_LOCAL_DIR/global_step_360/actor,$DEFAULT_HF_MODEL_PATH,$DEFAULT_TARGET_DIR/hf_global_step_360"
)

# 遍历模型，执行转换
for model_name in "${!models[@]}"; do
    # 获取每个模型的本地目录、源模型路径和输出目录
    IFS=',' read -r local_dir hf_model_path target_dir <<< "${models[$model_name]}"

    echo "Converting model: $model_name"
    echo "Local Directory: $local_dir"
    echo "HF Model Path: $hf_model_path"
    echo "Target Directory: $target_dir"

    # 调用 Python 脚本进行 FSDP 模型的合并与转换
    python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai/verl/scripts/model_merger.py merge \
        --backend fsdp \
        --local_dir "$local_dir" \
        --hf_model_path "$hf_model_path" \
        --target_dir "$target_dir"

    if [ $? -eq 0 ]; then
        echo "$model_name conversion successful."
    else
        echo "$model_name conversion failed."
    fi
done

echo "All models have been processed."
