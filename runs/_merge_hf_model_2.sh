#!/bin/bash

DEFAULT_LOCAL_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai/verl/results/checkpoint/rl/longcat_18b_V1_10_sanqing/0730_longcat_add-trained-grm03_add-bge05_episodes_25_actor_lr_1e-6_nsamples_4"
DEFAULT_HF_MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/checkpoints/llama_factory/longcat_18b_kefu_base_sft_training_train_data_v2_4_lr-1e-5_epoch-5"
DEFAULT_TARGET_DIR=${DEFAULT_LOCAL_DIR}

model_list=(
    "model1,$DEFAULT_LOCAL_DIR/global_step_120/actor,$DEFAULT_HF_MODEL_PATH,$DEFAULT_TARGET_DIR/hf_global_step_120"
    "model2,$DEFAULT_LOCAL_DIR/global_step_300/actor,$DEFAULT_HF_MODEL_PATH,$DEFAULT_TARGET_DIR/hf_global_step_300"
    "model3,$DEFAULT_LOCAL_DIR/global_step_600/actor,$DEFAULT_HF_MODEL_PATH,$DEFAULT_TARGET_DIR/hf_global_step_600"
    "model4,$DEFAULT_LOCAL_DIR/global_step_900/actor,$DEFAULT_HF_MODEL_PATH,$DEFAULT_TARGET_DIR/hf_global_step_900"
)

for model_info in "${model_list[@]}"; do
    IFS=',' read -r model_name local_dir hf_model_path target_dir <<< "$model_info"

    echo "Converting model: $model_name"
    echo "Local Directory: $local_dir"
    echo "HF Model Path: $hf_model_path"
    echo "Target Directory: $target_dir"

    python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai/verl/scripts/model_merger.py merge \
        --backend fsdp \
        --local_dir "$local_dir" \
        --hf_model_path "$hf_model_path" \
        --target_dir "$target_dir"

    if [ $? -eq 0 ]; then
        mkdir -p "$target_dir"
        cp "$DEFAULT_HF_MODEL_PATH/configuration_llama.py" "$DEFAULT_HF_MODEL_PATH/modeling_llama.py" "$target_dir"
        echo "$model_name conversion successful."
    else
        echo "$model_name conversion failed."
    fi
done

echo "All models have been processed."
