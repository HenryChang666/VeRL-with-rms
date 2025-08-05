#!/bin/bash
set -x
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cur_work_dir=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai/verl
cd $cur_work_dir
source $cur_work_dir/utils/export_env_verl.sh
export PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai/envs/myverl-3/bin:$PATH


# 默认值
NUM_EPISODES=20
LR_ACTOR=1e-6
n_samples_per_prompt=4
DATADIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai/data/rider_v4/train.parquet
MODELDIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/checkpoints/llama_factory/longcat_18b_kefu_base_sft_training_train_data_v2_4_lr-1e-5_epoch-5
REWARD_MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/program/llm/llm/QwQ-32B
WARM=no
experiment_name=zrl_grm_qwq
is_custome_model=False
VAL_DATADIR=""

THE_HOME=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai

DATAPATH=$DATADIR
VALDATAPATH=$DATADIR

PRETRAIN_DIR=$MODELDIR
MODEL_SAVE_NAME=${experiment_name}_episodes_${NUM_EPISODES}_actor_lr_${LR_ACTOR}_nsamples_${n_samples_per_prompt}
SAVE_DIR=$WORKDIR/results/checkpoint/rl/longcat_18b_V1_10_sanqing/$MODEL_SAVE_NAME

echo "save dir:" $SAVE_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ..
THE_HOME=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai
MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/checkpoints/llama_factory/longcat_18b_kefu_base_sft_training_train_data_v2_4_lr-1e-5_epoch-5
# MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/program/llm/llm/Qwen2.5-7B-Instruct
# REWARD_MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/program/llm/llm/Qwen2.5-32B-Instruct
REWARD_MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/program/llm/llm/QwQ-32B
TRAIN_FILE=$THE_HOME/data/rider_v4/train.parquet



if [ "$NODE_RANK" -eq 0 ]; then
   # 在容器中启动 Ray 的主节点
   # ray start --head --node-ip-address $MASTER_ADDR --port=6379 \
   # --num-gpus $GPUS_PER_NODE \
   # --temp-dir $WANDB_DIR\
   # --storage $RAY_STORAGE \
   #--include-dashboard=false
   ray start --head --node-ip-address $MASTER_ADDR --port=6379 --num-gpus $GPUS_PER_NODE
   # python3 -u $WORKDIR/utils/ray_start.py &
   sleep 30
   # 记录当前目录
   CUR_DIR=$(pwd)
   echo $CUR_DIR
   echo $PATH
   ray job submit --runtime-env-json="{\"working_dir\": \"$CUR_DIR\",
                           \"env_vars\": {\"PATH\": \"$PATH\"}, 
                           \"excludes\": [
                              \"afo-base-0.0.1-SNAPSHOT.jar\", 
                              \"jdk-11.0.12/lib/modules\", 
                              \"jdk-11.0.12/lib/src.zip\", 
                              \"jdk-11.0.12/lib/server/libjvm.so\", 
                              \"jdk-11.0.12/jmods/java.base.jmod\", 
                              \"jdk-11.0.12/jmods/java.desktop.jmod\",
                              \"data/*\",
                              \"results/*\",
                              \"hope.code.tar.gz\",
                              \"afo-dist-user-files.tar.gz\",
                              \"libbwfs76.so\"
                           ]}" \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TRAIN_FILE \
    data.train_batch_size=128 \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    reward_model.enable=True \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.micro_batch_size=4 \
    reward_model.reward_model_type=grm \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.model.input_tokenizer=$MODEL_PATH \
    reward_model.reward_vllm_rollout.gpu_memory_utilization=0.8 \
    reward_model.model.tensor_model_parallel_size=4 \
    reward_model.reward_vllm_rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.project_name='0620_grpo_rider_debug' \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=1000000 \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log
else
   echo "WORKER NODE"
   # python3 -u $WORKDIR/utils/ray_start.py &
   sleep 10
   # 在更多节点上启动 Ray
   RAY_START_CMD="ray start --block --address=${MASTER_ADDR}:6379 --num-gpus ${GPUS_PER_NODE}"
   # RAY_START_CMD="ray start --block --address=${MASTER_ADDR}:6379 \
   # --num-gpus ${GPUS_PER_NODE} \
   # --temp-dir $WANDB_DIR \
   # --storage $RAY_STORAGE"
   echo $RAY_START_CMD
   $RAY_START_CMD
fi