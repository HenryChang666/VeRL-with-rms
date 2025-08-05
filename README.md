<h1 style="text-align: center;">VeRL-with-RMs: 使用奖励模型更好地指导 Policy 训练 </h1>

🌟 关键特性：
支持多种奖励模型（RM）的强化学习训练。基于 vllm 后端实现高效的 RM 打分，同时通过动态装卸载 RM 及各组件，优化显存利用。支持 RM 与规则的混合打分模式。支持多 RM 实例协同打分模式。


## 使用方法（暂支持到GRM）：

1. 在标准的基于规则的 RL 训练流程所需要准备的组件基础上，准备一个 RM 或 off-the-shelf LLM 作为打分模型。
2. 参考 `verl/verl/trainer/config/ppo_trainer.yaml` 中 `reward_model` 项下定义的相关参数，在运行 `verl.trainer.main_ppo` 通过命令行参数传入以控制模型行为。比如 `reward_model.model.path` 控制读取的 RM 的路径，`reward_model.reward_model_type` 控制 RM 打分类型等等。
3. 数据处理格式要求：需要在模型读取的 parquet 文件中定义如下数据项，`prompt`（policy model 的 querys），`rm_prompt`（reward model 的 prompts，也可以是空str，在后续 `reward_function` 中再定义），`reward_model`（参考答案）。数据处理文件可参考 `verl/examples/data_preprocess/process_rider.py`。
4. Reward 前后处理：

   (1) 在 `verl/workers/reward_function` 文件夹下定义用于 Reward 计算前处理与后处理方法的 `.py` 文件，并且需要确保 config 的 `reward_model.reward_function` 的值与文件名相同从而使方法能够正确注册。比如，在 `rider_reward.py` 里定义了用于骑手 RL 训练的 reward 处理规则，就要在 config 或命令行传参中将 `reward_model.reward_function` 的值赋为 `rider_reward`。
   
   (2) 每个 `reward_function` 文件中都需要实现两个函数 `rm_preprocess()` 以及 `rm_postprocess()`，`rm_preprocess()` 负责制作 RM 的输入，需传入 `text_prompts`（RM 的prompt），`text_responses`（Policy 的rollout文本），`tgt_tokenizer`（RM 的 tokenizer），传出要输入给 RM 的文本。`rm_postprocess()` 需要对 RM 输出内容进行后处理提取出分数，传入 RM 输出的 `results`，传出提取后的分数。
5. 如果要做 rm 和规则的混合打分，`reward_model.combine_reward` 设为 True，然后在 `verl/verl/utils/reward_score` 中定义规则 reward 的计算方法（这块儿写的有点儿脏。。。）。
6. 然后就可以愉快地用 rm 辅助 rl 训练啦～

## Demo：
以电话骑手联系不到用户场景下知识与数据融合模型训练为例，policy model 使用 @智泉哥 提供的 sft 冷启动 longcat 18B 模型，reward model 使用 off-the-shelf qwq 32B 模型。训练数据为 @智泉哥 提供的业务数据，rm 打分规则为与 @景晴姐 拟定的三环节分维度打分模式。
数据处理脚本使用 `verl/examples/data_preprocess/process_rider.py`
reward_function 使用 `rider_reward`

### 单机多卡运行脚本：
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ..
THE_HOME=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai
MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/checkpoints/llama_factory/longcat_18b_kefu_base_sft_training_train_data_v2_4_lr-1e-5_epoch-5
# MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/program/llm/llm/Qwen2.5-7B-Instruct
# REWARD_MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/program/llm/llm/Qwen2.5-32B-Instruct
REWARD_MODEL_PATH1=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/checkpoints/llama_factory/longcat_18b_kefu_base_sft_training_train_data_v2_4_lr-1e-5_epoch-5
REWARD_MODEL_PATH2=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/checkpoints/llama_factory/longcat_18b_kefu_base_sft_training_train_data_v2_4_lr-1e-5_epoch-5
TRAIN_FILE=$THE_HOME/data/rider_v4/train_2.parquet
REWARD_FUCTION1=rider_reward
REWARD_FUCTION2=consistency_reward

python3 verl/utils/bge_sim.py &
echo "bge_sim.py started"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=reinforce_plus_plus \
 data.train_files=$TRAIN_FILE \
 data.val_files=$TRAIN_FILE \
 data.train_batch_size=32 \
 data.max_prompt_length=6144 \
 data.max_response_length=2048 \
 data.filter_overlong_prompts=True \
 reward_model.enable=True \
 reward_model.reward_function=[${REWARD_FUCTION1},${REWARD_FUCTION2}] \
 reward_model.model.path=[${REWARD_MODEL_PATH1},${REWARD_MODEL_PATH2}] \
 reward_model.micro_batch_size=4 \
 reward_model.reward_vllm_rollout.temperature=[0.7,0.7] \
 reward_model.reward_combine_func=['add','add'] \
 reward_model.reward_model_type=grm \
 reward_model.model.fsdp_config.param_offload=True \
 reward_model.model.input_tokenizer=$MODEL_PATH \
 reward_model.reward_vllm_rollout.gpu_memory_utilization=[0.8,0.8] \
 reward_model.reward_weight=[0.3,0.5] \
 reward_model.model.tensor_model_parallel_size=4 \
 reward_model.reward_vllm_rollout.tensor_model_parallel_size=4 \
 actor_rollout_ref.model.path=$MODEL_PATH \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.n=4 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.project_name='0705_grpo_rider_debug' \
 trainer.logger=['console'] \
 trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=30 \
 trainer.test_freq=1000000 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log

# actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
#  critic.optim.lr=1e-5 \
#  critic.model.path=$MODEL_PATH \
#  critic.ppo_micro_batch_size_per_gpu=4 \
#  reward_model.enable=True \
#  reward_model.model.path=$REWARD_MODEL_PATH \
#  reward_model.micro_batch_size=4 \
```

### 多机多卡运行脚本，参考自 @景晴姐 的脚本：
```bash
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
DATADIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai/data/rider_v4/train_2.parquet
MODELDIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/checkpoints/llama_factory/longcat_18b_kefu_base_sft_training_train_data_v2_4_lr-1e-5_epoch-5
# REWARD_MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/models/longcat_18b_V1_10_sanqing
WARM=no
experiment_name=zrl_grm_longcat_w03_add-add
is_custome_model=False
VAL_DATADIR=""

THE_HOME=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai

DATAPATH=$DATADIR
VALDATAPATH=$DATADIR

PRETRAIN_DIR=$MODELDIR
MODEL_SAVE_NAME=${experiment_name}_episodes_${NUM_EPISODES}_actor_lr_${LR_ACTOR}_nsamples_${n_samples_per_prompt}
SAVE_DIR=$WORKDIR/results/checkpoint/rl/longcat_18b_V1_10_sanqing/$MODEL_SAVE_NAME
# REWARD_FUCTION1=rider_reward

REWARD_MODEL_PATH1=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/checkpoints/llama_factory/longcat_18b_kefu_base_sft_training_train_data_v2_4_lr-1e-5_epoch-5
REWARD_MODEL_PATH2=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/checkpoints/llama_factory/longcat_18b_kefu_base_sft_training_train_data_v2_4_lr-1e-5_epoch-5
TRAIN_FILE=$THE_HOME/data/rider_v4/train_2.parquet
REWARD_FUCTION1=rider_reward
REWARD_FUCTION2=consistency_reward

python3 verl/utils/bge_sim.py &
echo "bge_sim.py started"

echo "save dir:" $SAVE_DIR

# export WANDB_DIR=${SAVE_DIR}/wandb
# mkdir -p $WANDB_DIR

# export RAY_STORAGE=$WANDB_DIR/ray_storage
# mkdir -p $RAY_STORAGE



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
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=$DATAPATH \
    data.val_files=$VALDATAPATH \
    data.train_batch_size=128 \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    reward_model.enable=True \
    reward_model.reward_function=[${REWARD_FUCTION1},${REWARD_FUCTION2}] \
    reward_model.model.path=[${REWARD_MODEL_PATH1},${REWARD_MODEL_PATH2}] \
    reward_model.micro_batch_size=4 \
    reward_model.reward_vllm_rollout.temperature=[0.7,0.7] \
    reward_model.reward_combine_func=['add','add'] \
    reward_model.reward_model_type=grm \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.model.input_tokenizer=$PRETRAIN_DIR \
    reward_model.reward_vllm_rollout.gpu_memory_utilization=[0.8,0.8] \
    reward_model.reward_weight=[0.3,0.5] \
    reward_model.model.tensor_model_parallel_size=4 \
    reward_model.reward_vllm_rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.model.path=$PRETRAIN_DIR\
    actor_rollout_ref.actor.optim.lr=$LR_ACTOR \
    actor_rollout_ref.actor.use_dynamic_bsz=True\
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$n_samples_per_prompt \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger=['console'] \
    trainer.project_name='RL-GRM-Longcat' \
    trainer.val_before_train=False \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=30 \
    trainer.test_freq=1000000 \
    trainer.total_epochs=$NUM_EPISODES
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

```

## TODO List：
1. GRM vllm 打分支持。☑
2. BT / critique + BT 的 vllm 打分支持。 □
3. RM sglang 后端支持。 □
4. 性能优化：模型卸载显存碎片  □
5. 性能优化：policy 和 reward tp_size 捆绑的性能限制 □
6. 多 rm 实例协同打分。☑
7. ...