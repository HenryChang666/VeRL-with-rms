export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ..
THE_HOME=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhangrunlai
MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/checkpoints/llama_factory/longcat_18b_kefu_base_sft_training_train_data_v2_4_lr-1e-5_epoch-5
# MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/program/llm/llm/Qwen2.5-7B-Instruct
# REWARD_MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/program/llm/llm/Qwen2.5-32B-Instruct
REWARD_MODEL_PATH1=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/models/longcat_18b_V1_10_sanqing
REWARD_MODEL_PATH2=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/caozhiquan/models/longcat_18b_V1_10_sanqing
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
 reward_model.use_bge=True \
 reward_model.bge_weight=0.5 \
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