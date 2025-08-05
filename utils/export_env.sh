#!/bin/bash
export PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/env/logic/bin:$PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/env/logic/lib/python3.9/site-packages/nvidia/cublas/lib"

export WORKDIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/program/llm/voc/verl_v1_new"
export VLLM_ATTENTION_BACKEND=XFORMERS

# export PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/env/v1/bin:$PATH"
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/env/v1/lib/python3.10/site-packages/nvidia/cublas/lib"
# export WORKDIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/ruanjingqing/program/llm/voc/verl_v1_new"
# # V1 和 Xformers 只能二选一
# # export VLLM_ATTENTION_BACKEND=XFORMERS
# export VLLM_USE_V1=1


# 读取GPU环境
export NNODES=`python ${WORKDIR}/utils/parse_environment.py nnodes`
export MASTER_ADDR=`python ${WORKDIR}/utils/parse_environment.py master_addr`
export MASTER_PORT=`python ${WORKDIR}/utils/parse_environment.py master_port`
export GPUS_PER_NODE=`python ${WORKDIR}/utils/parse_environment.py nproc_per_node`
export NODE_RANK=`python ${WORKDIR}/utils/parse_environment.py node_rank`
export WORLD_SIZE=$((NNODES*GPUS_PER_NODE))
export WANDB_MODE=offline
# export WANDB_DIR=${WORKDIR}/wandb

export HYDRA_FULL_ERROR=1
export RAY_memory_monitor_refresh_ms=0

echo $NNODES
echo $MASTER_ADDR
echo $MASTER_PORT
echo $GPUS_PER_NODE
echo $NODE_RANK
echo $WORLD_SIZE
echo $WORKDIR
echo $PATH