import argparse
import os

template = """[base]
type = ml-easy-job

[resource]
usergroup = hadoop-mtai
queue = root.zw05_training_cluster.hadoop-llm.pool

[roles]
workers = {worker_num}
worker.memory = {memory_num}
worker.vcore = 96
worker.gcores80g = {gpu_num}
# worker.script = pip3 install hope-track --upgrade -i http://pypi.sankuai.com/simple/ --trusted-host pypi.sankuai.com --force-reinstall --no-cache && pip3 install safetensors==0.4.3 && pip3 install pulp && pip3 install transformers==4.51.3 && {script_path}
worker.script = {script_path}

[user_args]

[am]
afo.app.am.resource.mb = 4096

[tensorboard]
with.tensor.board = false

[docker]
afo.docker.image.name = registryonline-hulk.sankuai.com/custom_prod/com.sankuai.phxmlp.mtjupyter.singleuser/hdp_base_pytorch2.3.1_cuda12.4.0_python3.9_fa2.5.8_ncclmt2.20.5_ompi4_gcc11_ofed5.8_0654e424

[data]
afo.data.prefetch = false

[failover]
afo.app.support.engine.failover = true

[conda]

[config]

[others]
afo.app.env.YARN_CONTAINER_RUNTIME_DOCKER_SHM_SIZE_BYTES = 107374182400
afo.xm.notice.receivers.account = {mis_id}
with_requirements = false
afo.role.worker.task.attempt.max.retry = 3
afo.role.worker.not.node_name=zw04-data-k8s-gpu-a100-node1030.mt,zw04-data-k8s-gpu-a100-node1023.mt,zw04-data-k8s-gpu-a100-node1071.mt
afo.role.worker.env.LAMP_PYTORCH_COMMUNICATION_TIMEOUT = 36000
afo.role.worker.env.NCCL_SOCKET_IFNAME = ^lo,docker0
afo.role.worker.env.NCCL_IB_DISABLE = 0
afo.role.worker.env.NCCL_NET_GDR_LEVEL = 2
afo.network.mode = RDMA
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--gpu_num", type=int, required=True)
    parser.add_argument("--mis_id", type=str, required=True)
    parser.add_argument("--script_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    
    memory_num = 800000 if args.gpu_num >= 4 else 400000
    hope_config = template.format(worker_num=args.worker, gpu_num=args.gpu_num, script_path=args.script_path, mis_id=args.mis_id, memory_num=memory_num)
    
    with open(args.save_path, "w") as f:
        f.write(hope_config)
    