#!/bin/bash
#SBATCH --job-name=all_reduce_bench
#SBATCH --nodes=121
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -x -e

source /lustre/scwpod02/client/huggingface/start_bench
conda activate p5-bench
CUDA_VERSION=12.0
#NCCL_TEST_PATH=/usr/local/cuda-${CUDA_VERSION}/efa/test-cuda-${CUDA_VERSION}
# do not remove or the training will hang and nodes will be lost w/o this workaround
# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

echo "START TIME: $(date)"
printenv
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    "

CMD="\
all_reduce_bench.py \
"

echo $CMD

conda activate p5-bench

export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_SL=0
export NCCL_IB_TC=41
export NCCL_IB_QPS_PER_CONNECTION=1
export NCCL_ALGO=Ring
export NCCL_IB_HCA=^=mlx5_2,mlx5_8,mlx5_1,mlx5_7
export NCCL_SOCKET_IFNAME=ibp
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --container-image=gsalouovh/allreducebench:latest \
    --container-mounts /dev/infiniband/:/dev/infiniband/ \
    "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD" 2>&1 | tee $LOG_PATH

echo "END TIME: $(date)"