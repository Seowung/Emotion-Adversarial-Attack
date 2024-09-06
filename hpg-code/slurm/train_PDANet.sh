#!/bin/bash
#
# Script to launch a multi-gpu distributed training using MONAI Core
# on UF HiperGator's AI partition, a SLURM cluster using Singularity
# as container runtime.
#
# This script uses `pt_multinode_helper_funcs.sh`, and
# either `run_on_node.sh`(for single-node multi-gpu training)
# or `run_on_multinode.sh` (for multi-node multi-gpu training). All
# the three `.sh` files are in \monaicore_multigpu\util_multigpu.
#
# We use torch.distributed.launch to launch the training, so please
# set as follows:
#   set #SBATCH --ntasks=--nodes
#   set #SBATCH --ntasks-per-node=1
#   set #SBATCH --gpus=total number of processes to run on all nodes
#   set #SBATCH --gpus-per-task=--gpus/--ntasks
#
#   for multi-node training, replace `run_on_node.sh` in
#   `PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_on_node.sh")`
#   with `run_on_multinode.sh`.
#
#   Modify paths to your own paths.
#
# (c) 2021, Brian J. Stucky, UF Research Computing
# 2022, modified by Huiwen Ju, hju@nvidia.com

# Resource allocation.

#################################################
#################################################
#modified Yunchao Yang
#SBATCH --job-name=train_PDANet  # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8            ###
#SBATCH --gpus-per-task=8    ###
#SBATCH --cpus-per-task=8  ###
#SBATCH --mem=128gb          ###
##SBATCH --account=ruogu.fang
##SBATCH --qos=ruogu.fang
#SBATCH --time=72:00:00           ###########
##SBATCH --mail-type=End,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=leem.s@ufl.edu     # Where to send mail
##################################################
#################################################
##SBATCH --reservation=el8-hpgai
##SBATCH --exclude=c0900a-s23,c0900a-s35,c0910a-s17,c1106a-s11,c0901a-s11,c1100a-s11
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-ai
#SBATCH --output=%x_%j.log   # Standard output and error log
#SBATCH --wait-all-nodes=1

#cd /red/ruogu.fang/leem.s/NSF-SCH/code
module load conda; conda activate /red/ruogu.fang/leem.s/conda/envs/monai-0.9.1
#cd /red/ruogu.fang/leem.s/Emotion-Adversarial-Attack/
#conda activate /blue/ruogu.fang/leem.s/conda/envs/AD
#export PATH=/blue/ruogu.fang/leem.s/conda/envs/AD/bin:$PATH

export NCCL_DEBUG=INFO
# can be set to either OFF (default), INFO, or DETAIL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
# Training command specification: training_script -args.
TRAINING_SCRIPT="$(realpath "../train_multi.py")"
TRAINING_CMD="$TRAINING_SCRIPT --model PDANet"

# Python location (if not provided, system default will be used).
# Here we run within a MONAI Core Singularity container,
# see `build_container.sh` to build a MONAI Core Singularity container.
#PYTHON_PATH="/blue/ruogu.fang/leem.s/conda/envs/NSF-SCH/bin/python3"

# Location of the PyTorch launch utilities,
# i.e. `pt_multinode_helper_funcs.sh`, `run_on_node.sh` and `run_on_multinode`.
# PT_LAUNCH_UTILS_PATH=/red/ruogu.fang/leem.s/NSF-SCH/code/util_multigpu
PT_LAUNCH_UTILS_PATH=../util_multigpu
source "${PT_LAUNCH_UTILS_PATH}/pt_multinode_helper_funcs.sh"

init_node_info

pwd; hostname; date

echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"
echo "Secondary nodes: $SECONDARIES"

#PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_on_node.sh")
PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_on_multinode.sh")
echo "Running \"$TRAINING_CMD\" on each node..."

srun --unbuffered "$PT_LAUNCH_SCRIPT" "$(realpath $PT_LAUNCH_UTILS_PATH)" \
    "$TRAINING_CMD" "$PYTHON_PATH"
