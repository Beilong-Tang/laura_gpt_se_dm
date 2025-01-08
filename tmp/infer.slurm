#!/bin/bash
#SBATCH -J inf_laura_se_tmp
#SBATCH -N 1
#SBATCH -o tmp/log_infer_tmp.out
#SBATCH -e tmp/log_infer_tmp.err
#SBATCH -p kshdnormal
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -o ... 'error in pipeline', -x 'print commands',
set -e
set -o pipefail

export MIOPEN_FIND_MODE=3
export HSA_FORCE_FINE_GRAIN_PRICE=1
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0

# export ROCBLAS_TENSILE_LIBPATH=/public/software/compiler/rocm/dtk-23.10/lib/rocblas/library_dcu2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate bltang_new

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/rocm/dtk-23.10

###########
# Setting #
###########

name="libri2mix"
config_name="config.yaml"
scp="/public/home/qinxy/bltang/laura_gpt/laura_gpt_se/tmp/tmp.scp"
num_proc=2
gpus="cuda:0 cuda:1"

output_dir=$(dirname $scp)/output
###############
# DONT CHANGE #
###############

config_path=exp/$name/$config_name
model_ckpt="$(dirname "$config_path")/$(basename "$config_path" .yaml)/ckpt/best.pth"

# Inference #

echo "[Inference tmp]"
python infer.py --scp $scp --config $config_path --model_ckpt $model_ckpt --output_dir "$output_dir/wavs" --num_proc $num_proc --gpus $gpus

