#!/bin/bash
#SBATCH -J inf_laura_se_libri2mix
#SBATCH -N 1
#SBATCH -o log_infer.out
#SBATCH -e log_infer.err
#SBATCH -p kshdnormal01
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4

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
scp="/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/Libri2Mix/wav16k/min/list/test/mix.scp"
# DDP #
num_proc=8
gpus="cuda:0 cuda:1 cuda:2 cuda:3"

###############
# DONT CHANGE #
###############
config_path=exp/$name/$config_name
output_dir="output/$name/$(basename $config_name .yaml)"
model_ckpt="$(dirname "$config_path")/$(basename "$config_path" .yaml)/best.pth"

#######
# Run #
#######
python infer.py --scp $scp --config $config_path --model_ckpt $model_ckpt --output_dir $output_dir --num_proc $num_proc --gpus $gpus
