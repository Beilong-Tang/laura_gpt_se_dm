#!/bin/bash
#SBATCH -J inf_laura_se_libri2mix
#SBATCH -N 1
#SBATCH -o log_infer.out
#SBATCH -e log_infer.err
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
stage=2
stop_stage=2 # 0: infer, 1: DNSMOS + SPKSIM 2: WER 

# 0. libri2mix mixture without norm
name="libri2mix"
config_name="config.yaml"
scp="/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/Libri2Mix/wav16k/min/list/test/mix.scp"
libri2mix_clean_dir="/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/Libri2Mix/wav16k/min/test/s1" # for wespeaker only

# DDP #
num_proc=8
gpus="cuda:0 cuda:1 cuda:2 cuda:3"

########
# Eval #
########
dns_model_dir="/public/home/qinxy/bltang/ml_framework_slurm/recipes/DNSMOS"
wer_model="base"
wer_num_proc=8

# DONT CHANGE #
wer_reference="/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/Libri2Mix/wav16k/min/whisper/whisper_$wer_model.txt"

###############
# DONT CHANGE #
###############
config_path=exp/$name/$config_name
output_dir="output/$name/$(basename $config_name .yaml)"
model_ckpt="$(dirname "$config_path")/$(basename "$config_path" .yaml)/ckpt/best.pth"

#######
# Run #
#######

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
echo "[Inference]"
python infer.py --scp $scp --config $config_path --model_ckpt $model_ckpt --output_dir "$output_dir/wavs" --num_proc $num_proc --gpus $gpus
fi


########
# Eval #
########


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "[Evaluation]"
  # DNSMOS
  echo "[DNSMOS 16k]"
  python eval/dnsmos.py --model_dir $dns_model_dir -t "$output_dir/wavs" -o "$output_dir/dnsmos.csv"

  # SPK SIM
  echo "[SPKSIM WeSpeaker]"
  python eval/wespeaker_eval.py -t "$output_dir/wavs" \
    -r $libri2mix_clean_dir -o "$output_dir/wespeaker.csv" 
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ## WER 
  echo "[WER $wer_model]"
  ## TODO: Change WER TO Large After downloading
  python eval/wer.py -t "$output_dir"/wavs -r $wer_reference \
    -o "$output"/transcript_"$wer_model".txt -m $wer_model --num_proc $wer_num_proc
fi 
