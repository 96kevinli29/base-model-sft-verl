#!/bin/bash
#SBATCH -J merge-fsdp
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o merge-%j.out
#SBATCH -e merge-%j.err
set -euo pipefail
cd /project/home/p201251/hyl_ppo
conda activate verl
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /project/home/p201251/hyl_ppo/outputs/sft_qwen3_8b_sft_50k_apex-run1/global_step_2000 \
  --target_dir /project/home/p201251/hyl_ppo/Qwen3-8B-SFT3-2000steps \
  --trust-remote-code
