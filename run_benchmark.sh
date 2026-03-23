#!/bin/bash -l
#===============================================================================
# 6-Dataset Benchmark 评测
#
# 支持的模型（通过 MODEL_PATH 指定，未设则默认 Qwen3-4B-Base）:
#   Qwen3-4B-Base  基座
#   Qwen3-4B-SFT   SFT 模型
#   Qwen3-4B-SFT-200steps  SFT 200 步 checkpoint
#   Qwen3-4B-RL    RL 后模型
#   Qwen3-4B       通用名（如合并后的单模型）
#
# 用法:
#   默认 Slurm 日志（绝对路径，与提交时当前目录无关）:
#     /project/home/p201251/hyl_ppo/logs/bench_<jobid>.out
#     /project/home/p201251/hyl_ppo/logs/bench_<jobid>.err
#   若无法创建 out/err，先执行: mkdir -p /project/home/p201251/hyl_ppo/logs
#   也可用 sbatch -o/-e 覆盖默认路径。
#
#   # 快速测试（每个数据集 5 个 cases，结果输出到 wandb，可查看 prompt/response/score/ground_truth）
#   sbatch run_benchmark.sh test
#
#   # 指定模型（SFT 在项目根目录，非 outputs 下）:
#   MODEL_PATH=${WORK_DIR}/Qwen3-4B-SFT sbatch run_benchmark.sh test
#   MODEL_PATH=${WORK_DIR}/Qwen3-4B-SFT-200steps sbatch ... run_benchmark.sh test
#
#   # 完整评测（6617 条 × 3 轮）
#   sbatch run_benchmark.sh full
#
# 环境变量（可选）:
#   MODEL_PATH=...        模型目录（默认 ${WORK_DIR}/Qwen3-4B-Base）
#   N_SAMPLES=1           每 prompt 采样数（默认 16，快速测试可设 1）
#   MAX_NEW_TOKENS=4096   最大生成 token 数（默认 4096，AIME 可试 8192）
#   MAX_PER_SET=20        test 时每数据集条数（默认 5，full 时忽略）
#   ENABLE_THINKING=0     关闭思考模式（默认已开启 thinking）
#   NO_CHAT_TEMPLATE=1    不使用 chat template，用原始 prompt
#   EXCLUDE_DATASETS=gsm8k,math_lighteval  跳过指定数据集（逗号分隔）
#
# AIME 准确率对比（SFT 后仍低时可分别跑三种配置对比）:
#   # 1) 默认（thinking + 脚本内 MAX_NEW_TOKENS）
#   MODEL_PATH=${WORK_DIR}/Qwen3-4B-SFT sbatch run_benchmark.sh full
#   # 2) 关闭 thinking（与默认对比）
#   ENABLE_THINKING=0 MODEL_PATH=${WORK_DIR}/Qwen3-4B-SFT sbatch run_benchmark.sh full
#   # 3) 长 token（AIME 题解可能较长）
#   MAX_NEW_TOKENS=8192 MODEL_PATH=${WORK_DIR}/Qwen3-4B-SFT sbatch run_benchmark.sh full
#   区分方式：WandB 里 run 的 config（enable_thinking、max_new_tokens）已记录，按 setting 对比 aime2024、aime2025 即可。
#===============================================================================

#SBATCH --job-name=bench-qwen3
#SBATCH --account=p201251
#SBATCH --partition=gpu
#SBATCH --qos=default
# 预估时间：test 约 0.5–1h，full 约 4–6h。设为 6h 保证 full 能跑完（test 会提前结束）
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=4
# 绝对路径：避免从其它目录 sbatch 时 out/err 写到错误位置
#SBATCH --chdir=/project/home/p201251/hyl_ppo
#SBATCH --output=/project/home/p201251/hyl_ppo/logs/bench_%j.out
#SBATCH --error=/project/home/p201251/hyl_ppo/logs/bench_%j.err

set -euxo pipefail

# ============ 参数 ============
RUN_MODE=${1:-test}
ENABLE_THINKING="${ENABLE_THINKING:-1}"
NO_CHAT_TEMPLATE="${NO_CHAT_TEMPLATE:-0}"
EXCLUDE_DATASETS="${EXCLUDE_DATASETS:-}"

WORK_DIR=/project/home/p201251/hyl_ppo
# 默认 Base；支持 Qwen3-4B-SFT、Qwen3-4B-SFT-200steps、Qwen3-4B-RL、Qwen3-4B 等
MODEL_PATH="${MODEL_PATH:-${WORK_DIR}/Qwen3-4B-Base}"

cd "${WORK_DIR}"
mkdir -p "${WORK_DIR}/logs"

source activate_verl.sh

# FlashInfer sampling 需要 nvcc JIT 编译；当前集群无 nvcc，禁用以回退到 PyTorch 采样
export VLLM_USE_FLASHINFER_SAMPLER=0

# ============ 配置: test vs full ============
# 从 MODEL_PATH 取目录名作为模型名，WandB/输出目录与之一致（如 Qwen3-4B-SFT-200steps）
MODEL_NAME=$(basename "${MODEL_PATH}")
N_SAMPLES="${N_SAMPLES:-16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TP_SIZE=4
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
# max_model_len 不设置；整条序列 = prompt + 生成，prompt 随样本变化，由 vLLM/模型默认
GPU_MEM_UTIL=0.90

if [[ "$RUN_MODE" == "full" ]]; then
    MAX_PER_SET=-1
    WANDB_RUN_NAME="${MODEL_NAME}-full-n${N_SAMPLES}"
    OUTPUT_DIR="${WORK_DIR}/outputs/benchmark/${MODEL_NAME}/full"
else
    # 每个数据集取 5 个 cases，便于在 wandb 中查看回溯：prompt / response / score / ground_truth
    MAX_PER_SET="${MAX_PER_SET:-5}"
    WANDB_RUN_NAME="${MODEL_NAME}-test-n${N_SAMPLES}"
    OUTPUT_DIR="${WORK_DIR}/outputs/benchmark/${MODEL_NAME}/test"
fi

# ============ 打印 ============
echo "=============================================="
echo "6-Dataset Benchmark [${RUN_MODE}]"
echo "=============================================="
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "slurm_job_id:     ${SLURM_JOB_ID}"
  echo "slurm_stdout:     ${WORK_DIR}/logs/bench_${SLURM_JOB_ID}.out"
  echo "slurm_stderr:     ${WORK_DIR}/logs/bench_${SLURM_JOB_ID}.err"
  echo "=============================================="
fi
echo "model_path:       ${MODEL_PATH}"
echo "n_samples:        ${N_SAMPLES}"
echo "max_new_tokens:   ${MAX_NEW_TOKENS}"
echo "temperature:      ${TEMPERATURE}"
echo "enable_thinking:  ${ENABLE_THINKING}"
echo "no_chat_template: ${NO_CHAT_TEMPLATE}"
echo "exclude_datasets: ${EXCLUDE_DATASETS:-none}"
echo "tp_size:          ${TP_SIZE}"
echo "max_per_set:      ${MAX_PER_SET}"
echo "output_dir:       ${OUTPUT_DIR}"
echo "started_at:       $(date -Iseconds)"
echo "=============================================="

# ============ 运行 ============
NUM_SCHEDULER_STEPS=8
EXTRA_ARGS=()
[[ "$ENABLE_THINKING" == "1" ]] && EXTRA_ARGS+=(--enable_thinking)
[[ "$NO_CHAT_TEMPLATE" == "1" ]] && EXTRA_ARGS+=(--no_chat_template)
[[ "$RUN_MODE" == "test" ]] && EXTRA_ARGS+=(--enforce_eager)
[[ -n "$EXCLUDE_DATASETS" ]] && EXTRA_ARGS+=(--exclude_datasets "$EXCLUDE_DATASETS")

python scripts/run_benchmark.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${WORK_DIR}/my_data" \
    --output_dir "${OUTPUT_DIR}" \
    --n_samples ${N_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --tp_size ${TP_SIZE} \
    --gpu_memory_util ${GPU_MEM_UTIL} \
    --max_per_set ${MAX_PER_SET} \
    --num_scheduler_steps ${NUM_SCHEDULER_STEPS} \
    --wandb_project "LLM-Benchmark" \
    --wandb_name "${WANDB_RUN_NAME}" \
    "${EXTRA_ARGS[@]}"

echo "=============================================="
echo "Benchmark finished: $(date -Iseconds)"
echo "=============================================="
