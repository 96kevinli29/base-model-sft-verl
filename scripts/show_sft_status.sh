#!/bin/bash
# 展示当前 SFT 训练任务：任务名、节点数、GPU 数、状态（便于截图）
# 用法：bash scripts/show_sft_status.sh [JOBID]
# 不传 JOBID 则显示本用户正在运行的第一个 sft-qwen 相关作业

JOBID="${1:-}"
if [[ -z "$JOBID" ]]; then
  JOBID=$(squeue -u "$USER" -n sft-qwen -h -o "%i" | head -1)
  if [[ -z "$JOBID" ]]; then
    echo "未找到运行中的 sft-qwen 作业。请指定 JOBID: bash $0 <JOBID>"
    exit 0
  fi
fi

# 取各字段（分行输出便于截图）
NODES=$(squeue -j "$JOBID" -h -o "%D")
NAME=$(squeue -j "$JOBID" -h -o "%j")
STATE=$(squeue -j "$JOBID" -h -o "%T")
ELAPSED=$(squeue -j "$JOBID" -h -o "%M")
TLIMIT=$(squeue -j "$JOBID" -h -o "%l")
NODELIST=$(squeue -j "$JOBID" -h -o "%N")
GPUS=$((NODES * 4))

echo "=============================================="
echo "  SFT 训练任务状态"
echo "=============================================="
echo ""
echo "  JOBID:      $JOBID"
echo "  任务名:     $NAME"
echo "  状态:       $STATE"
echo "  节点数:     $NODES"
echo "  GPU 总数:   $NODES × 4 = $GPUS 卡"
echo "  GPU 型号:   NVIDIA A100-SXM4-40GB"
echo "  已运行:     $ELAPSED"
echo "  时间限制:   $TLIMIT"
echo "  节点列表:   $NODELIST"
echo ""
echo "=============================================="
