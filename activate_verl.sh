# 每次新开终端后执行: source activate_verl.sh  或  . activate_verl.sh
# 然后就可以用 python、pip 了（都是 verl 环境里的）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="$SCRIPT_DIR/tools/miniconda3"

# 先 source conda.sh，再 activate；不要先把 base/bin 塞进 PATH，否则激活后 python 会误用 base
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
fi
conda activate verl
# 不再跑 which/python --version，加快激活；需要时自己执行 which python

# WandB：从文件读取 API Key（不在脚本中明文存储）
WANDB_KEY_FILE="$SCRIPT_DIR/.wandb_key"
if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY=$(tr -d '[:space:]' < "$WANDB_KEY_FILE")
fi
