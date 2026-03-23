# Cursor 新建终端：固定进入项目目录并激活 verl（路径与 docs/环境配置说明 一致）
# 项目规范路径，不可改错
PROJECT_ROOT="/project/home/p201251/hyl_ppo"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "[错误] 项目目录不存在: $PROJECT_ROOT"
  return 1 2>/dev/null || exit 1
fi

source "$PROJECT_ROOT/activate_verl.sh" || return 1
cd "$PROJECT_ROOT" || return 1
echo "[verl] 项目目录: $PROJECT_ROOT   ($(pwd))"
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
