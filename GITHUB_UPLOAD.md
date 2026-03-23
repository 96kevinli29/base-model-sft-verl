# GitHub 上传清单

本文档说明如何把本目录作为**代码仓库**推到 GitHub（不含大数据与模型权重）。

## 1. 本仓库会包含什么

| 类型 | 说明 |
|------|------|
| 脚本与配置 | `run_sft.sh`、`run_benchmark.sh`、`merge_fsdp.sh`、`configs/`、`scripts/`、`activate_verl.sh` 等 |
| 文档 | `sft.md`、`docs/`、各子目录 README |
| `verl/` | 训练依赖的 verl 代码（见下文「嵌套 Git」） |

## 2. 不会进入 Git 的内容（已由根目录 `.gitignore` 排除）

- `.wandb_key`、`.wandb_entity` 等密钥与本地配置
- `data/`、`my_data/`、各类 `Qwen3-*` 模型与 checkpoint 目录
- `logs/`、`wandb/`、`outputs/`、`artifacts/`、`results/`
- `.hf_cache/`、`tools/miniconda3/`、`.local_tools/`

**上传前请确认**：仓库里没有任何密钥；若曾误提交过密钥，应在对应平台**轮换密钥**。

## 3. 嵌套仓库：`verl/` 与 `Qwen3-*-Base/`

当前 `verl/` 内存在独立的 `.git`。若你在**项目根目录**执行 `git init` 并 `git add .`，Git 对 `verl/` 的处理取决于版本与设置，常见做法是二选一：

**方案 A — 把 `verl` 当作普通目录一并提交（适合「整包备份」）**

若已执行过 `git add .` 且 `git status` 里 `verl` 显示为嵌入子模块（如 `Am verl`），先取消暂存再删内层 `.git`：

```bash
git rm --cached verl
rm -rf verl/.git
git add verl/
```

若尚未 `git add`，可直接：

```bash
rm -rf verl/.git
git add verl/
```

（删除 `verl/.git` 前建议自行备份；之后 `verl` 不再独立跟踪上游 commit。）

**方案 B — 使用子模块（适合长期跟踪上游 verl）**

在 GitHub 上新建空仓库后，按官方文档将 `verl` 添加为 submodule，或继续从上游 clone verl 再链接。

根目录下若还有带 `.git` 的模型目录（例如 `Qwen3-8B-Base`），已被 `.gitignore` 整体忽略，不会进入本仓库；无需额外处理。

## 4. 初始化并推送到 GitHub

在**本机项目根目录**执行（将 `YOUR_USER` / `YOUR_REPO` 换成你的）：

```bash
cd /path/to/hyl_ppo
git init
git branch -M main
git add .
git status   # 检查：不应出现 data/、my_data/、Qwen3-*/、.wandb_key 等
git commit -m "Initial commit: SFT scripts and configs"
git remote add origin https://github.com/96kevinli29/base-model-sft-verl.git
git push -u origin main
```

若 GitHub 要求认证，使用 **Personal Access Token (classic)** 或 **SSH**，勿把 token 写进仓库。

## 5. 克隆后在新环境怎么用

1. `git clone` 本仓库。  
2. 按 `README.md` / `sft.md` 准备 conda/verl 环境与依赖。  
3. 将基座模型放到例如 `Qwen3-8B-Base/`（或设置 `SFT_MODEL_PATH` 为绝对路径），数据放到 `my_data/...`（或覆盖 `SFT_DATA_DIR`）。  
4. 本地创建 `.wandb_key`（或环境变量）**不要提交**。

数据与模型建议用对象存储或 Hugging Face Hub 单独分发，仓库内只保留路径说明与 `README`。

与 **GitHub + Hugging Face Hub** 的分工、`huggingface-cli download` 示例及占位符说明见根目录 **[README.md](./README.md)**（中英双语）。
