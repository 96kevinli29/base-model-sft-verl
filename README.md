# hyl_ppo

**GitHub：** [96kevinli29/base-model-sft-verl](https://github.com/96kevinli29/base-model-sft-verl)

[简体中文](#简体中文) · [English](#english)

---

## 简体中文

### 概述

本仓库为基于 **[verl](https://github.com/volcengine/verl)** 的 Qwen3 **监督微调（SFT）**与评测相关脚本与配置。**代码在 GitHub；大体积基座模型、微调权重与训练数据在 Hugging Face Hub（或其它对象存储）**，二者通过环境变量与本地的目录约定联动（见下文）。

### GitHub 与 Hugging Face 如何配合

| 平台 | 放什么 | 本仓库中的位置 |
|------|--------|----------------|
| **GitHub** | 训练/评测脚本、`configs/`、`scripts/`、`verl/`、文档 | 本仓库 |
| **Hugging Face Hub** | 基座或微调后的 **Model**、处理好的 **Dataset**（可选 **Space** 做演示） | 不在 Git 中；运行时下载或同步到本地目录 |

推荐工作流：

1. 在 GitHub 维护**可复现的代码与说明**（本仓库）。  
2. 在 Hugging Face 上创建 **Model** / **Dataset** 仓库，上传权重与 parquet/jsonl 等（或使用已有公开资源）。  
3. 新机器上：`git clone` 本仓库 → 用 `huggingface-cli download` 或 `git lfs clone` 把模型与数据拉到**约定路径**（或通过 `SFT_MODEL_PATH`、`SFT_DATA_DIR` 指到下载目录）。  
4. **不要**把 `HF_TOKEN`、`.wandb_key` 写进仓库；使用环境变量或本地忽略文件。

```bash
# 示例：登录 HF（写入 ~/.cache/huggingface/token，勿提交仓库）
huggingface-cli login

# 示例：下载模型到本地目录（与 run_sft.sh 中 SFT_MODEL_PATH 约定一致时可直接训练）
huggingface-cli download <your-org>/<your-model-repo> --local-dir ./Qwen3-8B-Base

# 示例：下载数据集到 my_data 下某一子目录
huggingface-cli download <your-org>/<your-dataset-repo> --repo-type dataset --local-dir ./my_data/<your_dataset_name>
```

将下方占位符换成你在 HF 上的实际仓库名，并在 README 或 Wiki 里固定记录，便于协作者对齐：

- 模型：`https://huggingface.co/<your-org>/<your-model-repo>`  
- 数据：`https://huggingface.co/datasets/<your-org>/<your-dataset-repo>`

### 快速开始（新环境）

1. `git clone https://github.com/96kevinli29/base-model-sft-verl.git && cd base-model-sft-verl`  
2. 按 `sft.md`、`activate_verl.sh` 准备 conda / verl 环境。  
3. 从 HF 下载或使用软链接，使 **`SFT_MODEL_PATH`**（默认如 `Qwen3-8B-Base`）与 **`SFT_DATA_DIR`**（如 `my_data/sft_50k_apex`）指向真实路径。  
4. 参考 `run_sft.sh` 头部注释提交 Slurm 任务或本地调试。

更细的 **Git 忽略规则与 `verl` 嵌套仓库处理** 见 **[GITHUB_UPLOAD.md](./GITHUB_UPLOAD.md)**。

### 目录说明（精简）

| 路径 | 说明 |
|------|------|
| `run_sft.sh` | SFT 主入口（`SFT_MODEL_PATH`、`SFT_DATA_DIR`、`SFT_EXPERIMENT_NAME` 等） |
| `run_benchmark.sh` | 评测 |
| `configs/`、`scripts/` | 配置与数据处理 |
| `verl/` | verl 训练框架 |
| `sft.md` | SFT 笔记与说明 |

---

## English

### Overview

This repository contains **supervised fine-tuning (SFT)** and evaluation scripts for Qwen3, built on **[verl](https://github.com/volcengine/verl)**. **Source code lives on GitHub; base weights, fine-tuned checkpoints, and large datasets are expected on [Hugging Face Hub](https://huggingface.co/) (or other object storage)** and are wired in via environment variables and local path conventions.

### How GitHub and Hugging Face work together

| Platform | What to store | In this repo |
|----------|----------------|--------------|
| **GitHub** | Training/eval scripts, `configs/`, `scripts/`, `verl/`, docs | This repository |
| **Hugging Face Hub** | **Models** (base or fine-tuned), **Datasets** (processed parquet/jsonl, etc.) | Not in Git; download or sync at runtime |

Suggested workflow:

1. Keep **reproducible code and docs** on GitHub (this repo).  
2. Create **Model** / **Dataset** repos on Hugging Face and upload weights and data (or rely on existing public artifacts).  
3. On a new machine: `git clone` this repo → use `huggingface-cli download` (or LFS) into paths that match **`SFT_MODEL_PATH`** and **`SFT_DATA_DIR`**.  
4. Never commit **`HF_TOKEN`**, `.wandb_key`, or secrets; use environment variables or local-only files excluded by `.gitignore`.

```bash
# Log in (token stored under ~/.cache/huggingface by default — do not commit)
huggingface-cli login

# Example: model into a local dir that matches your SFT_MODEL_PATH
huggingface-cli download <your-org>/<your-model-repo> --local-dir ./Qwen3-8B-Base

# Example: dataset into my_data/
huggingface-cli download <your-org>/<your-dataset-repo> --repo-type dataset --local-dir ./my_data/<your_dataset_name>
```

Replace placeholders with your real HF org/repo URLs and document them for collaborators:

- Model: `https://huggingface.co/<your-org>/<your-model-repo>`  
- Dataset: `https://huggingface.co/datasets/<your-org>/<your-dataset-repo>`

### Quick start (fresh environment)

1. `git clone https://github.com/96kevinli29/base-model-sft-verl.git && cd base-model-sft-verl`  
2. Set up conda / verl as described in `sft.md` and `activate_verl.sh`.  
3. Download or symlink assets so **`SFT_MODEL_PATH`** and **`SFT_DATA_DIR`** resolve to real paths.  
4. See the header of `run_sft.sh` for Slurm or local runs.

For **what is ignored by Git and how to handle the nested `verl` repo**, see **[GITHUB_UPLOAD.md](./GITHUB_UPLOAD.md)**.

### Repository layout (short)

| Path | Role |
|------|------|
| `run_sft.sh` | Main SFT entry (`SFT_MODEL_PATH`, `SFT_DATA_DIR`, `SFT_EXPERIMENT_NAME`, …) |
| `run_benchmark.sh` | Evaluation |
| `configs/`, `scripts/` | Config and data utilities |
| `verl/` | verl framework |
| `sft.md` | SFT notes |
