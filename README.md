# base-model-sft-verl

GitHub: [96kevinli29/base-model-sft-verl](https://github.com/96kevinli29/base-model-sft-verl)

Minimal SFT training/evaluation workspace based on [verl](https://github.com/volcengine/verl).

## Why this repo

This project focuses on an often-missing step in open pipelines: **SFT alignment before reinforcement learning (RL)**.
For many papers, this stage is not fully open-sourced, especially for **Qwen3-XB-Base**, **Qwen3.5-XB-Base**...

This repository provides a practical SFT workflow centered on **high-difficulty math and science problems** so the base model has stronger reasoning alignment before entering RL stages (such as PPO/GRPO).

## Quick Start

```bash
git clone https://github.com/96kevinli29/base-model-sft-verl.git
cd base-model-sft-verl
```

1. Prepare environment via `activate_verl.sh` and dependencies.
2. Download model/data from Hugging Face (replace placeholders):
   - Model: `https://huggingface.co/<your-org>/<your-model-repo>`
   - Dataset: `https://huggingface.co/datasets/<your-org>/<your-dataset-repo>`
3. Set paths with env vars:
   - `SFT_MODEL_PATH`
   - `SFT_DATA_DIR`
4. Run SFT and evaluation using `run_sft.sh` and `run_benchmark.sh`.

## Docs

- Chinese SFT guide: `sft_zh.md`
- English SFT guide: `sft_en.md`
- Git upload notes: `GITHUB_UPLOAD.md`
