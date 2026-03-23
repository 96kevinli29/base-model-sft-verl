# base-model-sft-verl

GitHub: [96kevinli29/base-model-sft-verl](https://github.com/96kevinli29/base-model-sft-verl)

Minimal SFT training/evaluation workspace based on [verl](https://github.com/volcengine/verl).

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
