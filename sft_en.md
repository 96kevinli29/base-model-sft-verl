# SFT Flow (Concise English)

## 1) Training Flow

1. Prepare environment:
   - Activate the `verl` environment (see `activate_verl.sh`).
2. Prepare model and dataset:
   - Set base model path via `SFT_MODEL_PATH`.
   - Set training dataset path via `SFT_DATA_DIR`.
3. Run a short sanity job first:
   - `run_sft.sh test`
4. Run full training:
   - `run_sft.sh run`
5. Training outputs:
   - Checkpoints are saved under `outputs/`.
6. (Optional) Merge checkpoints to Hugging Face format:
   - Use `verl.model_merger` to export a loadable model directory.

## 2) Evaluation Flow

1. Set evaluation model path (usually the merged checkpoint directory).
2. Run a quick evaluation first.
3. Run full benchmark:
   - Execute via `run_benchmark.sh`.
4. Check outputs:
   - Logs under `logs/`, benchmark results under the result directory.

## 3) Minimal Command Example

```bash
# 0) Activate environment
source activate_verl.sh

# 1) SFT quick test
sbatch -o logs/sft_test_%j.out -e logs/sft_test_%j.err run_sft.sh test

# 2) SFT full run
sbatch -o logs/sft_run_%j.out -e logs/sft_run_%j.err run_sft.sh run

# 3) Evaluation (full)
sbatch -o logs/bench_full_%j.out -e logs/bench_full_%j.err run_benchmark.sh full
```

## 4) Notes

- `run_sft.sh` is the source of truth for actual hyperparameters.
- You can replace the HF model/dataset placeholders in `README.md` later.
