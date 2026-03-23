# SFT 流程（中文精简版）

## 1) 训练流程

1. 准备环境：
   - 激活 `verl` 环境（参考 `activate_verl.sh`）。
2. 准备模型与数据：
   - 基座模型路径通过 `SFT_MODEL_PATH` 指定。
   - 训练数据路径通过 `SFT_DATA_DIR` 指定。
3. 先做快速检查（小步数）：
   - `run_sft.sh test`
4. 正式训练：
   - `run_sft.sh run`
5. 训练产物：
   - Checkpoint 默认在 `outputs/` 下。
6. （可选）合并为 Hugging Face 格式：
   - 使用 `verl.model_merger` 将 FSDP checkpoint 合并为可直接加载的模型目录。

## 2) 评估流程

1. 设置评估模型路径（通常是训练后合并得到的目录）。
2. 先跑小规模评估（快速验证流程）。
3. 再跑全量评估：
   - 通过 `run_benchmark.sh` 执行。
4. 查看结果：
   - 日志在 `logs/`，评估输出在对应结果目录。

## 3) 最小命令示例

```bash
# 0) 激活环境
source activate_verl.sh

# 1) SFT 快速测试
sbatch -o logs/sft_test_%j.out -e logs/sft_test_%j.err run_sft.sh test

# 2) SFT 正式训练
sbatch -o logs/sft_run_%j.out -e logs/sft_run_%j.err run_sft.sh run

# 3) 评估（full）
sbatch -o logs/bench_full_%j.out -e logs/bench_full_%j.err run_benchmark.sh full
```

## 4) 说明

- 具体超参数以 `run_sft.sh` 为准。
- 你后续可将 HF 数据/模型链接替换到 `README.md` 中的占位符。
