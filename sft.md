# SFT 训练流程文档

## 概览

基于 [verl](https://github.com/volcengine/verl) 框架对 **Qwen3-4B-Base** 进行 SFT（Supervised Fine-Tuning），使用 40k 混合数据（竞赛数学 + 科学推理），目标提升 AIME/AMC/GPQA 等 benchmark 表现。

```
数据构建 → 格式检查 → SFT 训练 → Checkpoint 合并 → Benchmark 评测
```

---

## 文件总览

### 运行脚本（项目根目录）

| 文件 | 作用 |
|------|------|
| `run_sft.sh` | SFT 主入口，SLURM 提交，支持 test/run 模式，4 节点×4 卡 |
| `run_benchmark.sh` | 6 数据集 benchmark 评测，支持 test/full 模式 |
| `activate_verl.sh` | 激活 conda `verl` 环境 + 加载 WandB Key |

### 工具脚本（`scripts/`）

| 文件 | 作用 |
|------|------|
| `scripts/build_40k_sft.py` | 从 NuminaMath-CoT、OpenThoughts、science_sft 构建 40k 混合数据集 |
| `scripts/check_sft_format.py` | 检查 parquet schema、messages 格式、chat_template 对齐 |
| `scripts/run_benchmark.py` | benchmark 主逻辑：vLLM 生成 + 答案提取 + 评分 |
| `scripts/show_sft_status.sh` | 查看 SLURM 训练任务状态 |

### 数据（`my_data/sft_40k_mix/`）

| 文件 | 说明 |
|------|------|
| `train.parquet` | 39,007 条训练样本 |
| `test.parquet` | 1,040 条验证样本 |

格式：`messages` 列，每条 `[{role: user, content: 题目}, {role: assistant, content: <think>推理</think>\n\n答案}]`

### verl 框架代码（`verl/verl/`）

| 文件 | 作用 |
|------|------|
| `trainer/sft_trainer.py` | SFT Trainer 主类：构建数据集、引擎、训练循环、WandB 日志 |
| `trainer/config/sft_trainer_engine.yaml` | SFT 默认配置（data、optim、trainer 参数） |
| `trainer/config/model/hf_model.yaml` | 模型默认配置（lora_rank=0 即全量微调） |
| `utils/dataset/multiturn_sft_dataset.py` | **MultiTurnSFTDataset**：tokenize 对话 + 计算 loss_mask（**已修改**） |
| `utils/dataset/dataset_utils.py` | SFTTensorCollator：batch 拼接 + padding |
| `utils/chat_template.py` | 提取 system_prompt / generation_prompt token 序列 |
| `utils/checkpoint/checkpoint_handler.py` | Checkpoint 保存/恢复（FSDP + dataloader state） |
| `workers/engine/fsdp/transformer_impl.py` | FSDPEngine：HF 模型加载、LoRA 应用、梯度检查点 |
| `workers/config/model.py` | HFModelConfig 数据类（lora_rank/alpha/target_modules 等） |

---

## 数据流

```
build_40k_sft.py                 run_sft.sh
       │                              │
       ▼                              ▼
 sft_40k_mix/              verl.trainer.sft_trainer
 train.parquet                        │
                                      ▼
                           MultiTurnSFTDataset.__getitem__()
                                      │
                           apply_chat_template(context)
                                      │
                                      ▼
                           input_ids + loss_mask (只在 assistant 上计算 loss)
                                      │
                                      ▼
                           FSDPEngine (全量微调 or LoRA)
                                      │
                                      ▼
                           outputs/sft_qwen3_4b_40k-runN/global_step_*
                                      │
                     python -m verl.model_merger merge
                                      │
                                      ▼
                           Qwen3-4B-SFT*/  (HF 格式模型)
                                      │
                           run_benchmark.sh → run_benchmark.py
                                      │
                                      ▼
                           AIME/AMC/GPQA/GSM8K/MATH 评分
```

---

## 关键配置（`run_sft.sh`）

```bash
data.train_files=".../train.parquet"
data.max_length=8192                    # 单条最大 token 长度
data.max_token_len_per_gpu=24576        # 每 GPU 最大 token 总量（动态 batch）
data.enable_thinking_default=true       # 全部样本启用 <think> 模式

model.path=Qwen3-4B-Base
engine.model_dtype=bfloat16
model.enable_gradient_checkpointing=true

# 全量微调（不传 model.lora_rank 则默认 0 → 全量微调）
# 恢复 LoRA: 加 model.lora_rank=64 model.lora_alpha=128

optim.lr=2e-5
optim.lr_scheduler_type=cosine
optim.weight_decay=0.1
trainer.total_epochs=5
```

---

## 对 verl 原版的修改

**仅修改一个文件**：`verl/verl/utils/dataset/multiturn_sft_dataset.py`

### 问题

verl 原版的 `_process_single_message()` 对每条消息**独立** tokenize：

```python
# 原版：只传入单条消息
inputs = processor.apply_chat_template([message], ...)
```

Qwen3 的 chat_template 在处理 assistant 消息时，依赖对话上下文判断是否保留 `<think>` 块：

```jinja
{%- if loop.index0 > ns.last_query_index %}
    {{- '<think>\n' + reasoning + '\n</think>\n\n' + answer }}
{%- else %}
    {{- answer }}  ← 单独 tokenize 时走这里，<think> 被丢弃！
{%- endif %}
```

**后果**：模型训练时**从未见过** `<think>` 推理过程，只学到了最终答案（如 `The answer is \boxed{3}`），丢失 ~98% 的训练内容。

### 修复

改为**上下文感知**的 tokenize，通过前缀差分提取当前消息的 token：

```python
# 修复后：传入完整对话上下文
context = full_message[:index + 1]
inputs = processor.apply_chat_template(context, ...)

# 减去前缀，得到当前消息的 token
if index > 0:
    prefix_inputs = processor.apply_chat_template(full_message[:index], ...)
    prefix_len = prefix_inputs["input_ids"].shape[-1]
    input_ids = full_input_ids[prefix_len:]
```

同时修改 `__getitem__` 中的调用，始终传入 `tools`（上下文 tokenize 需要完整 tools 信息）。

### 修复效果

| | 修复前 | 修复后 |
|---|---|---|
| assistant tokens (sample 0) | 21（只有答案） | **163**（推理 + 答案） |
| `<think>` 保留 | ✗ | ✓ |
| concat == whole-conversation | ✗ mismatch | ✓ 完全一致 |

### Diff 摘要

```diff
# _process_single_message(): 上下文感知 tokenize
- inputs = processor.apply_chat_template([message], ...)
+ context = full_message[:index + 1]
+ inputs = processor.apply_chat_template(context, ...)
  ...
- input_ids = inputs.pop("input_ids")[0]
- if index != 0: input_ids = input_ids[len(self.system_prompt):]
+ full_input_ids = inputs.pop("input_ids")[0]
+ if index > 0:
+     prefix_len = apply_chat_template(full_message[:index])["input_ids"].shape[-1]
+     input_ids = full_input_ids[prefix_len:]

# __getitem__(): tools 始终传入
- tools=tools if i == 0 else None,
+ tools=tools,
```

此修改对非 Qwen3 模板（Llama、ChatML 等）无影响——对它们来说上下文 tokenize 和单独 tokenize 结果一致。

### 修改 2：验证样本可视化（`verl/verl/trainer/sft_trainer.py`）

在 `SFTTrainer` 中新增 `_log_sample_table()` 方法，每次验证步骤（`test_freq` 间隔）自动向 WandB 记录一个 `val/samples` Table：

| 列 | 内容 |
|---|---|
| `step` | 当前 global step |
| `seq_len` | 样本总 token 数 |
| `loss_tokens` | 计算 loss 的 token 数（assistant 部分） |
| `prompt` | 用户 prompt（截断） |
| `target_response` | 完整 assistant 目标（含 `<think>` 块） |

通过该 Table 可以在 WandB 中直接确认：
- `<think>` 推理块是否存在于训练目标中
- `loss_tokens` 是否合理（不是 0 或负数）
- 样本内容质量

---

## 训练与评测命令

```bash
# 格式检查
source activate_verl.sh && python scripts/check_sft_format.py

# 测试（5 步验证）
sbatch -o logs/sft_test_%j.out -e logs/sft_test_%j.err run_sft.sh test

# 正式训练
sbatch -o logs/sft_run5_%j.out -e logs/sft_run5_%j.err run_sft.sh run5

# 合并 checkpoint（FSDP → HuggingFace 格式）
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir outputs/sft_qwen3_4b_40k-run5/global_step_2437 \
  --target_dir Qwen3-4B-SFT5-1epoch \
  --trust-remote-code

# 评测（开启 thinking）
ENABLE_THINKING=1 MODEL_PATH=Qwen3-4B-SFT5-1epoch \
  sbatch -o logs/bench_full_%j.out -e logs/bench_full_%j.err run_benchmark.sh full
```

---

## 注意事项

1. 修复后应**去掉** `run_sft.sh` 中的 `data.ignore_input_ids_mismatch=true`，让 sanity check 自然通过以验证修复正确
2. Benchmark 时必须加 `ENABLE_THINKING=1`，因为模型训练时学的是带 `<think>` 的格式
3. 推荐先用 **LoRA (r=64) + 1-2 epochs** 快速验证效果，确认后再尝试全量微调
4. Val loss 曲线是判断过拟合的关键指标：如果 epoch 2 后 val_loss 上升，应使用更早的 checkpoint
