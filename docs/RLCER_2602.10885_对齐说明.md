# RLCER (arXiv 2602.10885) 与当前 SFT/测试 设置对齐说明

论文：**Reinforcing Chain-of-Thought Reasoning with Self-Evolving Rubrics**  
项目页：https://alphalab-ustc.github.io/rlcer-alphalab/

---

## 1. 论文里的设置（从正文 + 项目页归纳）

### 1.1 评估（Evaluation）

- **数据集**：AIME24、AIME25、AMC23、GPQA-Diamond、SuperGPQA 子集（项目页）；正文 Figure 1 为「三个数学数据集」。
- **评估方式**：**pass@1 under multi-sample decoding**（多采样解码后按 pass@1 报告）。
- **模型**：Qwen 4B/8B 骨干。
- **训练数据**：基于 **DAPO-Math-17k**（RL 阶段）；论文重点是 RL+RLCER，不是 SFT 配方。
- 正文 **outcome reward** 与 DAPO 一致：答案等价则 +1，否则 -1（与当前 verifier 二值判分一致）。

### 1.2 SFT / 冷启动

- 论文未在项目页或抓取到的正文里给出 SFT 的 **max_length、lr、epoch、batch** 等具体超参。
- 训练流程是：RLVR（含 RLCER）基于 DAPO-Math-17k，不强调单独 SFT 阶段设置。

---

## 2. 当前项目设置

### 2.1 SFT（`run_sft.sh`）

| 项 | 当前值 | 说明 |
|----|--------|------|
| 数据 | OpenThoughts math/science | 与论文 DAPO-Math-17k 不同数据源 |
| 模型 | Qwen3-4B-Base | 与论文 Qwen 4B 同族 |
| max_length | 32768 | 论文未给，常见 8k–32k |
| lr | 2e-5 | 论文未给，常见 1e-5–2e-5 |
| epochs | 5（正式）/ 1（测试） | 论文未给 |
| LoRA rank | 32 | 论文未给 |

### 2.2 测试（`run_six_sets_test.sh` / `scripts/run_benchmark.py`）

| 项 | 当前值 | 论文/项目页 |
|----|--------|-------------|
| 数据集 | aime_2024, aime_2025, amc23, GPQA-Diamond, gsm8k, MATH-lighteval | 前 4 个与 RLCER 一致；多 gsm8k、MATH-lighteval |
| 每题采样数 n | 5 | 与「multi-sample decoding」一致，可算 pass@1、best@5、maj@5 |
| temperature | 0.7 | 论文未明确，常见 0.6–0.8 |
| max_new_tokens | 4084 | 论文未明确，数学题常用 2k–4k |
| 判分 | 答案等价二值 + verl 各 data_source 的 verifier | 与论文 outcome reward（等价=1 否则 -1）一致 |

---

## 3. 对齐结论

### 3.1 已对齐的部分

1. **评估数据集**：aime_2024、aime_2025、amc23、GPQA-Diamond 与 RLCER 一致；你方多 gsm8k、MATH-lighteval，属于常见数学/推理 benchmark 扩展，可保留。
2. **评估协议**：多采样（n=5）+ 按题聚合（mean@5、best@5、maj@5）与论文「pass@1 under multi-sample decoding」一致，且你方多报了 best/maj 指标。
3. **答案判分**：二值（正确/错误）、数学等价与论文/DAPO 的 outcome reward 一致；verl 的 `data_source` 与各数据集 verifier 与常见做法一致。

### 3.2 无法逐项对齐的部分（论文未给出）

- **SFT**：论文未给出 SFT 的 max_length、lr、epochs、batch 等，因此无法说「与论文完全一致」；当前 OpenThoughts + 5 epoch、32k、2e-5 是合理且常见的 SFT 配置。
- **测试**：temperature、max_new_tokens 论文未写，当前 0.7、4084 在常见范围内，无需为对齐论文而改。

### 3.3 若严格复现 RLCER 的评估

- 数据集：仅用 AIME24、AIME25、AMC23、GPQA-Diamond（可再加 SuperGPQA 若你有数据）。
- 指标：报告 **pass@1**（即每题 1 个样本正确即算对，或从 n 次采样中取 best@1 作为 pass@1）；你方已有 mean@5 / best@5，可把 **best@1** 或单次 run 的准确率当作 pass@1。
- 训练：若要做 RLCER 复现，需用 DAPO-Math-17k 做 RL（且加 RLCER 的 rubricator 等），与当前仅 SFT + 离线评测 不同。

---

## 4. 总结表

| 维度 | 论文 RLCER | 当前项目 | 是否对齐 |
|------|------------|----------|----------|
| 评估数据集（核心） | AIME24/25, AMC23, GPQA-Diamond | 含上述 + gsm8k + MATH-lighteval | 是（你方为超集） |
| 评估方式 | pass@1, multi-sample | n=5，mean@5 / best@5 / maj@5 | 是 |
| 答案 reward | 二值、等价即正确 | verl 各 data_source 二值判分 | 是 |
| temperature | 未给出 | 0.7 | 无法对照，合理 |
| max_new_tokens | 未给出 | 4084 | 无法对照，合理 |
| SFT 数据 | 未强调，RL 用 DAPO-Math-17k | OpenThoughts | 不同流程，SFT 超参论文未给 |
| SFT 超参 | 未给出 | 32k, 2e-5, 5 epoch, LoRA 32 | 无法对照，合理 |

**结论**：  
- **测试设置**与 RLCER 论文/项目页描述**对齐**：数据集重叠、多采样 pass@1、二值判分一致；temperature/max_new_tokens 论文未写，当前取值合理。  
- **SFT 设置**论文未给出具体超参，无法逐项对齐；当前 OpenThoughts + 32k/2e-5/5 epoch 是常见且合理的 SFT 配置，与「用 RLCER 做评估」不冲突。

若你之后拿到 PDF 里附录的完整实验设置（例如 SFT 的 max_length、lr、epoch 或评估的 temperature、max_tokens），可以再根据附录把本说明补一栏「论文附录」做逐项对比。
