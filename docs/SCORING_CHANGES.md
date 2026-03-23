# Benchmark 打分逻辑说明与相对原版的修改

## 一、总体说明（Intro）

本仓库的 6 数据集 Benchmark 使用**两种**打分逻辑，与 verl 标准一致：

| 类型 | 数据集 | 实现位置 | 说明 |
|------|--------|----------|------|
| **math_verify** | aime2024, aime2025, amc2023, gsm8k, math_lighteval（5 个） | `verl/utils/reward_score/math_verify.py` | 从模型输出抽答案（LaTeX/表达式），用 math_verify 做数学等价判定；标准答案包成 `\boxed{ground_truth}` 参与验证。等价 1.0，否则 0.0；异常/超时 0.0。 |
| **GPQA 选项匹配** | gpqa_diamond（1 个） | `verl/utils/reward_score/__init__.py` 中 `_gpqa_compute_score` | 在模型输出末 600 字内优先匹配 `Answer : X`、`\boxed{X}`、`\boxed{\text{X}}`，否则在 “Answer” 后或末 100 字取最后一个 A–D，与标准答案字母（忽略大小写）比较，一致 1.0 否则 0.0。 |

Benchmark 入口脚本 `scripts/run_benchmark.py` 中：

- 数学 5 集：`score_one` → `score_math_verify`（调用 `math_verify.compute_score`）。
- GPQA：`score_one` → `score_gpqa`（内部调用 `default_compute_score("gpqa_diamond", ...)`，即走上述 GPQA 逻辑）。

---

## 二、相对原版的修改（Modifications）

### 1. `verl/verl/utils/reward_score/__init__.py`（GPQA 打分）

**原版逻辑：**

- 只取模型输出**末 500 字符**。
- 仅做两件事：
  1. 找**最后一个** `\boxed{...}`，解析花括号取内容作为预测；
  2. 若没有 `\boxed{}`，在**末 100 字符**用 `\b([A-Da-d])\b` 取最后一个字母。
- 没有对 “Answer : X” 或 “\boxed{\text{X}}” 的显式支持。

**修改后逻辑：**

- 取模型输出**末 600 字符**（不足 600 则整段）。
- 新增两个正则，与 simple-evals / 常见 GPQA 评测约定对齐：
  - `_GPQA_ANSWER_PATTERN`：`(?i)Answer[ \t]*:[ \t]*\$?([A-Da-d])\$?`（匹配 “Answer : X”）。
  - `_GPQA_BOXED_PATTERN`：`\\boxed{\s*(?:\\text{\s*)?([A-Da-d])\s*(?:\}\s*)*\}`（匹配 `\boxed{X}`、`\boxed{\text{X}}`）。
- 在末 600 字内，对上述两个模式做 **finditer**，按**出现位置**取**最后一次**匹配作为预测。
- 若没有任何匹配，则 **fallback**：
  - 若存在子串 `"Answer"`，则在**最后一个 “Answer” 之后**的片段内用 `\b([A-Da-d])\b` 取最后一个字母；
  - 否则在**末 100 字符**内取最后一个 A–D。
- 预测与 `ground_truth` 忽略大小写比较，一致返回 1.0，否则 0.0。
- 新增 `_compute_gpqa_choice_score(solution_str, ground_truth) -> float`，与上述逻辑一致，供需要纯 float 的调用方使用。
- `_gpqa_compute_score` 改为复用同一套逻辑，返回值仍为 `{"score", "acc", "pred"}`，供 `default_compute_score` 使用。

**小结：** 原版只有“末 500 字 + 最后一个 \boxed{} 或末 100 字 A–D”；现版改为“末 600 字 + 优先 Answer : X / \boxed{X} / \boxed{\text{X}}（按位置取最后一次）+ 同上 fallback”，与常见 GPQA 评测标准一致。

---

### 2. `scripts/run_benchmark.py`（GPQA 打分入口与说明）

**原版：**

- `score_gpqa` 使用脚本内的 `extract_answer(response, GPQA_DATASET)`：
  - 取末 800 字，先取最后一个 `\boxed{...}`，没有则在末 200 字用 `\b([A-Da-d])\b` 取最后一个字母。
  - 与 `ground_truth` 比较后返回 `{"correct", "pred", "gt"}`。
- 与 verl 里 `_gpqa_compute_score` 的规则（500 字、仅 \boxed 与末 100 字）不一致，且没有 “Answer : X” 等标准格式。

**修改后：**

- GPQA 统一走 verl：`score_gpqa` 内部调用 `default_compute_score("gpqa_diamond", response, ground_truth)`，不再用本地的 `extract_answer` 做 GPQA 判分。
- 对返回值的处理：
  - 若 `res` 为 dict：`score = float(res.get("score", res.get("acc", 0)))`，`pred = res.get("pred")`。
  - 否则：`score = float(res)`，`pred = None`。
  - `correct = score > 0`，返回 `{"correct": correct, "pred": pred, "gt": ground_truth}`。
- 文件顶部说明与注释已更新：GPQA 使用 verl 的 `default_compute_score`，规则为“末 600 字内优先 Answer : X / \boxed{X}，否则在 Answer 后或末 100 字取最后 A–D”。
- `extract_answer` 保留供调试/其他用途，其文档注释已说明：GPQA 打分不再经由此函数，由 verl 标准实现负责。

**小结：** Benchmark 的 GPQA 打分与 verl 的 `_gpqa_compute_score` / `default_compute_score` 完全一致，并统一了返回值格式（correct / pred / gt）。

---

## 三、涉及文件一览

| 文件 | 修改内容 |
|------|----------|
| `verl/verl/utils/reward_score/__init__.py` | 新增 GPQA 正则与 `_compute_gpqa_choice_score`；`_gpqa_compute_score` 改为 600 字 + Answer/\boxed 优先 + fallback。 |
| `scripts/run_benchmark.py` | `score_gpqa` 改为调用 `default_compute_score("gpqa_diamond", ...)` 并统一解析 dict/float；更新顶部说明与 `extract_answer` 注释。 |

math_verify 相关逻辑（`math_verify.py` 及 5 个数学集的调用）未改，仍为原版行为。
