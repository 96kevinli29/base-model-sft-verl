"""
构建 40K 混合 SFT 数据集 → my_data/sft_40k_v2/

数据类型        比例     数量     数据来源
─────────────────────────────────────────────────────────
高难度数学竞赛   50%     20,000   OpenR1-Math 精选 12k + NuminaMath-CoT 竞赛级 8k
通用复杂逻辑     30%     12,000   Magpie-Pro-Filtered
代码与算法       10%      4,000   CodeFeedback-Filtered
科学推理/论文级  10%      4,000   ScienceQA 2,800 + GPQA-Science 1,200
─────────────────────────────────────────────────────────

约束:
  - 所有样本总 length < 8K tokens (含 thinking), 按 ~3 chars/token 估算
  - \\boxed{} 与否取决于具体问题类型
  - 数学竞赛 → <think>推理</think>\\n\\n{answer}, 尊重推导过程
  - 通用逻辑/代码 → 直接回答, 不加假 thinking
  - 科学推理 → 复杂题用 <think>, 简单事实题直接回答, 尊重数据本身目的
  - GPQA → 保留原始格式 (多数含 <think>), 不强制过滤
"""

import glob
import json
import random
import re
import os
from typing import List, Dict, Set

import pandas as pd
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE = "/project/home/p201251/hyl_ppo"
OUT_DIR = os.path.join(BASE, "my_data", "sft_40k_v2")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_TOKENS = 8000
CHARS_PER_TOKEN = 3
MAX_TOTAL_CHARS = MAX_TOKENS * CHARS_PER_TOKEN  # 24000

# ─── 工具函数 ──────────────────────────────────────────────────

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def msgs_within_limit(msgs: List[Dict], max_tokens: int = MAX_TOKENS) -> bool:
    total = sum(estimate_tokens(m["content"]) for m in msgs)
    return total < max_tokens


def find_all_boxed(text: str) -> List[str]:
    results = []
    prefix = "\\boxed{"
    start = 0
    while True:
        idx = text.find(prefix, start)
        if idx == -1:
            break
        brace_start = idx + len(prefix) - 1
        depth = 1
        pos = brace_start + 1
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            results.append(text[idx:pos])
        start = pos
    return results


def has_think_tags(content: str) -> bool:
    return "<think>" in content and "</think>" in content


def has_quality_think(content: str) -> bool:
    match = THINK_RE.search(content)
    if not match:
        return False
    think_body = match.group(1).strip()
    if len(think_body) < 20:
        return False
    after_think = content.split("</think>")[-1].strip()
    if think_body == after_think:
        return False
    return True


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def token_set(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_ws(text)))


def jaccard_similarity(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_bad_openr1_assistant(content: str) -> bool:
    """轻量质量过滤: 只排除空内容和明显垃圾, 长度由 8K token 限制统一控制"""
    if not content or not content.strip():
        return True
    if len(content.strip()) < 50:
        return True
    if has_think_tags(content):
        think_match = THINK_RE.search(content)
        if think_match:
            think_body = think_match.group(1).strip()
            tail = content.split("</think>")[-1].strip()
            sim = jaccard_similarity(think_body, tail)
            if sim >= 0.85:
                return True
    return False


def is_proof_problem(content: str) -> bool:
    proof_markers = ["\\blacksquare", "\\qed", "Q.E.D.", "QED", "□"]
    return any(m in content for m in proof_markers)


def wrap_math_in_think(content: str) -> str:
    if has_think_tags(content):
        return fix_think_final_answer(content)
    boxed = find_all_boxed(content)
    if boxed:
        return f"<think>\n{content}\n</think>\n\nThe final answer is ${boxed[-1]}$."
    if is_proof_problem(content):
        return f"<think>\n{content}\n</think>\n\n$\\blacksquare$"
    lines = content.strip().split("\n")
    last_line = lines[-1].strip() if lines else content.strip()
    return f"<think>\n{content}\n</think>\n\n{last_line}"


def fix_think_final_answer(content: str) -> str:
    after = content.split("</think>")[-1].strip()
    if len(after) >= 10:
        return content
    think_match = THINK_RE.search(content)
    if not think_match:
        return content
    think_body = think_match.group(1)
    boxed = find_all_boxed(think_body)
    if boxed:
        return f"<think>\n{think_body}\n</think>\n\nThe final answer is ${boxed[-1]}$."
    if is_proof_problem(think_body):
        return f"<think>\n{think_body}\n</think>\n\n$\\blacksquare$"
    return content


def wrap_science_in_think(solution: str, answer: str) -> str:
    return f"<think>\n{solution}\n</think>\n\nThe answer is: {answer}"


def to_messages(user_content: str, assistant_content: str) -> List[Dict]:
    return [
        {"role": "user", "content": user_content.strip()},
        {"role": "assistant", "content": assistant_content.strip()},
    ]


def validate_messages(msgs, require_think: bool = False) -> bool:
    if not isinstance(msgs, (list, np.ndarray)) or len(msgs) < 2:
        return False
    for m in msgs:
        if not isinstance(m, dict):
            return False
        if "role" not in m or "content" not in m:
            return False
        if not m["content"] or not m["content"].strip():
            return False
    roles = [m["role"] for m in msgs]
    if "user" not in roles or "assistant" not in roles:
        return False
    if require_think:
        for m in msgs:
            if m["role"] == "assistant":
                if not has_think_tags(m["content"]):
                    return False
                if not has_quality_think(m["content"]):
                    return False
    return True


def normalize_messages(msgs) -> List[Dict]:
    if isinstance(msgs, np.ndarray):
        msgs = msgs.tolist()
    result = []
    for m in msgs:
        result.append({"role": m["role"], "content": m["content"].strip()})
    return result


TOXIC_STRINGS = [
    "</s>", "<eos>", "<|endoftext|>", "<|im_end|>", "<|im_start|>",
    "\nHuman:", "\nAssistant:", "<|eot_id|>", "<|end_of_turn|>",
]


def is_toxic(content: str) -> bool:
    return any(tok in content for tok in TOXIC_STRINGS)


# ──────────────────────────────────────────────────────────────
# 1. 高难度数学竞赛: 20,000 条 (50%)
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("[1/4] 高难度数学竞赛 (目标 20,000)")
print("=" * 60)

# 1a) OpenR1-Math → 12,000 条 (30%)
print("  加载 OpenR1-Math ...")
or1_files = sorted(glob.glob(f"{BASE}/data/OpenR1-Math-220k/data/train-*.parquet"))
or1_dfs = []
for f in or1_files:
    df = pd.read_parquet(f, columns=["messages", "source", "correctness_count", "problem_type"])
    or1_dfs.append(df)
or1 = pd.concat(or1_dfs, ignore_index=True)
print(f"  OpenR1-Math 原始: {len(or1)}")

or1 = or1[or1["correctness_count"] >= 1]
print(f"  过滤 correctness_count>=1 后: {len(or1)}")

before_or1_qc = len(or1)
or1["bad_openr1"] = or1["messages"].apply(
    lambda x: (
        True
        if not isinstance(x, (list, np.ndarray)) or len(x) <= 1 or "content" not in x[1]
        else is_bad_openr1_assistant(str(x[1]["content"]))
    )
)
or1 = or1[~or1["bad_openr1"]]
print(f"  质量过滤移除: {before_or1_qc - len(or1)}")
print(f"  质量过滤后: {len(or1)}")

hard_sources = ["amc_aime", "aops_forum", "number_theory", "inequalities", "olympiads_ref"]
or1_hard = or1[or1["source"].isin(hard_sources)]
or1_rest = or1[~or1["source"].isin(hard_sources)]
target_or1 = 12000
n_rest = max(0, target_or1 - len(or1_hard))
or1_rest_sample = or1_rest.sample(n=min(n_rest, len(or1_rest)), random_state=SEED)
or1_sample = pd.concat([or1_hard, or1_rest_sample], ignore_index=True)
print(f"  竞赛级 source 全纳入: {len(or1_hard)}, 补充其他: {len(or1_rest_sample)}")

or1_msgs = []
for _, row in or1_sample.iterrows():
    msgs = normalize_messages(row["messages"])
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            msgs[i]["content"] = wrap_math_in_think(m["content"])
    if validate_messages(msgs, require_think=True) and msgs_within_limit(msgs):
        or1_msgs.append({"messages": msgs, "data_source": "openr1_math", "category": "math_competition"})

if len(or1_msgs) < target_or1:
    print(f"  ⚠ OpenR1 初筛 {len(or1_msgs)} 条不足 {target_or1}, 尝试补采 ...")
    already_used = set()
    for d in or1_msgs:
        already_used.add(d["messages"][0]["content"][:200])
    or1_extra = or1_rest[~or1_rest.index.isin(or1_sample.index)]
    or1_extra = or1_extra.sample(frac=1, random_state=SEED + 1)
    for _, row in or1_extra.iterrows():
        if len(or1_msgs) >= target_or1:
            break
        msgs = normalize_messages(row["messages"])
        key = msgs[0]["content"][:200]
        if key in already_used:
            continue
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                msgs[i]["content"] = wrap_math_in_think(m["content"])
        if validate_messages(msgs, require_think=True) and msgs_within_limit(msgs):
            or1_msgs.append({"messages": msgs, "data_source": "openr1_math", "category": "math_competition"})
            already_used.add(key)

or1_msgs = or1_msgs[:target_or1]
print(f"  OpenR1-Math 最终: {len(or1_msgs)}")

# 1b) NuminaMath-CoT → 8,000 条 (20%)
print("  加载 NuminaMath-CoT ...")
numina = pd.read_parquet(f"{BASE}/my_data/NuminaMath-CoT/train.parquet")
print(f"  NuminaMath-CoT 原始: {len(numina)}")

hard_numina_sources = ["olympiads", "aops_forum", "amc_aime"]
numina_hard = numina[numina["source"].isin(hard_numina_sources)]
print(f"  过滤竞赛级 source 后: {len(numina_hard)}")

target_numina = 8000
numina_sample = numina_hard.sample(n=min(target_numina * 2, len(numina_hard)), random_state=SEED)

numina_msgs = []
for _, row in numina_sample.iterrows():
    msgs = normalize_messages(row["messages"])
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            msgs[i]["content"] = wrap_math_in_think(m["content"])
    if validate_messages(msgs, require_think=True) and msgs_within_limit(msgs):
        numina_msgs.append({"messages": msgs, "data_source": "numina_cot", "category": "math_competition"})
    if len(numina_msgs) >= target_numina:
        break

numina_msgs = numina_msgs[:target_numina]
print(f"  NuminaMath-CoT 最终: {len(numina_msgs)}")

math_data = or1_msgs + numina_msgs
print(f"  → 数学竞赛合计: {len(math_data)}")

# ──────────────────────────────────────────────────────────────
# 2. 通用复杂逻辑: 12,000 条 (30%)
# ──────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("[2/4] 通用复杂逻辑 (目标 12,000)")
print("=" * 60)

print("  加载 Magpie-Pro-Filtered ...")
magpie_files = sorted(glob.glob(f"{BASE}/data/Magpie-Llama-3.1-Pro-300K-Filtered/data/train-*.parquet"))
magpie_dfs = []
for f in magpie_files:
    df = pd.read_parquet(f, columns=["instruction", "response", "task_category", "difficulty",
                                     "instruct_reward", "input_quality"])
    magpie_dfs.append(df)
magpie = pd.concat(magpie_dfs, ignore_index=True)
print(f"  Magpie 原始: {len(magpie)}")

magpie = magpie[magpie["task_category"] != "Math"]
magpie = magpie[magpie["difficulty"].isin(["medium", "hard", "very hard"])]
magpie = magpie[magpie["instruct_reward"] > 0]
magpie = magpie.dropna(subset=["instruction", "response"])
magpie = magpie[magpie["instruction"].str.len() > 20]
magpie = magpie[magpie["response"].str.len() > 50]
print(f"  过滤后 (非Math, 中高难度, reward>0): {len(magpie)}")

diff_weight = {"medium": 1.0, "hard": 3.0, "very hard": 5.0}
magpie["weight"] = magpie["difficulty"].map(diff_weight)

target_logic = 12000
magpie_sample = magpie.sample(n=min(target_logic * 2, len(magpie)), weights="weight", random_state=SEED)

logic_data = []
for _, row in magpie_sample.iterrows():
    msgs = to_messages(row["instruction"], row["response"])
    if validate_messages(msgs) and msgs_within_limit(msgs):
        logic_data.append({"messages": msgs, "data_source": "magpie_pro", "category": "complex_logic"})
    if len(logic_data) >= target_logic:
        break

logic_data = logic_data[:target_logic]
print(f"  → 通用逻辑合计: {len(logic_data)}")

# ──────────────────────────────────────────────────────────────
# 3. 代码与算法: 4,000 条 (10%)
# ──────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("[3/4] 代码与算法 (目标 4,000)")
print("=" * 60)

print("  加载 CodeFeedback ...")
code_path = f"{BASE}/data/CodeFeedback-Filtered-Instruction/CodeFeedback-Filtered-Instruction.jsonl"
code_records = []
with open(code_path) as f:
    for line in f:
        d = json.loads(line)
        q = d.get("query", "").strip()
        a = d.get("answer", "").strip()
        if len(q) > 20 and len(a) > 50:
            code_records.append({"query": q, "answer": a})
print(f"  CodeFeedback 过滤后: {len(code_records)}")

random.shuffle(code_records)

target_code = 4000
code_data = []
for rec in code_records:
    msgs = to_messages(rec["query"], rec["answer"])
    if validate_messages(msgs) and msgs_within_limit(msgs):
        code_data.append({"messages": msgs, "data_source": "codefeedback", "category": "code_algorithm"})
    if len(code_data) >= target_code:
        break

code_data = code_data[:target_code]
print(f"  → 代码算法合计: {len(code_data)}")

# ──────────────────────────────────────────────────────────────
# 4. 科学推理/论文级: 4,000 条 (10%)
# ──────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("[4/4] 科学推理/论文级 (目标 4,000)")
print("=" * 60)

# 4a) ScienceQA → 2,800 条 (7%)
print("  加载 ScienceQA ...")
sciqa_files = sorted(glob.glob(f"{BASE}/data/TheMrguiller-ScienceQA/data/train-*.parquet"))
sciqa_dfs = []
for f in sciqa_files:
    df = pd.read_parquet(f)
    sciqa_dfs.append(df)
sciqa = pd.concat(sciqa_dfs, ignore_index=True)
print(f"  ScienceQA 原始: {len(sciqa)}")

sciqa = sciqa.sample(frac=1, random_state=SEED).reset_index(drop=True)
target_sciqa = 2800
SCIQA_THINK_THRESHOLD = 200  # solution > 200 chars → 复杂推理, 用 <think>; 否则直接回答
sciqa_records = []
for _, row in sciqa.iterrows():
    question = str(row.get("question", "")).strip()
    choices = str(row.get("choices", "")).strip()
    answer = str(row.get("answer", "")).strip()
    solution = str(row.get("solution", "")).strip()

    user_content = question
    if choices and choices not in ("None", "nan", ""):
        user_content += "\n" + choices

    has_solution = solution not in ("", "None", "nan") and len(solution) > 10
    has_answer = answer not in ("", "None", "nan")

    if not has_solution:
        continue

    if len(solution) > SCIQA_THINK_THRESHOLD:
        if has_answer:
            assistant_content = wrap_science_in_think(solution, answer)
        else:
            assistant_content = wrap_science_in_think(solution, solution.strip().split("\n")[-1])
    else:
        if has_answer:
            assistant_content = f"{solution}\n\nThe answer is: {answer}"
        else:
            assistant_content = solution

    if len(user_content) > 10:
        msgs = to_messages(user_content, assistant_content)
        if validate_messages(msgs) and msgs_within_limit(msgs):
            sciqa_records.append({"messages": msgs, "data_source": "scienceqa", "category": "science_reasoning"})
    if len(sciqa_records) >= target_sciqa:
        break

sciqa_records = sciqa_records[:target_sciqa]
print(f"  ScienceQA 有效: {len(sciqa_records)}")

# 4b) GPQA-Science → 1,200 条 (3%)
print("  加载 GPQA-Science ...")
gpqa = pd.read_parquet(f"{BASE}/data/GPQA_science/train.parquet")
print(f"  GPQA-Science 原始: {len(gpqa)}")

gpqa = gpqa.sample(frac=1, random_state=SEED).reset_index(drop=True)
target_gpqa = 1200
gpqa_records = []
for _, row in gpqa.iterrows():
    msgs = normalize_messages(row["messages"])
    if validate_messages(msgs) and msgs_within_limit(msgs):
        gpqa_records.append({"messages": msgs, "data_source": "gpqa_science", "category": "science_reasoning"})
    if len(gpqa_records) >= target_gpqa:
        break

gpqa_records = gpqa_records[:target_gpqa]
print(f"  GPQA-Science 有效: {len(gpqa_records)}")

science_data = sciqa_records + gpqa_records
print(f"  → 科学推理合计: {len(science_data)} (ScienceQA: {len(sciqa_records)}, GPQA: {len(gpqa_records)})")

# ──────────────────────────────────────────────────────────────
# 合并 & 后处理
# ──────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("合并所有数据 & 后处理")
print("=" * 60)

all_data = math_data + logic_data + code_data + science_data
print(f"合并前总计: {len(all_data)}")

# 全局去重 (user + assistant)
seen = set()
deduped = []
for d in all_data:
    key = (d["messages"][0]["content"], d["messages"][1]["content"])
    if key not in seen:
        seen.add(key)
        deduped.append(d)
print(f"去重移除: {len(all_data) - len(deduped)} 条")
all_data = deduped

# 有害 token 清洗
before_toxic = len(all_data)
all_data = [d for d in all_data if not is_toxic(d["messages"][1]["content"])]
print(f"有害 token 移除: {before_toxic - len(all_data)} 条")

# thinking 样本 final answer 格式校验
before_fmt = len(all_data)
cleaned = []
for d in all_data:
    asst = d["messages"][1]["content"]
    if "</think>" in asst:
        after = asst.split("</think>")[-1].strip()
        if len(after) < 5:
            continue
    cleaned.append(d)
all_data = cleaned
print(f"final answer 残缺移除: {before_fmt - len(all_data)} 条")

# 二次确认 8K token 限制
before_len = len(all_data)
all_data = [d for d in all_data if msgs_within_limit(d["messages"])]
print(f"8K token 超限移除: {before_len - len(all_data)} 条")

random.shuffle(all_data)

print(f"\n最终总计: {len(all_data)}")
print(f"  数学竞赛:   {sum(1 for d in all_data if d['category'] == 'math_competition')}")
print(f"  通用逻辑:   {sum(1 for d in all_data if d['category'] == 'complex_logic')}")
print(f"  代码算法:   {sum(1 for d in all_data if d['category'] == 'code_algorithm')}")
print(f"  科学推理:   {sum(1 for d in all_data if d['category'] == 'science_reasoning')}")

# 数据源细分
print(f"\n数据源细分:")
from collections import Counter
src_counts = Counter(d["data_source"] for d in all_data)
for src, cnt in src_counts.most_common():
    print(f"  {src}: {cnt} ({100 * cnt / len(all_data):.1f}%)")

# train / test 拆分 (98% / 2%, 最少 500 test)
n_test = max(500, int(len(all_data) * 0.02))
test_data = all_data[:n_test]
train_data = all_data[n_test:]

print(f"\n  train: {len(train_data)}, test: {len(test_data)}")

# 存为 parquet
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_df.to_parquet(os.path.join(OUT_DIR, "train.parquet"), index=False)
test_df.to_parquet(os.path.join(OUT_DIR, "test.parquet"), index=False)

print(f"\n✓ 保存到 {OUT_DIR}/")
print(f"  train.parquet: {len(train_df)} 条")
print(f"  test.parquet:  {len(test_df)} 条")

# ──────────────────────────────────────────────────────────────
# 格式验证
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("格式验证")
print("=" * 60)

think_count = 0
no_think_count = 0
think_cats = Counter()
no_think_cats = Counter()
token_lengths = []

for _, row in train_df.iterrows():
    msgs = row["messages"]
    cat = row["category"]
    total_tokens = sum(estimate_tokens(m["content"]) for m in msgs)
    token_lengths.append(total_tokens)

    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    if all(has_think_tags(m["content"]) for m in assistant_msgs):
        think_count += 1
        think_cats[cat] += 1
    else:
        no_think_count += 1
        no_think_cats[cat] += 1

total = think_count + no_think_count
print(f"  Thinking 模式 (有 <think>): {think_count} ({100 * think_count / total:.1f}%)")
print(f"  Non-thinking 模式 (无 <think>): {no_think_count} ({100 * no_think_count / total:.1f}%)")

print(f"\n  按类别 thinking 分布:")
for cat in ["math_competition", "science_reasoning", "complex_logic", "code_algorithm"]:
    t = think_cats.get(cat, 0)
    nt = no_think_cats.get(cat, 0)
    print(f"    {cat}: thinking={t}, non-thinking={nt}")

# 格式正确性校验
print(f"\n  格式规则检查:")
errors = []
if no_think_cats.get("math_competition", 0) > 0:
    errors.append(f"  ⚠ math_competition 有 {no_think_cats['math_competition']} 条缺少 <think>!")
for cat in ["complex_logic", "code_algorithm"]:
    if think_cats.get(cat, 0) > 0:
        errors.append(f"  ⚠ {cat} 有 {think_cats[cat]} 条意外包含 <think>!")
if errors:
    for e in errors:
        print(e)
else:
    print("  ✓ 数学有 <think>, 逻辑/代码无 <think>")
# 科学类: thinking 与否取决于问题复杂度, 不做强制要求
sci_t = think_cats.get("science_reasoning", 0)
sci_nt = no_think_cats.get("science_reasoning", 0)
print(f"  ℹ science_reasoning: {sci_t} 条有 <think> (复杂推理), {sci_nt} 条直接回答 (简单事实)")

# Token 长度统计
token_arr = np.array(token_lengths)
print(f"\n  Token 长度统计 (估算):")
print(f"    min={token_arr.min()}, max={token_arr.max()}, "
      f"mean={token_arr.mean():.0f}, median={np.median(token_arr):.0f}")
print(f"    >4K: {(token_arr > 4000).sum()}, >6K: {(token_arr > 6000).sum()}, "
      f">7K: {(token_arr > 7000).sum()}, >8K: {(token_arr > 8000).sum()}")

print("\n✓ 构建完成!")
