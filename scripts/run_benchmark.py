#!/usr/bin/env python3
"""
6-Dataset Benchmark Evaluation for Qwen3 model.

Datasets: aime2024, aime2025, amc2023, gpqa_diamond, gsm8k, math_lighteval
Backend : vLLM (tensor-parallel across GPUs)
Scoring（仅 2 种，均与 verl 标准一致）:
  - math_verify：5 个数学/推理集（aime2024, aime2025, amc2023, gsm8k, math_lighteval），
    规则为「抽答案 + 数学等价判定」（verl.utils.reward_score.math_verify）
  - GPQA：仅 gpqa_diamond，使用 verl.utils.reward_score.default_compute_score
    （末 600 字内优先 Answer : X / \\boxed{X}，否则在 Answer 后或末 100 字取最后 A-D，与标准答案比）
Logging : wandb (test_score/<dataset> format)

Usage:
    python scripts/run_benchmark.py \
        --model_path Qwen3-4B-Base \
        --data_dir my_data \
        --n_samples 3 \
        --temperature 0.7 \
        --tp_size 4
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

WORK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK_DIR / "verl"))

DATASETS = [
    "aime2024",
    "aime2025",
    "amc2023",
    "gpqa_diamond",
    "gsm8k",
    "math_lighteval",
]

# 逻辑名 -> 磁盘目录名（test.parquet 所在目录）
DATASET_DIR_MAP = {
    "aime2024": "aime_2024",
    "aime2025": "aime_2025",
    "amc2023": "amc23",
    "gpqa_diamond": "GPQA-Diamond",
    "gsm8k": "gsm8k",
    "math_lighteval": "MATH-lighteval",
}

PROMPT_INSTRUCTION = {
    "gpqa_diamond": (
        "\n\nPlease reason step by step, and put your final answer "
        "(A, B, C, or D) within \\boxed{}."
    ),
}
DEFAULT_INSTRUCTION = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)

# 5 个数学/推理集用 math_verify，1 个用 GPQA 选项匹配
MATH_VERIFY_DATASETS = {"aime2024", "aime2025", "amc2023", "gsm8k", "math_lighteval"}
GPQA_DATASET = "gpqa_diamond"

# aime_2024, aime_2025, amc23 等：用 startswith("aime") 或 in ["amc23", ...] 判定为“数学题通用”分支



# ═══════════════════════════════════════════════════════════════════
#  Ground-Truth Extraction
# ═══════════════════════════════════════════════════════════════════

def _last_boxed(s: str):
    idx = s.rfind("\\boxed{")
    if idx < 0:
        return None
    i, depth = idx, 0
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[idx + 7 : i]
        i += 1
    return None


def extract_ground_truth(raw_gt: str, dataset_key: str) -> str:
    """按数据集键从 raw_gt 提取参考答案（dataset_key 为 6 个数据集名之一）。"""
    if dataset_key == "gsm8k":
        m = re.search(r"####\s*(.+)", raw_gt)
        return m.group(1).strip().replace(",", "") if m else raw_gt.strip()
    if dataset_key == "math_lighteval":
        boxed = _last_boxed(raw_gt)
        return boxed if boxed is not None else raw_gt.strip()
    return str(raw_gt).strip()


# ═══════════════════════════════════════════════════════════════════
#  Answer Extraction from Model Response
# ═══════════════════════════════════════════════════════════════════

def extract_answer(response: str, dataset_key: str):
    """从模型输出中抽取答案（math_verify 自己抽答案；GPQA 使用 verl default_compute_score，不经过此函数；此处保留供调试/其他用途）。"""
    tail = response[-800:]

    boxed = _last_boxed(tail)
    if boxed is not None:
        return boxed

    if dataset_key == "gsm8k":
        m = re.findall(r"####\s*(-?[\d,.]+)", tail)
        if m:
            return m[-1].replace(",", "")

    m = re.findall(
        r"(?i)(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([^\n.。,，]+)", tail
    )
    if m:
        return m[-1].strip()

    if dataset_key == GPQA_DATASET:
        m = re.findall(r"\b([A-Da-d])\b", tail[-200:])
        if m:
            return m[-1]

    return None


# ═══════════════════════════════════════════════════════════════════
#  Scoring：仅 2 种 —— math_verify（5 个数学集）+ GPQA 选项匹配（gpqa_diamond）
# ═══════════════════════════════════════════════════════════════════

def score_math_verify(response: str, ground_truth: str) -> dict:
    """5 个数学/推理集：抽答案 + 数学等价判定（math_verify）。"""
    pred = _last_boxed(response[-800:]) if len(response) > 800 else _last_boxed(response)
    try:
        from verl.utils.reward_score import math_verify
        ret = math_verify.compute_score(response, ground_truth)
        correct = float(ret) > 0
        return {"correct": correct, "pred": pred, "gt": ground_truth}
    except Exception as e:
        print(f"  [WARN] math_verify failed: {e}")
        return {"correct": False, "pred": pred, "gt": ground_truth}


def score_gpqa(response: str, ground_truth: str) -> dict:
    """GPQA 统一走 verl default_compute_score，与 eval 标准一致。"""
    from verl.utils.reward_score import default_compute_score

    res = default_compute_score("gpqa_diamond", response, ground_truth)

    if isinstance(res, dict):
        score = float(res.get("score", res.get("acc", 0)))
        pred = res.get("pred")
    else:
        score = float(res)
        pred = None

    correct = score > 0
    return {"correct": correct, "pred": pred, "gt": ground_truth}


def score_one(response: str, ground_truth: str, dataset_key: str) -> dict:
    """统一入口：gpqa_diamond 走 GPQA 选项匹配，其余 5 个走 math_verify。"""
    if dataset_key == GPQA_DATASET:
        return score_gpqa(response, ground_truth)
    return score_math_verify(response, ground_truth)


# ═══════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════

def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Codex paper, Chen et al. 2021).
    n = total samples, c = correct samples, k = selection size.
    pass@k = 1 - C(n-c, k) / C(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_metrics(scores_per_prompt: list[list[bool]], n: int) -> dict:
    """
    scores_per_prompt: list (num_prompts) of list (n) of bool.
    Returns:
      - pass@1:  标准 pass@1（无偏估计），等价于所有样本的平均正确率
      - maj@N:   每题 N 个回答多数投票后的准确率
      - pass@N:  每题 N 个采样中至少 1 个正确的题目占比（coverage）
    """
    if not scores_per_prompt:
        return {}

    pass_at_1_per_prompt = []
    for row in scores_per_prompt:
        c = sum(row)
        pass_at_1_per_prompt.append(_pass_at_k(n, c, 1))
    pass_1 = float(np.mean(pass_at_1_per_prompt))

    maj_correct = sum(
        1 for row in scores_per_prompt if sum(row) > n / 2
    )
    maj_at_n = maj_correct / len(scores_per_prompt)

    pass_n = float(np.mean([any(row) for row in scores_per_prompt]))

    return {
        "pass@1": pass_1,
        f"maj@{n}": maj_at_n,
        f"pass@{n}": pass_n,
    }


# ═══════════════════════════════════════════════════════════════════
#  Data Loading & Prompt Building
# ═══════════════════════════════════════════════════════════════════

def load_datasets(data_dir: str, max_per_set: int = -1, exclude: set[str] | None = None) -> dict[str, pd.DataFrame]:
    datasets = {}
    for name in DATASETS:
        if exclude and name in exclude:
            print(f"  {name}: excluded")
            continue
        dir_name = DATASET_DIR_MAP.get(name, name)
        path = Path(data_dir) / dir_name / "test.parquet"
        if not path.exists():
            print(f"  [WARN] {path} not found, skipping {name}")
            continue
        df = pd.read_parquet(path)
        if 0 < max_per_set < len(df):
            df = df.head(max_per_set)
        datasets[name] = df
        print(f"  {name}: {len(df)} samples")
    return datasets


def _get_ground_truth_raw(row_reward_model) -> str:
    """从 reward_model 列取出标准答案字符串（支持 dict 与纯字符串）。"""
    rm = row_reward_model
    if isinstance(rm, dict) and "ground_truth" in rm:
        gt = rm["ground_truth"]
        if isinstance(gt, list):
            return str(gt[0]) if gt else ""
        return str(gt)
    return str(rm)


def build_entries(datasets: dict[str, pd.DataFrame]) -> list[dict]:
    entries = []
    for name, df in datasets.items():
        suffix = PROMPT_INSTRUCTION.get(name, DEFAULT_INSTRUCTION)
        for idx in range(len(df)):
            row = df.iloc[idx]
            raw_gt = _get_ground_truth_raw(row["reward_model"])
            ground_truth = extract_ground_truth(raw_gt, name)
            entries.append({
                "dataset": name,
                "index": idx,
                "prompt_text": str(row["prompt"]).strip() + suffix,
                "ground_truth": ground_truth,
            })
    return entries


# ═══════════════════════════════════════════════════════════════════
#  vLLM Generation
# ═══════════════════════════════════════════════════════════════════

def patch_tokenizer():
    """Add missing all_special_tokens_extended for Qwen2Tokenizer."""
    try:
        from transformers import Qwen2Tokenizer
        if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
            Qwen2Tokenizer.all_special_tokens_extended = property(
                lambda self: self.all_special_tokens
            )
    except ImportError:
        pass


def format_prompts(
    prompt_texts: list[str],
    tokenizer,
    enable_thinking: bool,
    no_chat_template: bool = False,
) -> list[str]:
    if no_chat_template:
        return list(prompt_texts)
    formatted = []
    for text in prompt_texts:
        messages = [{"role": "user", "content": text}]
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        formatted.append(out)
    return formatted


def generate_responses(
    formatted_prompts: list[str],
    args,
) -> list[list[dict]]:
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_new_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # max_num_seqs 与 max_new_tokens 折中（KV）：≤4096→256，8192→128，≥16384→64；中间线性插值
    _mnt = max(int(args.max_new_tokens), 1)
    if _mnt >= 16384:
        max_num_seqs = 64
    elif _mnt >= 8192:
        max_num_seqs = 128
    elif _mnt <= 4096:
        max_num_seqs = 256
    else:
        max_num_seqs = int(
            round(256 - (256 - 128) * (_mnt - 4096) / (8192 - 4096))
        )
        max_num_seqs = max(128, min(256, max_num_seqs))
    max_num_seqs = max(8, max_num_seqs)

    llm_kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_util,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
        num_scheduler_steps=args.num_scheduler_steps,
        max_num_seqs=max_num_seqs,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    print(f"Generating {args.n_samples} responses × {len(formatted_prompts)} prompts ...")
    t0 = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Generation done in {elapsed:.1f}s "
          f"({len(formatted_prompts) * args.n_samples / elapsed:.1f} samples/s)")

    all_responses = []
    token_lens = []
    for output in outputs:
        responses = []
        for c in output.outputs:
            token_ids = getattr(c, "token_ids", None)
            new_tokens = len(token_ids) if token_ids is not None else None
            if new_tokens is not None:
                token_lens.append(new_tokens)
            responses.append(
                {
                    "response": c.text,
                    "new_tokens": new_tokens,
                    "finish_reason": getattr(c, "finish_reason", None),
                    "stop_reason": getattr(c, "stop_reason", None),
                }
            )
        all_responses.append(responses)
    if token_lens:
        print(
            "Generated new tokens stats: "
            f"mean={np.mean(token_lens):.1f}, p50={np.percentile(token_lens, 50):.1f}, "
            f"p95={np.percentile(token_lens, 95):.1f}, max={np.max(token_lens)}"
        )
    return all_responses


def summarize_new_tokens(lengths: list[int], max_new_tokens: int) -> dict:
    """Summarize generated token lengths."""
    if not lengths:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
            "hit_max_ratio": None,
        }

    arr = np.asarray(lengths, dtype=np.int32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
        "hit_max_ratio": float(np.mean(arr >= max_new_tokens)),
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="6-Dataset Benchmark for LLM")
    parser.add_argument("--model_path", required=True, help="HF model directory")
    parser.add_argument("--data_dir", default="my_data")
    parser.add_argument("--output_dir", default="outputs/benchmark")
    parser.add_argument("--n_samples", type=int, default=3, help="samples per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tp_size", type=int, default=4, help="tensor parallel size")
    parser.add_argument("--gpu_memory_util", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="不设则用 vLLM/模型默认，整条序列 = prompt + 生成（prompt 随样本变）")
    parser.add_argument("--max_per_set", type=int, default=-1,
                        help="max samples per dataset (-1 = all)")
    parser.add_argument("--wandb_project", default="Qwen3-Benchmark")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--no_chat_template", action="store_true", default=False,
                        help="use raw prompt text, no chat template")
    parser.add_argument("--enforce_eager", action="store_true", default=False,
                        help="skip torch.compile + CUDA graph capture for faster startup")
    parser.add_argument("--num_scheduler_steps", type=int, default=1,
                        help="multi-step scheduling (>1 reduces Python-GPU round trips)")
    parser.add_argument("--exclude_datasets", type=str, default="",
                        help="comma-separated dataset names to skip, e.g. 'gsm8k,math_lighteval'")
    args = parser.parse_args()

    # ── Resolve paths ──
    args.model_path = str(Path(args.model_path).resolve())
    args.data_dir = str(Path(args.data_dir).resolve())

    # 模型目录必须存在且含 config.json，否则会被误当作 Hub repo_id 报错
    model_dir = Path(args.model_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"模型目录不存在: {args.model_path}\n"
            f"请设置正确的 MODEL_PATH，例如项目根下的 Qwen3-4B-SFT:\n"
            f"  MODEL_PATH=/project/home/p201251/hyl_ppo/Qwen3-4B-SFT sbatch ... run_benchmark.sh test"
        )
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"模型目录内缺少 config.json: {args.model_path}\n"
            f"请确认该路径为 HuggingFace 格式的模型目录。"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb_name is None:
        args.wandb_name = Path(args.model_path).name + f"-bench-n{args.n_samples}"

    # ── Banner ──
    print("=" * 54)
    print("  6-Dataset Benchmark Evaluation")
    print("=" * 54)
    print(f"  model_path       : {args.model_path}")
    print(f"  data_dir         : {args.data_dir}")
    print(f"  n_samples        : {args.n_samples}")
    print(f"  max_new_tokens   : {args.max_new_tokens}")
    print(f"  temperature      : {args.temperature}")
    print(f"  enable_thinking  : {args.enable_thinking}")
    print(f"  no_chat_template : {args.no_chat_template}")
    print(f"  tp_size          : {args.tp_size}")
    print(f"  max_model_len    : {args.max_model_len if args.max_model_len is not None else '(not set, prompt + max_new_tokens)'}")
    print(f"  wandb_project    : {args.wandb_project}")
    print(f"  wandb_name       : {args.wandb_name}")
    print(f"  output_dir       : {out_dir}")
    print(f"  started_at       : {datetime.now().isoformat()}")
    print("=" * 54)

    # ── wandb ──
    import wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
    )

    # ── Load data ──
    print("\nLoading datasets ...")
    exclude_set = set(s.strip() for s in args.exclude_datasets.split(",") if s.strip()) if args.exclude_datasets else None
    datasets = load_datasets(args.data_dir, args.max_per_set, exclude=exclude_set)
    if not datasets:
        print("No datasets found. Exiting.")
        return
    entries = build_entries(datasets)
    total_prompts = len(entries)
    print(f"Total prompts: {total_prompts}")

    # ── Format prompts ──
    if args.no_chat_template:
        print("\nUsing raw prompts (no chat template) ...")
    else:
        print("\nApplying chat template ...")
    patch_tokenizer()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    prompt_texts = [e["prompt_text"] for e in entries]
    formatted = format_prompts(
        prompt_texts, tokenizer, args.enable_thinking, args.no_chat_template
    )

    # ── Generate ──
    print()
    all_responses = generate_responses(formatted, args)
    assert len(all_responses) == total_prompts

    # ── Score ──
    print("\nScoring responses ...")
    per_dataset: dict[str, list[list[bool]]] = defaultdict(list)
    detailed_results = []

    all_new_tokens: list[int] = []
    per_dataset_new_tokens: dict[str, list[int]] = defaultdict(list)

    for entry, responses in zip(entries, all_responses):
        sample_correct = []
        sample_details = []
        for sample in responses:
            resp = sample["response"]
            result = score_one(resp, entry["ground_truth"], entry["dataset"])
            sample_correct.append(result["correct"])
            new_tokens = sample.get("new_tokens")
            if isinstance(new_tokens, int):
                all_new_tokens.append(new_tokens)
                per_dataset_new_tokens[entry["dataset"]].append(new_tokens)
            sample_details.append({
                "response": resp,
                "pred": result["pred"],
                "gt": result["gt"],
                "correct": result["correct"],
                "new_tokens": new_tokens,
                "finish_reason": sample.get("finish_reason"),
                "stop_reason": sample.get("stop_reason"),
            })
        per_dataset[entry["dataset"]].append(sample_correct)
        detailed_results.append({
            "dataset": entry["dataset"],
            "index": entry["index"],
            "prompt": entry["prompt_text"],
            "ground_truth": entry["ground_truth"],
            "samples": sample_details,
        })

    # ── Compute & log metrics ──
    n = args.n_samples
    print()
    header = f"{'Dataset':<20} {'Samples':>7}  {'pass@1':>8}  {'maj@'+str(n):>8}  {'pass@'+str(n):>8}"
    sep = "-" * len(header)
    print("=" * len(header))
    print("  6-Dataset Benchmark Results")
    print("=" * len(header))
    print(header)
    print(sep)

    all_per_prompt = []
    summary = {}

    maj_key = f"maj@{n}"
    pass_n_key = f"pass@{n}"

    for ds_name in DATASETS:
        if ds_name not in per_dataset:
            continue
        scores = per_dataset[ds_name]
        metrics = compute_metrics(scores, n)
        n_prompts = len(scores)

        print(
            f"  {ds_name:<18} {n_prompts:>7}  "
            f"{metrics['pass@1']:>7.1%}  "
            f"{metrics[maj_key]:>7.1%}  "
            f"{metrics[pass_n_key]:>7.1%}"
        )

        summary[ds_name] = {"n_prompts": n_prompts, **metrics}

        dataset_token_stats = summarize_new_tokens(per_dataset_new_tokens.get(ds_name, []), args.max_new_tokens)
        if dataset_token_stats["count"] > 0:
            summary[ds_name]["new_tokens"] = dataset_token_stats

        log_payload = {
            f"test_score/{ds_name}": metrics["pass@1"],
            f"test_score/{ds_name}/pass@1": metrics["pass@1"],
            f"test_score/{ds_name}/maj@{n}": metrics[maj_key],
            f"test_score/{ds_name}/pass@{n}": metrics[pass_n_key],
        }
        if dataset_token_stats["count"] > 0:
            log_payload.update({
                f"gen_tokens/{ds_name}/mean": dataset_token_stats["mean"],
                f"gen_tokens/{ds_name}/p50": dataset_token_stats["p50"],
                f"gen_tokens/{ds_name}/p90": dataset_token_stats["p90"],
                f"gen_tokens/{ds_name}/p95": dataset_token_stats["p95"],
                f"gen_tokens/{ds_name}/max": dataset_token_stats["max"],
                f"gen_tokens/{ds_name}/hit_max_ratio": dataset_token_stats["hit_max_ratio"],
            })
        wandb.log(log_payload)

        for row in scores:
            all_per_prompt.append(row)

    # Overall
    overall_metrics = compute_metrics(all_per_prompt, n)
    print(sep)
    print(
        f"  {'Overall':<18} {len(all_per_prompt):>7}  "
        f"{overall_metrics['pass@1']:>7.1%}  "
        f"{overall_metrics[maj_key]:>7.1%}  "
        f"{overall_metrics[pass_n_key]:>7.1%}"
    )
    print("=" * len(header))
    print(
        f"\n  Metric definitions (N={n}):\n"
        f"    pass@1  : unbiased estimator of P(correct) when picking 1 sample (= avg accuracy)\n"
        f"    maj@{n:<3d}: fraction of prompts correct by majority vote among {n} samples\n"
        f"    pass@{n:<3d}: fraction of prompts with at least 1 correct among {n} samples (coverage)\n"
    )

    summary["overall"] = {"n_prompts": len(all_per_prompt), **overall_metrics}
    overall_token_stats = summarize_new_tokens(all_new_tokens, args.max_new_tokens)
    if overall_token_stats["count"] > 0:
        summary["overall"]["new_tokens"] = overall_token_stats

    overall_log_payload = {
        "test_score/overall": overall_metrics["pass@1"],
        f"test_score/overall/pass@1": overall_metrics["pass@1"],
        f"test_score/overall/maj@{n}": overall_metrics[maj_key],
        f"test_score/overall/pass@{n}": overall_metrics[pass_n_key],
    }
    if overall_token_stats["count"] > 0:
        print(
            "Overall new tokens stats: "
            f"mean={overall_token_stats['mean']:.1f}, p50={overall_token_stats['p50']:.1f}, "
            f"p95={overall_token_stats['p95']:.1f}, max={overall_token_stats['max']}, "
            f"hit_max_ratio={overall_token_stats['hit_max_ratio']:.1%}"
        )
        overall_log_payload.update({
            "gen_tokens/overall/mean": overall_token_stats["mean"],
            "gen_tokens/overall/p50": overall_token_stats["p50"],
            "gen_tokens/overall/p90": overall_token_stats["p90"],
            "gen_tokens/overall/p95": overall_token_stats["p95"],
            "gen_tokens/overall/max": overall_token_stats["max"],
            "gen_tokens/overall/hit_max_ratio": overall_token_stats["hit_max_ratio"],
        })
    wandb.log(overall_log_payload)

    # ── Save results ──
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")

    details_path = out_dir / "detailed_results.json"
    with open(details_path, "w") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {details_path}")

    args_path = out_dir / "args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ── WandB 表格：每个 case 一行，完整记录 prompt / response / pred / ground_truth / score ──
    table_rows = []
    for rec in detailed_results:
        dataset = rec["dataset"]
        prompt_idx = rec["index"]
        for sample_idx, s in enumerate(rec["samples"]):
            table_rows.append([
                dataset,
                prompt_idx,
                sample_idx + 1,
                rec["prompt"],
                s["response"],
                str(s["pred"]) if s["pred"] is not None else "",
                rec["ground_truth"],
                float(s["correct"]),
                "✓" if s["correct"] else "✗",
                s["new_tokens"] if s["new_tokens"] is not None else -1,
                s["finish_reason"] if s["finish_reason"] is not None else "",
                s["stop_reason"] if s["stop_reason"] is not None else "",
            ])
    if table_rows:
        case_table = wandb.Table(
            columns=[
                "dataset", "case_id", "sample_id",
                "prompt", "response", "pred", "ground_truth",
                "score", "score_label", "new_tokens", "finish_reason", "stop_reason",
            ],
            data=table_rows,
        )
        wandb.log({"benchmark/cases": case_table})
        art = wandb.Artifact(name="benchmark_detailed_results", type="benchmark_results")
        art.add_file(str(details_path), name="detailed_results.json")
        wandb.log_artifact(art)
        print("WandB: logged benchmark/cases table and artifact benchmark_detailed_results")

    wandb.finish()
    print(f"\nBenchmark completed at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
