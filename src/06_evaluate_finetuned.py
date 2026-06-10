"""
06_evaluate_finetuned.py
─────────────────────────
评测微调后的 Qwen2.5-1.5B 模型，与三个 baseline 对比。

使用方式：
  # 从本地 checkpoint 加载（QLoRA adapter 目录）
  python src/06_evaluate_finetuned.py --model-path outputs/qwen-asap-qlora/checkpoint-xxx

  # 从 HuggingFace Hub 加载
  python src/06_evaluate_finetuned.py --hub-repo your-username/qwen2.5-1.5b-asap-qlora

注意：
  - 微调模型的任务是预测 problem_type / action_priority / operator_action 三个字段
  - 基线模型预测全部 6 个字段（含 sentiment / rating / aspect_sentiments）
  - 对比表中，微调模型的 sentiment_f1 / rating_mae / aspect_f1 标记为 N/A（任务不同）
  - JSON Validity 和 operational 字段准确率是主要评测维度
"""

import argparse
import json
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import ROOT, ASAP_PROCESSED_DIR

warnings.filterwarnings("ignore")

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

TEST_SAMPLE_SIZE = 200
RANDOM_SEED = 42

# ── 合法值（与 03_run_baselines.py 保持一致）────────────────────────────────────
VALID_SENTIMENTS = ["positive", "neutral", "negative"]
VALID_ASPECTS = [
    "location_traffic", "location_distance", "location_easy_to_find",
    "service_wait_time", "service_attitude", "service_parking", "service_speed",
    "price_level", "price_value", "price_discount",
    "env_decoration", "env_noise", "env_space", "env_cleanliness",
    "food_portion", "food_taste", "food_appearance", "food_recommendation",
]
VALID_PROBLEM_TYPE = [
    "taste_issue", "poor_service", "long_wait", "overpriced",
    "hygiene_issue", "location_issue", "packaging_issue", "none"
]
VALID_ACTION_PRIORITY = ["low", "medium", "high"]
VALID_OPERATOR_ACTION = [
    "improve_taste", "train_service", "reduce_wait",
    "review_pricing", "fix_hygiene", "no_action"
]

# ── System Prompt（与 02_label_asap.py / 05_prepare_finetune_data.py 一致）──────
ASAP_SYSTEM = """你是一个餐厅经营分析助手。根据评论内容和已知的 aspect 情感标签，补充以下 3 个字段：

- problem_type: 最主要的问题类型（只选一个）：
    taste_issue（口味问题）/ poor_service（服务差）/ long_wait（等待太久）/
    overpriced（价格偏高）/ hygiene_issue（卫生问题）/ location_issue（位置不便）/
    packaging_issue（包装问题）/ none（无明显问题）

- action_priority: 商家处理紧迫程度 (low / medium / high)

- operator_action: 商家最应该做的一件事（只选一个）：
    improve_taste / train_service / reduce_wait /
    review_pricing / fix_hygiene / no_action

规则：
1. 只输出 JSON，不要其他文字
2. 正面评论（无投诉）→ problem_type=none，action_priority=low，operator_action=no_action
3. 只选上面列出的合法值"""


# ── 数据加载（与 03_run_baselines.py 相同逻辑）──────────────────────────────────
def load_test_sample(n: int) -> list[dict]:
    import random
    random.seed(RANDOM_SEED)
    test_path = ASAP_PROCESSED_DIR / "test.jsonl"
    records = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    buckets = defaultdict(list)
    for r in records:
        buckets[r["label"]["rating_prediction"]].append(r)
    sampled = []
    per_star = n // 5
    for star in range(1, 6):
        pool = buckets[star]
        sampled.extend(random.sample(pool, min(per_star, len(pool))))
    random.shuffle(sampled)
    return sampled[:n]


# ── compute_metrics（与 03_run_baselines.py 完全相同）───────────────────────────
def star_to_sentiment(star: float) -> str:
    if star >= 4.0:
        return "positive"
    elif star == 3.0:
        return "neutral"
    else:
        return "negative"


def compute_metrics(predictions: list[dict], gold: list[dict]) -> dict:
    sentiment_gold, sentiment_pred = [], []
    rating_errors = []
    aspect_tp, aspect_fp, aspect_fn = 0, 0, 0
    json_valid = 0

    for pred, g in zip(predictions, gold):
        if pred.get("_valid_json", True):
            json_valid += 1

        g_sent = star_to_sentiment(g["star"])
        p_sent = pred.get("sentiment", "neutral")
        if p_sent not in VALID_SENTIMENTS:
            p_sent = "neutral"
        sentiment_gold.append(g_sent)
        sentiment_pred.append(p_sent)

        p_rating = pred.get("rating_prediction", 3)
        try:
            p_rating = int(p_rating)
            p_rating = max(1, min(5, p_rating))
        except Exception:
            p_rating = 3
        rating_errors.append(abs(p_rating - g["star"]))

        g_aspects = set(g["label"]["aspect_sentiments"].keys())
        raw_aspects = pred.get("aspect_sentiments", {})
        if isinstance(raw_aspects, dict):
            p_aspects = set(raw_aspects.keys())
        elif isinstance(raw_aspects, list):
            p_aspects = set(raw_aspects)
        else:
            p_aspects = set()
        p_aspects = {a for a in p_aspects if a in VALID_ASPECTS}
        aspect_tp += len(g_aspects & p_aspects)
        aspect_fp += len(p_aspects - g_aspects)
        aspect_fn += len(g_aspects - p_aspects)

    sentiment_f1 = f1_score(sentiment_gold, sentiment_pred,
                             labels=VALID_SENTIMENTS, average="macro",
                             zero_division=0)
    rating_mae = float(np.mean(rating_errors))
    precision = aspect_tp / (aspect_tp + aspect_fp + 1e-9)
    recall = aspect_tp / (aspect_tp + aspect_fn + 1e-9)
    aspect_f1 = 2 * precision * recall / (precision + recall + 1e-9)
    validity = json_valid / len(predictions) if predictions else 0

    return {
        "sentiment_f1": round(sentiment_f1, 3),
        "rating_mae": round(rating_mae, 3),
        "aspect_f1": round(aspect_f1, 3),
        "json_validity": round(validity, 3),
        "n": len(predictions),
    }


# ── 微调模型专属：operational 字段准确率 ────────────────────────────────────────
def compute_operational_metrics(predictions: list[dict], gold_labels: list[dict]) -> dict:
    """
    计算 problem_type / action_priority / operator_action 准确率。
    gold_labels 来自 data/finetune/test.jsonl（有完整标注）。
    """
    pt_gold, pt_pred = [], []
    ap_gold, ap_pred = [], []
    oa_gold, oa_pred = [], []

    for pred, g in zip(predictions, gold_labels):
        label = extract_label(g)
        if not label.get("problem_type"):
            continue

        pt_gold.append(label["problem_type"])
        pt_pred.append(pred.get("problem_type", "none")
                       if pred.get("problem_type") in VALID_PROBLEM_TYPE else "none")

        ap_gold.append(label["action_priority"])
        ap_pred.append(pred.get("action_priority", "low")
                       if pred.get("action_priority") in VALID_ACTION_PRIORITY else "low")

        oa_gold.append(label["operator_action"])
        oa_pred.append(pred.get("operator_action", "no_action")
                       if pred.get("operator_action") in VALID_OPERATOR_ACTION else "no_action")

    if not pt_gold:
        return {}

    return {
        "problem_type_acc": round(
            sum(a == b for a, b in zip(pt_gold, pt_pred)) / len(pt_gold), 3),
        "action_priority_acc": round(
            sum(a == b for a, b in zip(ap_gold, ap_pred)) / len(ap_gold), 3),
        "operator_action_acc": round(
            sum(a == b for a, b in zip(oa_gold, oa_pred)) / len(oa_gold), 3),
        "n_operational": len(pt_gold),
    }


# ── 模型加载 ────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_path: Optional[str], hub_repo: Optional[str]):
    """
    支持两种加载方式：
      1. 本地 QLoRA adapter 目录（peft + 基座模型）
      2. HuggingFace Hub 仓库（adapter 或 merged 模型）
    优先使用 unsloth 加速；如未安装则回退到 transformers + peft。
    """
    source = model_path or hub_repo

    # 尝试用 unsloth 加载（速度更快）
    try:
        from unsloth import FastLanguageModel
        print(f"使用 unsloth 加载模型：{source}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=source,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer, "unsloth"
    except ImportError:
        pass

    # 回退：transformers + peft
    print(f"使用 transformers + peft 加载模型：{source}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        source if hub_repo else BASE_MODEL,
        trust_remote_code=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 如果是 adapter 目录，叠加 LoRA 权重
    try:
        model = PeftModel.from_pretrained(base, source)
        model = model.merge_and_unload()
        print("已合并 LoRA adapter 到基座模型")
    except Exception:
        # source 可能已经是 merged 模型
        model = base

    model.eval()
    return model, tokenizer, "transformers"


# ── 推理 ────────────────────────────────────────────────────────────────────────
def extract_label(record: dict) -> dict:
    """兼容两种格式：原始 ASAP 格式（有 label 键）和 finetune 格式（messages 列表）"""
    if "label" in record:
        return record["label"]
    try:
        return json.loads(record["messages"][2]["content"])
    except Exception:
        return {}


def build_user_message(record: dict) -> str:
    label = extract_label(record)
    aspect_str = json.dumps(label.get("aspect_sentiments", {}), ensure_ascii=False)
    if "text" in record:
        text = record["text"][:300]
    else:
        user_content = record["messages"][1]["content"]
        m = re.search(r'评论：(.+?)\n\n输出 JSON', user_content, re.DOTALL)
        text = m.group(1)[:300] if m else user_content[:300]
    return f"""评论：{text}

已知 aspect 情感：{aspect_str}

输出 JSON（仅 3 个字段）："""


def parse_output(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


def run_inference(model, tokenizer, record: dict, backend: str) -> dict:
    messages = [
        {"role": "system", "content": ASAP_SYSTEM},
        {"role": "user", "content": build_user_message(record)},
    ]

    start = time.time()

    if backend == "unsloth":
        from unsloth import FastLanguageModel
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            output_ids[0][inputs.shape[1]:], skip_special_tokens=True
        )
    else:
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            output_ids[0][inputs.shape[1]:], skip_special_tokens=True
        )

    latency_ms = round((time.time() - start) * 1000)

    parsed = parse_output(raw)
    if parsed is not None:
        parsed["_valid_json"] = True
        parsed["_latency_ms"] = latency_ms
        parsed["_raw"] = raw
    else:
        parsed = {
            "_valid_json": False,
            "_latency_ms": latency_ms,
            "_raw": raw,
        }
    return parsed


# ── 成本回收点计算 ────────────────────────────────────────────────────────────
def compute_breakeven(
    training_cost_usd: float,
    api_cost_per_query_usd: float,
    local_cost_per_query_usd: float,
) -> dict:
    """
    回收点 = 训练总成本 / (每次 API 调用节省费用)
    
    默认参数估算（可通过命令行覆盖）：
      training_cost  : Colab T4 × 3h × $0.35/h ≈ $1.05，或 A100 × 3h × $2.5/h ≈ $7.5
      api_cost       : DeepSeek-V3 ~$0.001/query（~300 tokens input + 50 tokens output）
      local_cost     : 本地 GPU 推理电费 ≈ $0.00005/query（可忽略）
    """
    saving_per_query = api_cost_per_query_usd - local_cost_per_query_usd
    if saving_per_query <= 0:
        return {
            "error": "本地推理成本不低于 API 成本，无法计算回收点",
            "training_cost_usd": training_cost_usd,
            "api_cost_per_query_usd": api_cost_per_query_usd,
            "local_cost_per_query_usd": local_cost_per_query_usd,
        }
    breakeven_queries = training_cost_usd / saving_per_query
    return {
        "training_cost_usd": training_cost_usd,
        "api_cost_per_query_usd": api_cost_per_query_usd,
        "local_cost_per_query_usd": local_cost_per_query_usd,
        "saving_per_query_usd": round(saving_per_query, 6),
        "breakeven_queries": round(breakeven_queries),
        "breakeven_note": (
            f"训练成本 ${training_cost_usd:.2f}，每次节省 ${saving_per_query:.6f}，"
            f"需要处理 {round(breakeven_queries):,} 条查询后回收成本"
        ),
    }


# ── 打印对比表 ────────────────────────────────────────────────────────────────
def print_comparison_table(all_metrics: dict) -> None:
    methods = ["textblob", "zero_shot", "few_shot", "finetuned"]
    headers = ["TextBlob", "Zero-shot", "Few-shot", "Fine-tuned"]

    metric_labels = {
        "sentiment_f1": "Sentiment F1",
        "rating_mae":   "Rating MAE",
        "aspect_f1":    "Aspect F1",
        "json_validity": "JSON Validity",
        "avg_latency_ms": "Latency (ms)",
    }

    print("\n─── 评测结果对比 ───────────────────────────────────────────────────")
    header_row = f"{'指标':<20}" + "".join(f"{h:>14}" for h in headers)
    print(header_row)
    print("─" * (20 + 14 * len(methods)))

    for key, label in metric_labels.items():
        row = f"{label:<20}"
        for method in methods:
            val = all_metrics.get(method, {}).get(key, "N/A")
            if val == "N/A":
                row += f"{'N/A':>14}"
            else:
                row += f"{str(val):>14}"
        print(row)

    print("─" * (20 + 14 * len(methods)))
    print("注：Fine-tuned 的 sentiment/rating/aspect 为 N/A（模型专门预测 operational 字段）")


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="评测微调后的 Qwen2.5 模型")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--model-path", type=str,
        help="本地模型目录（QLoRA adapter 或 merged 模型）"
    )
    source_group.add_argument(
        "--hub-repo", type=str,
        help="HuggingFace Hub 仓库名，如 username/qwen2.5-asap-qlora"
    )
    parser.add_argument(
        "--training-cost", type=float, default=1.05,
        help="训练总成本（美元），默认 1.05（T4×3h×$0.35）"
    )
    parser.add_argument(
        "--api-cost-per-query", type=float, default=0.001,
        help="API 每次调用成本（美元），默认 0.001（DeepSeek ~350 tokens）"
    )
    parser.add_argument(
        "--local-cost-per-query", type=float, default=0.00005,
        help="本地推理每次成本（美元），默认 0.00005（电费估算）"
    )
    args = parser.parse_args()

    # 加载 test set
    print(f"加载 test set（{TEST_SAMPLE_SIZE} 条，分层采样）...")
    test_data = load_test_sample(TEST_SAMPLE_SIZE)
    print(f"实际加载：{len(test_data)} 条")

    # 加载微调模型
    model, tokenizer, backend = load_model_and_tokenizer(
        args.model_path, args.hub_repo
    )
    print(f"模型加载完成，backend={backend}")

    # 推理
    predictions = []
    latencies = []
    print(f"\n开始推理（{len(test_data)} 条）...")
    for record in tqdm(test_data, desc="Fine-tuned 推理"):
        pred = run_inference(model, tokenizer, record, backend)
        predictions.append(pred)
        latencies.append(pred.get("_latency_ms", 0))

    # 计算 compute_metrics（与 baseline 相同函数）
    # 微调模型只输出 3 个 operational 字段，sentiment/rating/aspect 将为默认值
    # 结果反映 JSON 格式合规率；operational 准确率单独报告
    ft_metrics = compute_metrics(predictions, test_data)
    ft_metrics["avg_latency_ms"] = round(np.mean(latencies))

    # 标注 N/A（让对比表语义正确）
    ft_metrics_display = dict(ft_metrics)
    ft_metrics_display["sentiment_f1"] = "N/A"
    ft_metrics_display["rating_mae"] = "N/A"
    ft_metrics_display["aspect_f1"] = "N/A"

    # 从 finetune test split 评测 operational 字段（如果文件存在）
    operational_metrics = {}
    ft_test_path = ROOT / "data" / "finetune" / "test.jsonl"
    if ft_test_path.exists():
        ft_test_records = []
        with open(ft_test_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ft_test_records.append(json.loads(line))
        print(f"\n在 finetune test split 上评测 operational 字段（{len(ft_test_records)} 条）...")
        op_predictions = []
        for record in tqdm(ft_test_records[:200], desc="Operational 推理"):
            pred = run_inference(model, tokenizer, record, backend)
            op_predictions.append(pred)
        operational_metrics = compute_operational_metrics(
            op_predictions, [r for r in ft_test_records[:200]]
        )
        if operational_metrics:
            print("\n─── Operational 字段准确率（finetune test split）────────────────")
            for k, v in operational_metrics.items():
                if k != "n_operational":
                    print(f"  {k:<25}: {v}")
            print(f"  评测条数: {operational_metrics.get('n_operational', 0)}")

    # 读取 baseline_results.json
    baseline_path = REPORTS_DIR / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as f:
            baseline_data = json.load(f)
    else:
        baseline_data = {"metrics": {}, "test_size": len(test_data), "predictions": {}}

    # 追加 finetuned 结果
    baseline_data["metrics"]["finetuned"] = ft_metrics
    baseline_data["metrics"]["finetuned"]["operational"] = operational_metrics
    baseline_data["predictions"]["finetuned"] = predictions

    # 打印对比表
    all_metrics_for_display = dict(baseline_data["metrics"])
    all_metrics_for_display["finetuned"] = ft_metrics_display
    all_metrics_for_display["finetuned"]["avg_latency_ms"] = ft_metrics["avg_latency_ms"]
    print_comparison_table(all_metrics_for_display)

    # 计算成本回收点
    breakeven = compute_breakeven(
        training_cost_usd=args.training_cost,
        api_cost_per_query_usd=args.api_cost_per_query,
        local_cost_per_query_usd=args.local_cost_per_query,
    )
    baseline_data["cost_analysis"] = breakeven

    print("\n─── 微调成本回收分析 ──────────────────────────────────────────────")
    for k, v in breakeven.items():
        print(f"  {k:<30}: {v}")

    # 保存
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已追加到：{baseline_path}（finetuned key）")


if __name__ == "__main__":
    main()
