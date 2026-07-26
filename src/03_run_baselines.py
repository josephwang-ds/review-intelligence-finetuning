"""
03_run_baselines.py
────────────────────
在 ASAP test set 上运行三个 baseline，输出对比结果。

运行：python src/03_run_baselines.py
费用：约 200 条 × zero-shot + few-shot ≈ $0.05
耗时：约 10 分钟
"""

import json
import time
from pathlib import Path
from collections import defaultdict

from textblob import TextBlob
from openai import OpenAI
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import ROOT, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, ASAP_PROCESSED_DIR
from schema import ASAP_PROFILE
from prompts import FEW_SHOT_BANK
from structured_client import call_structured

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

TEST_SAMPLE_SIZE = 200   # 从 test set 取多少条评测
RANDOM_SEED = 42

VALID_SENTIMENTS = ASAP_PROFILE.sentiments
VALID_ASPECTS = ASAP_PROFILE.aspects


# ── TextBlob Baseline ─────────────────────────────────────────────────────────
def is_chinese(text: str) -> bool:
    return any('一' <= c <= '鿿' for c in text)


def textblob_predict(text: str) -> dict:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    rating = max(1, min(5, round(3 + polarity * 2)))

    # 简单关键词 aspect 匹配（英文）
    kw_map = {
        "food_taste":       ["food", "taste", "delicious", "bland", "flavor", "yummy"],
        "service_attitude": ["service", "staff", "waiter", "rude", "friendly", "attentive"],
        "price_level":      ["price", "expensive", "cheap", "value", "worth", "overpriced"],
        "env_decoration":   ["ambiance", "atmosphere", "decor", "environment", "cozy"],
        "service_wait_time":["wait", "slow", "fast", "quickly", "forever"],
    }
    aspects = {}
    text_lower = text.lower()
    for aspect, keywords in kw_map.items():
        if any(kw in text_lower for kw in keywords):
            aspects[aspect] = sentiment

    return {
        "sentiment": sentiment,
        "rating_prediction": rating,
        "aspect_sentiments": aspects if aspects else {},
        "problem_type": "none" if polarity >= 0 else "taste_issue",
        "action_priority": "low" if polarity >= 0 else "high",
        "operator_action": "no_action" if polarity >= 0 else "improve_taste",
        "_polarity": round(polarity, 3),
        "_chinese_warning": is_chinese(text),
    }


# ── LLM Baseline（schema 校验 + 修复重试，见 structured_client.py）───────────────
def llm_predict(client: OpenAI, text: str, mode: str = "zero") -> dict:
    few_shot = FEW_SHOT_BANK["asap"] if mode == "few" else None
    result, meta = call_structured(
        client, ASAP_PROFILE, text, mode="full", few_shot=few_shot,
        model=DEEPSEEK_MODEL,  # max_tokens 用 structured_client 的默认值（reasoning 模型需要余量）
    )
    if result is not None:
        out = result.model_dump()
        out["_latency_ms"] = meta["latency_ms"]
        out["_valid_json"] = True
        out["_repaired"] = meta["repaired"]
        return out
    return {"_valid_json": False, "_latency_ms": meta["latency_ms"], "_error": meta["error"]}


# ── 评测函数 ──────────────────────────────────────────────────────────────────
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

        # Sentiment F1
        g_sent = star_to_sentiment(g["star"])
        p_sent = pred.get("sentiment", "neutral")
        if p_sent not in VALID_SENTIMENTS:
            p_sent = "neutral"
        sentiment_gold.append(g_sent)
        sentiment_pred.append(p_sent)

        # Rating MAE
        p_rating = pred.get("rating_prediction", 3)
        try:
            p_rating = int(p_rating)
            p_rating = max(1, min(5, p_rating))
        except Exception:
            p_rating = 3
        rating_errors.append(abs(p_rating - g["star"]))

        # Aspect F1
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

    # Compute
    sentiment_f1 = f1_score(sentiment_gold, sentiment_pred,
                             labels=list(VALID_SENTIMENTS), average="macro",
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


# ── 主流程 ────────────────────────────────────────────────────────────────────
def load_test_sample(n: int) -> list[dict]:
    import random
    random.seed(RANDOM_SEED)
    test_path = ASAP_PROCESSED_DIR / "test.jsonl"
    records = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    # 分层采样
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


def main():
    print(f"加载 test set（{TEST_SAMPLE_SIZE} 条）...")
    test_data = load_test_sample(TEST_SAMPLE_SIZE)
    print(f"实际加载：{len(test_data)} 条")

    results = {
        "textblob": {"predictions": [], "latencies": []},
        "zero_shot": {"predictions": [], "latencies": []},
        "few_shot": {"predictions": [], "latencies": []},
    }

    client = None
    if DEEPSEEK_API_KEY:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    else:
        print("⚠ 未设置 DEEPSEEK_API_KEY，只运行 TextBlob baseline")

    # ── TextBlob ──
    print("\n[1/3] TextBlob baseline...")
    for record in tqdm(test_data):
        start = time.time()
        pred = textblob_predict(record["text"])
        pred["_latency_ms"] = round((time.time() - start) * 1000)
        pred["_valid_json"] = True
        results["textblob"]["predictions"].append(pred)
        results["textblob"]["latencies"].append(pred["_latency_ms"])

    # ── Zero-shot ──
    if client:
        print("\n[2/3] Zero-shot LLM...")
        for record in tqdm(test_data):
            pred = llm_predict(client, record["text"], mode="zero")
            results["zero_shot"]["predictions"].append(pred)
            results["zero_shot"]["latencies"].append(pred.get("_latency_ms", 0))
            time.sleep(0.2)

        # ── Few-shot ──
        print("\n[3/3] Few-shot LLM...")
        for record in tqdm(test_data):
            pred = llm_predict(client, record["text"], mode="few")
            results["few_shot"]["predictions"].append(pred)
            results["few_shot"]["latencies"].append(pred.get("_latency_ms", 0))
            time.sleep(0.2)

    # ── 计算指标 ──
    print("\n\n─── 评测结果 ───────────────────────────────────────────")
    print(f"{'指标':<20} {'TextBlob':>12} {'Zero-shot':>12} {'Few-shot':>12}")
    print("─" * 60)

    all_metrics = {}
    for method in ["textblob", "zero_shot", "few_shot"]:
        if not results[method]["predictions"]:
            continue
        metrics = compute_metrics(results[method]["predictions"], test_data)
        avg_latency = round(np.mean(results[method]["latencies"]))
        metrics["avg_latency_ms"] = avg_latency
        all_metrics[method] = metrics

    metric_labels = {
        "sentiment_f1": "Sentiment F1",
        "rating_mae": "Rating MAE",
        "aspect_f1": "Aspect F1",
        "json_validity": "JSON Validity",
        "avg_latency_ms": "Latency (ms)",
    }

    for key, label in metric_labels.items():
        row = f"{label:<20}"
        for method in ["textblob", "zero_shot", "few_shot"]:
            val = all_metrics.get(method, {}).get(key, "N/A")
            row += f"{str(val):>12}"
        print(row)

    print("─" * 60)
    print("TextBlob Sentiment F1 低是预期结果（不支持中文）")

    # ── 保存 ──
    # 合并而不是覆盖：06_evaluate_finetuned.py 会往同一个文件里写 finetuned 指标和
    # cost_analysis，直接整体覆盖会把那部分结果抹掉（重跑 baseline 不应该让微调模型的
    # 评测结果消失）。
    out_path = REPORTS_DIR / "baseline_results.json"
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            output = json.load(f)
    else:
        output = {}

    output.setdefault("metrics", {}).update(all_metrics)
    output["test_size"] = len(test_data)
    output.setdefault("predictions", {}).update({
        method: results[method]["predictions"]
        for method in results
        if results[method]["predictions"]
    })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n已保存：{out_path}（保留了已有的 finetuned / cost_analysis 结果）")


if __name__ == "__main__":
    main()
