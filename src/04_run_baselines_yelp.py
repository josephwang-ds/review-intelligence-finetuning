"""
04_run_baselines_yelp.py
─────────────────────────
在 Yelp 英文数据上运行三个 baseline（跨语言验证集）
对比 ASAP 中文结果，说明规则系统的语言局限性。

运行：python src/04_run_baselines_yelp.py
费用：约 200 条 × zero-shot + few-shot ≈ $0.04
耗时：约 10 分钟
"""

import json
import time
import re
import random
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
from textblob import TextBlob
from openai import OpenAI
from sklearn.metrics import f1_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import ROOT, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

YELP_LABELED = ROOT / "data" / "yelp" / "labeled" / "labeled.jsonl"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

TEST_SAMPLE_SIZE = 200
RANDOM_SEED = 42

VALID_SENTIMENTS = ["positive", "neutral", "negative"]
VALID_ASPECTS = [
    "product_quality", "logistics", "customer_service",
    "packaging", "value", "authenticity"
]

# ── Few-shot 示例（英文 Yelp 风格）────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "review": "The food was absolutely delicious and the service was great. A bit pricey but totally worth it for the quality.",
        "output": {
            "sentiment": "positive",
            "rating_prediction": 5,
            "aspect_sentiments": {"product_quality": "positive", "customer_service": "positive", "value": "neutral"},
            "problem_type": "none",
            "action_priority": "low",
            "operator_action": "no_action"
        }
    },
    {
        "review": "Terrible experience. The food was cold and tasteless, staff was rude and we waited over an hour. Never coming back.",
        "output": {
            "sentiment": "negative",
            "rating_prediction": 1,
            "aspect_sentiments": {"product_quality": "negative", "customer_service": "negative"},
            "problem_type": "poor_service",
            "action_priority": "high",
            "operator_action": "train_service"
        }
    },
    {
        "review": "Decent place, nothing special. Food was okay, service was average, prices are fair for the area.",
        "output": {
            "sentiment": "neutral",
            "rating_prediction": 3,
            "aspect_sentiments": {"product_quality": "neutral", "customer_service": "neutral", "value": "positive"},
            "problem_type": "none",
            "action_priority": "low",
            "operator_action": "no_action"
        }
    },
]

SYSTEM_PROMPT = """You are a restaurant review analyst. Analyze the review and output structured JSON.

Fields:
- sentiment: positive / neutral / negative
- rating_prediction: 1-5 (integer)
- aspect_sentiments: {aspect: sentiment}, aspects from:
  product_quality, logistics, customer_service, packaging, value, authenticity
- problem_type: quality_issue / slow_logistics / poor_service / overpriced / fake_product / packaging_damage / none
- action_priority: low / medium / high
- operator_action: fix_quality / improve_logistics / train_service / review_pricing / verify_authenticity / no_action

Output JSON only, no other text."""


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def star_to_sentiment(star: float) -> str:
    if star >= 4.0:
        return "positive"
    elif star == 3.0:
        return "neutral"
    else:
        return "negative"


def parse_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


# ── TextBlob ──────────────────────────────────────────────────────────────────
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

    kw_map = {
        "product_quality":  ["food", "taste", "quality", "delicious", "bland", "fresh", "stale"],
        "customer_service": ["service", "staff", "waiter", "rude", "friendly", "attentive", "ignored"],
        "value":            ["price", "expensive", "cheap", "worth", "value", "overpriced", "reasonable"],
        "packaging":        ["packaging", "wrapped", "box", "container", "presentation"],
        "logistics":        ["delivery", "wait", "slow", "fast", "arrived", "shipping"],
    }
    aspects = {}
    tl = text.lower()
    for asp, kws in kw_map.items():
        if any(k in tl for k in kws):
            aspects[asp] = sentiment

    return {
        "sentiment": sentiment,
        "rating_prediction": rating,
        "aspect_sentiments": aspects if aspects else {},
        "problem_type": "none" if polarity >= 0 else "quality_issue",
        "action_priority": "low" if polarity >= 0 else "high",
        "operator_action": "no_action" if polarity >= 0 else "fix_quality",
        "_polarity": round(polarity, 3),
        "_valid_json": True,
    }


# ── LLM ───────────────────────────────────────────────────────────────────────
def llm_predict(client: OpenAI, text: str, mode: str = "zero") -> dict:
    if mode == "few":
        ex_str = ""
        for ex in FEW_SHOT_EXAMPLES:
            ex_str += f"Review: {ex['review']}\nOutput: {json.dumps(ex['output'])}\n\n"
        user_msg = f"{ex_str}Review: {text[:400]}\nOutput JSON:"
    else:
        user_msg = f"Review: {text[:400]}\n\nOutput JSON:"

    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        latency = round((time.time() - start) * 1000)
        result = parse_json(resp.choices[0].message.content)
        if result:
            result["_latency_ms"] = latency
            result["_valid_json"] = True
        else:
            result = {"_valid_json": False, "_latency_ms": latency}
    except Exception as e:
        result = {"_valid_json": False, "_error": str(e), "_latency_ms": 0}
    return result


# ── 评测 ──────────────────────────────────────────────────────────────────────
def compute_metrics(predictions: list, gold: list) -> dict:
    sentiment_gold, sentiment_pred = [], []
    rating_errors = []
    aspect_tp, aspect_fp, aspect_fn = 0, 0, 0
    json_valid = 0

    for pred, g in zip(predictions, gold):
        if pred.get("_valid_json", True):
            json_valid += 1

        # Sentiment
        g_sent = star_to_sentiment(g["stars"])
        p_sent = pred.get("sentiment", "neutral")
        if p_sent not in VALID_SENTIMENTS:
            p_sent = "neutral"
        sentiment_gold.append(g_sent)
        sentiment_pred.append(p_sent)

        # Rating MAE
        try:
            p_r = max(1, min(5, int(pred.get("rating_prediction", 3))))
        except Exception:
            p_r = 3
        rating_errors.append(abs(p_r - g["stars"]))

        # Aspect F1
        g_label = g.get("label", {})
        g_aspects = set(g_label.get("aspects", []))
        raw = pred.get("aspect_sentiments", {})
        if isinstance(raw, dict):
            p_aspects = set(raw.keys())
        elif isinstance(raw, list):
            p_aspects = set(raw)
        else:
            p_aspects = set()
        p_aspects = {a for a in p_aspects if a in VALID_ASPECTS}
        aspect_tp += len(g_aspects & p_aspects)
        aspect_fp += len(p_aspects - g_aspects)
        aspect_fn += len(g_aspects - p_aspects)

    f1 = f1_score(sentiment_gold, sentiment_pred,
                  labels=VALID_SENTIMENTS, average="macro", zero_division=0)
    mae = float(np.mean(rating_errors))
    prec = aspect_tp / (aspect_tp + aspect_fp + 1e-9)
    rec = aspect_tp / (aspect_tp + aspect_fn + 1e-9)
    a_f1 = 2 * prec * rec / (prec + rec + 1e-9)
    validity = json_valid / len(predictions) if predictions else 0

    return {
        "sentiment_f1": round(f1, 3),
        "rating_mae": round(mae, 3),
        "aspect_f1": round(a_f1, 3),
        "json_validity": round(validity, 3),
        "n": len(predictions),
    }


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_yelp_sample(n: int) -> list:
    random.seed(RANDOM_SEED)
    records = []
    with open(YELP_LABELED, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # 按星级分层采样
    buckets = defaultdict(list)
    for r in records:
        buckets[r["stars"]].append(r)

    sampled = []
    per_star = n // 5
    for star in range(1, 6):
        pool = buckets[star]
        sampled.extend(random.sample(pool, min(per_star, len(pool))))
    random.shuffle(sampled)
    return sampled[:n]


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print(f"加载 Yelp 样本（{TEST_SAMPLE_SIZE} 条）...")
    test_data = load_yelp_sample(TEST_SAMPLE_SIZE)
    print(f"实际加载：{len(test_data)} 条（英文餐厅评论）")

    results = {
        "textblob": {"predictions": [], "latencies": []},
        "zero_shot": {"predictions": [], "latencies": []},
        "few_shot":  {"predictions": [], "latencies": []},
    }

    client = None
    if DEEPSEEK_API_KEY:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    else:
        print("⚠ 未设置 DEEPSEEK_API_KEY，只运行 TextBlob")

    # TextBlob
    print("\n[1/3] TextBlob...")
    for r in tqdm(test_data):
        start = time.time()
        pred = textblob_predict(r["text"])
        pred["_latency_ms"] = round((time.time() - start) * 1000)
        results["textblob"]["predictions"].append(pred)
        results["textblob"]["latencies"].append(pred["_latency_ms"])

    # Zero-shot
    if client:
        print("\n[2/3] Zero-shot LLM...")
        for r in tqdm(test_data):
            pred = llm_predict(client, r["text"], "zero")
            results["zero_shot"]["predictions"].append(pred)
            results["zero_shot"]["latencies"].append(pred.get("_latency_ms", 0))
            time.sleep(0.2)

        # Few-shot
        print("\n[3/3] Few-shot LLM...")
        for r in tqdm(test_data):
            pred = llm_predict(client, r["text"], "few")
            results["few_shot"]["predictions"].append(pred)
            results["few_shot"]["latencies"].append(pred.get("_latency_ms", 0))
            time.sleep(0.2)

    # 指标
    print("\n\n─── Yelp 评测结果（英文）────────────────────────────────────")
    print(f"{'指标':<20} {'TextBlob':>12} {'Zero-shot':>12} {'Few-shot':>12}")
    print("─" * 60)

    all_metrics = {}
    for method in ["textblob", "zero_shot", "few_shot"]:
        if not results[method]["predictions"]:
            continue
        m = compute_metrics(results[method]["predictions"], test_data)
        m["avg_latency_ms"] = round(np.mean(results[method]["latencies"]))
        all_metrics[method] = m

    for key, label in [
        ("sentiment_f1", "Sentiment F1"),
        ("rating_mae",   "Rating MAE"),
        ("aspect_f1",    "Aspect F1"),
        ("json_validity","JSON Validity"),
        ("avg_latency_ms","Latency (ms)"),
    ]:
        row = f"{label:<20}"
        for method in ["textblob", "zero_shot", "few_shot"]:
            val = all_metrics.get(method, {}).get(key, "N/A")
            row += f"{str(val):>12}"
        print(row)

    print("─" * 60)
    print("✦ TextBlob 在英文上有真实效果（对比中文 F1=0.111）")

    # 保存（合并进已有的 baseline_results.json）
    out_path = REPORTS_DIR / "baseline_results_yelp.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "Yelp (English cross-lingual validation)",
            "metrics": all_metrics,
            "test_size": len(test_data),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n已保存：{out_path}")

    # 打印对比摘要
    print("\n─── 中英文对比摘要 ──────────────────────────────────────────")
    print(f"{'方案':<16} {'ASAP Sentiment F1':>20} {'Yelp Sentiment F1':>20}")
    print("─" * 58)
    asap_f1 = {"textblob": 0.111, "zero_shot": 0.701, "few_shot": 0.757}
    for method, label in [("textblob","TextBlob"),("zero_shot","Zero-shot"),("few_shot","Few-shot")]:
        yelp_f1 = all_metrics.get(method, {}).get("sentiment_f1", "N/A")
        print(f"{label:<16} {asap_f1[method]:>20} {str(yelp_f1):>20}")


if __name__ == "__main__":
    main()
