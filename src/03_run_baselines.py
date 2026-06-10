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
import re
from pathlib import Path
from typing import Optional
from collections import defaultdict

from textblob import TextBlob
from openai import OpenAI
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ROOT, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    ASAP_PROCESSED_DIR
)

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

TEST_SAMPLE_SIZE = 200   # 从 test set 取多少条评测
RANDOM_SEED = 42

# ── 合法值 ────────────────────────────────────────────────────────────────────
VALID_SENTIMENTS = ["positive", "neutral", "negative"]
VALID_ASPECTS = [
    "location_traffic", "location_distance", "location_easy_to_find",
    "service_wait_time", "service_attitude", "service_parking", "service_speed",
    "price_level", "price_value", "price_discount",
    "env_decoration", "env_noise", "env_space", "env_cleanliness",
    "food_portion", "food_taste", "food_appearance", "food_recommendation",
]

# ── Few-shot 示例（来自 ASAP 训练集风格）────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "review": "菜品非常新鲜，口味地道，服务也很好，就是价格稍微贵了点，但整体值得。",
        "output": {
            "sentiment": "positive",
            "rating_prediction": 4,
            "aspect_sentiments": {
                "food_taste": "positive",
                "service_attitude": "positive",
                "price_level": "negative"
            },
            "problem_type": "overpriced",
            "action_priority": "low",
            "operator_action": "review_pricing"
        }
    },
    {
        "review": "等了将近一个小时才上菜，服务员态度也很差，菜的味道一般，不会再来了。",
        "output": {
            "sentiment": "negative",
            "rating_prediction": 1,
            "aspect_sentiments": {
                "service_wait_time": "negative",
                "service_attitude": "negative",
                "food_taste": "neutral"
            },
            "problem_type": "poor_service",
            "action_priority": "high",
            "operator_action": "train_service"
        }
    },
    {
        "review": "环境不错，装修很有特色，菜量有点少，价格还可以接受，服务一般。",
        "output": {
            "sentiment": "neutral",
            "rating_prediction": 3,
            "aspect_sentiments": {
                "env_decoration": "positive",
                "food_portion": "negative",
                "price_level": "neutral",
                "service_attitude": "neutral"
            },
            "problem_type": "none",
            "action_priority": "low",
            "operator_action": "no_action"
        }
    },
]

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是餐厅经营分析助手。分析中文餐厅评论，输出结构化 JSON。

字段说明：
- sentiment: positive / neutral / negative
- rating_prediction: 1-5（整数）
- aspect_sentiments: 涉及的维度及情感，从以下选（可多选）：
  food_taste, food_portion, food_appearance, food_recommendation,
  service_attitude, service_wait_time, service_speed, service_parking,
  price_level, price_value, price_discount,
  env_decoration, env_noise, env_space, env_cleanliness,
  location_traffic, location_distance, location_easy_to_find
- problem_type: taste_issue / poor_service / long_wait / overpriced / hygiene_issue / none
- action_priority: low / medium / high
- operator_action: improve_taste / train_service / reduce_wait / review_pricing / fix_hygiene / no_action

只输出 JSON，不要其他文字。"""


def build_zero_shot_prompt(text: str) -> str:
    return f"评论：{text[:400]}\n\n输出 JSON："


def build_few_shot_prompt(text: str) -> str:
    examples = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples += f"评论：{ex['review']}\n"
        examples += f"输出：{json.dumps(ex['output'], ensure_ascii=False)}\n\n"
    return f"{examples}评论：{text[:400]}\n输出 JSON："


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


# ── LLM Baseline ──────────────────────────────────────────────────────────────
def parse_llm_output(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        # 尝试提取第一个 {...}
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


def llm_predict(client: OpenAI, text: str, mode: str = "zero") -> dict:
    prompt = build_zero_shot_prompt(text) if mode == "zero" else build_few_shot_prompt(text)
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        latency = (time.time() - start) * 1000
        result = parse_llm_output(resp.choices[0].message.content)
        if result:
            result["_latency_ms"] = round(latency)
            result["_valid_json"] = True
        else:
            result = {"_valid_json": False, "_latency_ms": round(latency)}
    except Exception as e:
        result = {"_valid_json": False, "_error": str(e), "_latency_ms": 0}
    return result


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
    output = {
        "metrics": all_metrics,
        "test_size": len(test_data),
        "predictions": {
            method: results[method]["predictions"]
            for method in results
            if results[method]["predictions"]
        }
    }
    out_path = REPORTS_DIR / "baseline_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n已保存：{out_path}")


if __name__ == "__main__":
    main()
