"""
01_load_asap.py
───────────────
加载 ASAP 数据集（美团点评中文餐厅评论）
- 使用官方 train/dev/test 划分
- 转换 18 个 aspect 标签为结构化 JSON
- 输出 data/asap_processed/ 下的 jsonl 文件

运行：python src/01_load_asap.py
"""

import json
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import ROOT

# ── 路径 ──────────────────────────────────────────────────────────────────────
ASAP_DIR = ROOT / "data" / "asap" / "data"
OUT_DIR = ROOT / "data" / "asap_dataset" / "processed"

# ── Aspect 列映射（train.csv 的列名 → 我们的简化名）─────────────────────────
ASPECT_MAP = {
    "Location#Transportation":  "location_traffic",
    "Location#Downtown":        "location_distance",
    "Location#Easy_to_find":    "location_easy_to_find",
    "Service#Queue":            "service_wait_time",
    "Service#Hospitality":      "service_attitude",
    "Service#Parking":          "service_parking",
    "Service#Timely":           "service_speed",
    "Price#Level":              "price_level",
    "Price#Cost_effective":     "price_value",
    "Price#Discount":           "price_discount",
    "Ambience#Decoration":      "env_decoration",
    "Ambience#Noise":           "env_noise",
    "Ambience#Space":           "env_space",
    "Ambience#Sanitary":        "env_cleanliness",
    "Food#Portion":             "food_portion",
    "Food#Taste":               "food_taste",
    "Food#Appearance":          "food_appearance",
    "Food#Recommend":           "food_recommendation",
}

SENTIMENT_MAP = {1: "positive", 0: "neutral", -1: "negative"}


def parse_aspects(row: pd.Series) -> dict:
    """将 18 列 aspect 值转成 {aspect_name: sentiment} 字典，跳过 -2（未提及）"""
    aspects = {}
    for col, name in ASPECT_MAP.items():
        val = int(row[col])
        if val != -2:  # -2 = 未提及，跳过
            aspects[name] = SENTIMENT_MAP.get(val, "neutral")
    return aspects


def derive_sentiment(star: float) -> str:
    """用星级推导整体情感"""
    if star >= 4.0:
        return "positive"
    elif star == 3.0:
        return "neutral"
    else:
        return "negative"


def process_split(csv_path: Path, split_name: str) -> list[dict]:
    """处理单个 split（train/dev/test）"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"\n{split_name}: {len(df)} 条")
    print(f"  列名: {list(df.columns[:5])}...")

    records = []
    for _, row in df.iterrows():
        aspects = parse_aspects(row)
        star = float(row["star"])

        record = {
            "id": f"asap_{split_name}_{row['id']}",
            "text": str(row["review"]).strip(),
            "star": star,
            "split": split_name,
            # Gold labels（人工标注）
            "label": {
                "sentiment": derive_sentiment(star),
                "rating_prediction": round(star),
                "aspect_sentiments": aspects,
                # 以下 3 个字段需要 DeepSeek 补标（见 02_label_asap.py）
                "problem_type": None,
                "action_priority": None,
                "operator_action": None,
            }
        }
        records.append(record)

    return records


def print_stats(records: list[dict]):
    """打印数据统计"""
    from collections import Counter

    # 星级分布
    stars = Counter(r["label"]["rating_prediction"] for r in records)
    print("\n星级分布：")
    for s in sorted(stars):
        bar = "█" * (stars[s] // 100)
        print(f"  {s}星 {stars[s]:5d} {bar}")

    # 情感分布
    sents = Counter(r["label"]["sentiment"] for r in records)
    print(f"\n情感分布：{dict(sents)}")

    # aspect 覆盖率
    all_aspects = []
    for r in records:
        all_aspects.extend(r["label"]["aspect_sentiments"].keys())
    top = Counter(all_aspects).most_common(5)
    print(f"\nTop 5 aspects：{top}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    for split in ["train", "dev", "test"]:
        csv_path = ASAP_DIR / f"{split}.csv"
        if not csv_path.exists():
            print(f"找不到 {csv_path}，跳过")
            continue

        records = process_split(csv_path, split)
        all_records.extend(records)

        # 保存单个 split
        out_path = OUT_DIR / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  已保存 → {out_path}")

    print(f"\n总计：{len(all_records)} 条")
    print_stats(all_records)

    # 预览
    print("\n─── 数据预览 ──────────────────────────────────")
    sample = all_records[0]
    print(f"  文本：{sample['text'][:60]}...")
    print(f"  星级：{sample['star']}")
    print(f"  情感：{sample['label']['sentiment']}")
    print(f"  Aspects：{sample['label']['aspect_sentiments']}")
    print("─────────────────────────────────────────────")


if __name__ == "__main__":
    main()
