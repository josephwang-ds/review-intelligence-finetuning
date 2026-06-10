"""
01_load_and_sample.py
─────────────────────
从 HuggingFace 加载 amazon_reviews_multi (zh)，
按星级分层采样，输出 data/raw/sample.jsonl

运行：python src/01_load_and_sample.py
耗时：首次运行约 2~5 分钟（下载数据集）；之后走缓存秒级完成
"""

import json
import random
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_DIR, DATASET_NAME, DATASET_LANG,
    SAMPLES_PER_STAR, RANDOM_SEED
)


def load_and_sample() -> list[dict]:
    """从 HuggingFace 加载 yelp_review_full 并分层采样"""
    print(f"加载数据集 {DATASET_NAME}...")
    # yelp_review_full 不需要语言参数
    ds = load_dataset(DATASET_NAME, trust_remote_code=True)

    # 只用 train split（650k 条，够用）
    all_data = list(ds["train"])
    print(f"原始数据量：{len(all_data):,} 条")

    # yelp 字段：label (0-4，对应 1-5星), text
    # 转换为统一的 stars (1-5)
    buckets: dict[int, list] = defaultdict(list)
    for item in all_data:
        stars = item["label"] + 1  # 0-4 → 1-5
        buckets[stars].append(item)

    print("\n各星级数量：")
    for star in sorted(buckets.keys()):
        print(f"  {star}星：{len(buckets[star]):,} 条")

    # 分层采样
    random.seed(RANDOM_SEED)
    sampled = []
    for star, n in SAMPLES_PER_STAR.items():
        pool = buckets[star]
        chosen = random.sample(pool, min(n, len(pool)))
        for i, item in enumerate(chosen):
            sampled.append({
                "id": f"yelp_{star}_{i}",
                "review_body": item["text"],
                "stars": star,
                "product_category": "restaurant",
                "text": item["text"][:500],  # 限制长度，节省 token
            })

    random.shuffle(sampled)
    print(f"\n采样完成：{len(sampled)} 条")
    _print_distribution(sampled)
    return sampled


def _print_distribution(data: list[dict]):
    from collections import Counter
    dist = Counter(d["stars"] for d in data)
    print("采样后分布：")
    for star in sorted(dist.keys()):
        bar = "█" * (dist[star] // 20)
        print(f"  {star}星 {dist[star]:4d} {bar}")


def save_sample(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n已保存：{path}  ({len(data)} 条)")


def main():
    out_path = RAW_DIR / "sample.jsonl"

    if out_path.exists():
        print(f"sample.jsonl 已存在（{out_path}），跳过重新采样。")
        print("如需重新采样，请先删除该文件。")
        return

    data = load_and_sample()
    save_sample(data, out_path)

    # 打印前 2 条预览
    print("\n─── 数据预览 ────────────────────────────────────────")
    for item in data[:2]:
        print(f"  stars={item['stars']} | {item['text'][:80]}...")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
