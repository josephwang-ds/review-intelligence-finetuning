"""
02_label_asap.py
─────────────────
用 DeepSeek 给 ASAP 数据补标 3 个字段：
  problem_type / action_priority / operator_action

注意：rating / sentiment / aspect_sentiments 已经是 gold label，不需要重新标注。

运行：python src/02_label_asap.py
费用估算：5000 条 × ~200 tokens = ~1M tokens ≈ $0.06（比 Yelp 便宜很多）
耗时：约 15~20 分钟
"""

import json
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LABEL_MAX_RETRIES, LABEL_TEMPERATURE, ROOT
from schema import ASAP_PROFILE
from structured_client import call_structured

ASAP_PROCESSED_DIR = ROOT / "data" / "asap_dataset" / "processed"
OUT_DIR = ROOT / "data" / "asap_dataset" / "labeled"


def label_one(client: OpenAI, text: str, aspects: dict) -> Optional[dict]:
    result, meta = call_structured(
        client, ASAP_PROFILE, text, mode="operational",
        aspect_context=aspects,
        temperature=LABEL_TEMPERATURE,  # 同上：不要再压 max_tokens
        max_repairs=LABEL_MAX_RETRIES - 1,
    )
    if result is None:
        print(f"  ⚠ 标注失败: {meta['error']}")
        return None
    return result.model_dump()


def load_done_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


def main():
    if not DEEPSEEK_API_KEY:
        print("错误：请在 .env 设置 DEEPSEEK_API_KEY")
        return

    # 只标注 train split（dev/test 用于评测，不参与训练标注）
    in_path = ASAP_PROCESSED_DIR / "train_sampled.jsonl"
    out_path = OUT_DIR / "train_labeled.jsonl"
    failed_path = OUT_DIR / "train_failed.jsonl"

    if not in_path.exists():
        print(f"找不到 {in_path}，请先运行 01_load_asap.py")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取数据
    records = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"待标注：{len(records)} 条")

    # 断点续跑
    done_ids = load_done_ids(out_path)
    records = [r for r in records if r["id"] not in done_ids]
    if done_ids:
        print(f"已完成 {len(done_ids)} 条，跳过，剩余 {len(records)} 条")

    if not records:
        print("全部完成！")
        return

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    success, failed = 0, 0

    with open(out_path, "a", encoding="utf-8") as f_out, \
         open(failed_path, "a", encoding="utf-8") as f_fail:

        for record in tqdm(records, desc="补标注中"):
            extra = label_one(
                client,
                record["text"],
                record["label"]["aspect_sentiments"]
            )

            if extra is not None:
                # 合并 gold label + 补充字段
                record["label"]["problem_type"] = extra["problem_type"]
                record["label"]["action_priority"] = extra["action_priority"]
                record["label"]["operator_action"] = extra["operator_action"]
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                success += 1
            else:
                f_fail.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_fail.flush()
                failed += 1

            time.sleep(0.2)

    print(f"\n完成：{success} 成功，{failed} 失败")
    print(f"输出：{out_path}")


if __name__ == "__main__":
    main()
