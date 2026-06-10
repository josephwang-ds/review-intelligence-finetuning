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
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    LABEL_MAX_RETRIES, LABEL_TEMPERATURE, ROOT
)

ASAP_PROCESSED_DIR = ROOT / "data" / "asap_dataset" / "processed"
OUT_DIR = ROOT / "data" / "asap_dataset" / "labeled"

# 只需要补这 3 个字段
VALID_PROBLEM_TYPE = [
    "taste_issue", "poor_service", "long_wait", "overpriced",
    "hygiene_issue", "location_issue", "packaging_issue", "none"
]
VALID_ACTION_PRIORITY = ["low", "medium", "high"]
VALID_OPERATOR_ACTION = [
    "improve_taste", "train_service", "reduce_wait",
    "review_pricing", "fix_hygiene", "no_action"
]

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个餐厅经营分析助手。根据评论内容和已知的 aspect 情感标签，补充以下 3 个字段：

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


def build_prompt(text: str, aspects: dict) -> str:
    aspect_str = json.dumps(aspects, ensure_ascii=False)
    return f"""评论：{text[:300]}

已知 aspect 情感：{aspect_str}

输出 JSON（仅 3 个字段）："""


def validate(label: dict) -> tuple[bool, str]:
    for field in ["problem_type", "action_priority", "operator_action"]:
        if field not in label:
            return False, f"缺少字段: {field}"
    if label["problem_type"] not in VALID_PROBLEM_TYPE:
        return False, f"problem_type 非法: {label['problem_type']}"
    if label["action_priority"] not in VALID_ACTION_PRIORITY:
        return False, f"action_priority 非法: {label['action_priority']}"
    if label["operator_action"] not in VALID_OPERATOR_ACTION:
        return False, f"operator_action 非法: {label['operator_action']}"
    return True, ""


def call_deepseek(client: OpenAI, text: str, aspects: dict) -> Optional[dict]:
    for attempt in range(LABEL_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(text, aspects)},
                ],
                temperature=LABEL_TEMPERATURE,
                max_tokens=100,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            label = json.loads(raw.strip())
            ok, err = validate(label)
            if ok:
                return label
            print(f"  ⚠ 校验失败（第{attempt+1}次）: {err}")
        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON 解析失败（第{attempt+1}次）: {e}")
        except Exception as e:
            print(f"  ⚠ API 错误（第{attempt+1}次）: {e}")
            time.sleep(2 ** attempt)
    return None


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
            extra = call_deepseek(
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
