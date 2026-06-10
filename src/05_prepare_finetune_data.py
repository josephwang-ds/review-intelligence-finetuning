"""
05_prepare_finetune_data.py
────────────────────────────
将已标注的 ASAP 数据转换为 Qwen2.5-Instruct chat 格式，用于微调。

输入：data/asap_dataset/labeled/train_labeled.jsonl（约 4000 条）
输出：data/finetune/train.jsonl / val.jsonl / test.jsonl

切分比例：8:1:1，随机种子 42

assistant 输出包含全部 6 个字段：
  gold 字段：sentiment / rating_prediction / aspect_sentiments
  silver 字段：problem_type / action_priority / operator_action
会过滤 problem_type 或 action_priority 为 null 的样本（标注失败）
"""

import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import ROOT

# ── 路径配置 ────────────────────────────────────────────────────────────────────

IN_PATH = ROOT / "data" / "asap_dataset" / "labeled" / "train_labeled.jsonl"
OUT_DIR = ROOT / "data" / "finetune"

# ── System Prompt（完整 6 字段评论分析，与 03_run_baselines.py 风格对齐）──────────

ASAP_SYSTEM = """你是餐厅经营分析助手。分析中文餐厅评论，输出结构化 JSON。

字段说明：
- sentiment: positive / neutral / negative
- rating_prediction: 1-5（整数）
- aspect_sentiments: 涉及的维度及情感，从以下选（可多选）：
  food_taste, food_portion, food_appearance, food_recommendation,
  service_attitude, service_wait_time, service_speed, service_parking,
  price_level, price_value, price_discount,
  env_decoration, env_noise, env_space, env_cleanliness,
  location_traffic, location_distance, location_easy_to_find
- problem_type: taste_issue / poor_service / long_wait / overpriced / hygiene_issue / location_issue / packaging_issue / none
- action_priority: low / medium / high
- operator_action: improve_taste / train_service / reduce_wait / review_pricing / fix_hygiene / no_action

只输出 JSON，不要其他文字。"""


def build_user_message(record: dict) -> str:
    """构建 user 侧消息：仅评论文本"""
    return f"评论：{record['text'][:400]}\n\n输出 JSON："


def build_assistant_message(record: dict) -> str:
    """构建 assistant 侧消息：合并 gold + silver 全部 6 个字段的完整 JSON"""
    label = record["label"]
    output = {
        "sentiment": label["sentiment"],
        "rating_prediction": label["rating_prediction"],
        "aspect_sentiments": label["aspect_sentiments"],
        "problem_type": label["problem_type"],
        "action_priority": label["action_priority"],
        "operator_action": label["operator_action"],
    }
    return json.dumps(output, ensure_ascii=False)


def to_chat_format(record: dict) -> dict:
    """将单条记录转为 Qwen2.5-Instruct chat 格式"""
    return {
        "id": record["id"],
        "messages": [
            {"role": "system", "content": ASAP_SYSTEM},
            {"role": "user", "content": build_user_message(record)},
            {"role": "assistant", "content": build_assistant_message(record)},
        ],
    }


def write_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    if not IN_PATH.exists():
        print(f"找不到输入文件：{IN_PATH}")
        print("请先运行 02_label_asap.py 生成标注数据")
        return

    # 读取全部标注数据
    records = []
    with open(IN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"读取总条数：{len(records)}")

    # 过滤 problem_type 或 action_priority 为 null 的样本（标注失败）
    before = len(records)
    records = [
        r for r in records
        if r["label"].get("problem_type") is not None
        and r["label"].get("action_priority") is not None
    ]
    print(f"过滤后条数：{len(records)}（过滤掉 {before - len(records)} 条标注失败）")

    # 随机打乱，固定种子
    random.seed(42)
    random.shuffle(records)

    # 8:1:1 切分
    total = len(records)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)

    train_raw = records[:n_train]
    val_raw = records[n_train:n_train + n_val]
    test_raw = records[n_train + n_val:]

    # 转换为 chat 格式
    train_data = [to_chat_format(r) for r in train_raw]
    val_data = [to_chat_format(r) for r in val_raw]
    test_data = [to_chat_format(r) for r in test_raw]

    # 写出
    write_jsonl(train_data, OUT_DIR / "train.jsonl")
    write_jsonl(val_data, OUT_DIR / "val.jsonl")
    write_jsonl(test_data, OUT_DIR / "test.jsonl")

    print(f"\n切分结果：")
    print(f"  train : {len(train_data)} 条 -> data/finetune/train.jsonl")
    print(f"  val   : {len(val_data)} 条 -> data/finetune/val.jsonl")
    print(f"  test  : {len(test_data)} 条 -> data/finetune/test.jsonl")

    # 打印一条样例
    print("\n样例预览（train[0]）：")
    sample = train_data[0]
    print(f"  id: {sample['id']}")
    for msg in sample["messages"]:
        role = msg["role"]
        content = msg["content"]
        preview = content[:120].replace("\n", " ")
        print(f"  [{role}]: {preview}{'...' if len(content) > 120 else ''}")


if __name__ == "__main__":
    main()
