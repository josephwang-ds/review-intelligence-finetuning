"""
02_label_with_llm.py
─────────────────────
用 DeepSeek API 对 sample.jsonl 批量打标签，
输出 data/labeled/labeled.jsonl

运行：python src/02_label_with_llm.py
费用估算：2000 条 × ~400 tokens = ~800k tokens ≈ $0.15（DeepSeek-chat）
耗时：约 15~30 分钟（含 API 限速等待）
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
    RAW_DIR, LABELED_DIR,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    LABEL_BATCH_SIZE, LABEL_MAX_RETRIES, LABEL_TEMPERATURE,
    VALID_SENTIMENT, VALID_ASPECTS, VALID_PROBLEM_TYPE,
    VALID_ACTION_PRIORITY, VALID_OPERATOR_ACTION
)

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个电商评论分析专家。你的任务是分析中文电商评论，输出结构化 JSON。

输出字段说明：
- sentiment: 评论整体情感 (positive / neutral / negative)
- rating_prediction: 预测星级 1-5（整数），基于评论内容
- aspects: 评论涉及的维度，从以下选：
    product_quality（商品质量）
    logistics（物流速度）
    customer_service（客服态度）
    packaging（包装）
    value（性价比）
    authenticity（正品保障）
  可多选，用数组
- problem_type: 主要问题类型（如果是负面评论）：
    quality_issue / slow_logistics / poor_service /
    overpriced / fake_product / packaging_damage / none
  只选一个
- action_priority: 商家需要处理的紧迫程度 (low / medium / high)
- operator_action: 商家最应该做的一件事：
    fix_quality / improve_logistics / train_service /
    review_pricing / verify_authenticity / no_action
  只选一个

严格要求：
1. 只输出 JSON，不要任何其他文字
2. 所有字段必须存在
3. 只能使用上面列出的合法值
4. aspects 如果只有一个维度也要用数组格式：["product_quality"]"""


def build_user_prompt(review_text: str) -> str:
    return f"""评论内容：
{review_text}

请分析并输出 JSON："""


# ── 校验 ──────────────────────────────────────────────────────────────────────

def validate_label(label: dict) -> tuple[bool, str]:
    """校验 LLM 输出是否合法，返回 (is_valid, error_msg)"""
    required = ["sentiment", "rating_prediction", "aspects",
                "problem_type", "action_priority", "operator_action"]

    for field in required:
        if field not in label:
            return False, f"缺少字段: {field}"

    if label["sentiment"] not in VALID_SENTIMENT:
        return False, f"sentiment 非法: {label['sentiment']}"

    rating = label["rating_prediction"]
    if not isinstance(rating, int) or not (1 <= rating <= 5):
        return False, f"rating_prediction 非法: {rating}"

    aspects = label["aspects"]
    if not isinstance(aspects, list):
        return False, "aspects 必须是数组"
    for a in aspects:
        if a not in VALID_ASPECTS:
            return False, f"aspects 包含非法值: {a}"

    if label["problem_type"] not in VALID_PROBLEM_TYPE:
        return False, f"problem_type 非法: {label['problem_type']}"

    if label["action_priority"] not in VALID_ACTION_PRIORITY:
        return False, f"action_priority 非法: {label['action_priority']}"

    if label["operator_action"] not in VALID_OPERATOR_ACTION:
        return False, f"operator_action 非法: {label['operator_action']}"

    return True, ""


# ── 一致性检查 ────────────────────────────────────────────────────────────────

def check_consistency(item: dict, label: dict) -> dict:
    """
    用 stars（原始真实标签）做一致性软校验。
    不过滤，只加 flag，用于后续分析。
    """
    star = item["stars"]
    sent = label["sentiment"]

    # stars 1-2 应该是 negative，4-5 应该是 positive
    expected = {1: "negative", 2: "negative", 3: "neutral",
                4: "positive", 5: "positive"}.get(star, "")
    label["_star_sentiment_match"] = (sent == expected)
    label["_original_stars"] = star
    return label


# ── API 调用 ──────────────────────────────────────────────────────────────────

def call_deepseek(client: OpenAI, text: str) -> Optional[dict]:
    """调用 DeepSeek，带重试，返回解析后的 dict 或 None"""
    for attempt in range(LABEL_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(text)},
                ],
                temperature=LABEL_TEMPERATURE,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()

            # 处理可能的 markdown 代码块
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            label = json.loads(raw)
            is_valid, err = validate_label(label)
            if is_valid:
                return label
            else:
                print(f"  ⚠ 格式校验失败（第{attempt+1}次）: {err}")

        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON 解析失败（第{attempt+1}次）: {e}")
        except Exception as e:
            print(f"  ⚠ API 错误（第{attempt+1}次）: {e}")
            time.sleep(2 ** attempt)  # 指数退避

    return None  # 所有重试失败


# ── 主流程 ────────────────────────────────────────────────────────────────────

def load_already_labeled(out_path: Path) -> set[str]:
    """加载已完成的 id，支持断点续跑"""
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                done.add(item["id"])
            except Exception:
                pass
    return done


def main():
    if not DEEPSEEK_API_KEY:
        print("错误：请在 .env 文件中设置 DEEPSEEK_API_KEY")
        print("参考 .env.example 文件")
        return

    in_path = RAW_DIR / "sample.jsonl"
    out_path = LABELED_DIR / "labeled.jsonl"
    failed_path = LABELED_DIR / "failed.jsonl"

    if not in_path.exists():
        print(f"错误：找不到 {in_path}")
        print("请先运行：python src/01_load_and_sample.py")
        return

    # 读取样本
    samples = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"共 {len(samples)} 条待标注")

    # 断点续跑：跳过已标注的
    done_ids = load_already_labeled(out_path)
    samples = [s for s in samples if s["id"] not in done_ids]
    if done_ids:
        print(f"已完成 {len(done_ids)} 条，跳过，剩余 {len(samples)} 条")

    if not samples:
        print("全部标注完成！")
        return

    # 初始化 client
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    success, failed = 0, 0

    with open(out_path, "a", encoding="utf-8") as f_out, \
         open(failed_path, "a", encoding="utf-8") as f_fail:

        for item in tqdm(samples, desc="标注中"):
            label = call_deepseek(client, item["text"])

            if label is not None:
                label = check_consistency(item, label)
                output = {**item, "label": label}
                f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
                f_out.flush()
                success += 1
            else:
                f_fail.write(json.dumps(item, ensure_ascii=False) + "\n")
                f_fail.flush()
                failed += 1

            # 简单限速：避免触发 rate limit
            time.sleep(0.3)

    total = success + failed
    print(f"\n标注完成：{success}/{total} 成功，{failed} 失败")
    print(f"输出：{out_path}")
    if failed > 0:
        print(f"失败记录：{failed_path}（可重新运行补全）")

    # 一致性统计
    _print_consistency_report(out_path)


def _print_consistency_report(path: Path):
    """输出 sentiment vs original stars 的一致性报告"""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception:
                pass

    if not data:
        return

    match = sum(1 for d in data if d["label"].get("_star_sentiment_match", False))
    print(f"\n一致性检查：{match}/{len(data)} 条 sentiment 与原始星级一致 "
          f"({match/len(data)*100:.1f}%)")

    # 标签分布
    from collections import Counter
    sent_dist = Counter(d["label"]["sentiment"] for d in data)
    print(f"Sentiment 分布：{dict(sent_dist)}")
    priority_dist = Counter(d["label"]["action_priority"] for d in data)
    print(f"Action Priority 分布：{dict(priority_dist)}")


if __name__ == "__main__":
    main()
