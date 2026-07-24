"""
prompts.py — 版本化的 prompt 组装层

系统提示词从 schema.DatasetProfile 生成，不再在各脚本里手写重复的字段列表——
这是修复 03_run_baselines.py 与 02/05/06 之间 VALID_PROBLEM_TYPE 漂移的根本手段：
以后改一个字段取值，只需要改 schema.py 一处，所有 prompt 自动同步。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from schema import DatasetProfile

PROMPT_VERSION = "v2-schema-unified"


def build_system_prompt(profile: DatasetProfile, mode: str = "full", lang: str | None = None) -> str:
    """mode="full": 6 字段（baseline / 训练数据）。mode="operational": 3 字段（silver 标注 / 微调模型评测）。"""
    lang = lang or profile.lang
    aspects = ", ".join(profile.aspects)
    sentiments = " / ".join(profile.sentiments)
    priorities = " / ".join(profile.priorities)
    problems = " / ".join(profile.problem_types)
    actions = " / ".join(profile.operator_actions)

    if lang == "zh":
        if mode == "full":
            return f"""你是餐厅经营分析助手。分析中文餐厅评论，输出结构化 JSON。

字段说明：
- sentiment: {sentiments}
- rating_prediction: 1-5（整数）
- aspect_sentiments: 涉及的维度及情感，从以下选（可多选）：
  {aspects}
- problem_type: {problems}
- action_priority: {priorities}
- operator_action: {actions}

只输出 JSON，不要其他文字。"""
        return f"""你是一个餐厅经营分析助手。根据评论内容和已知的 aspect 情感标签，补充以下 3 个字段：

- problem_type: 最主要的问题类型（只选一个）：{problems}
- action_priority: 商家处理紧迫程度（{priorities}）
- operator_action: 商家最应该做的一件事（只选一个）：{actions}

规则：
1. 只输出 JSON，不要其他文字
2. 正面评论（无投诉）→ problem_type=none，action_priority=low，operator_action=no_action
3. 只选上面列出的合法值"""

    # lang == "en"
    if mode == "full":
        return f"""You are a restaurant review analyst. Analyze the review and output structured JSON.
Fields: sentiment({sentiments.replace(' / ', '/')}), rating_prediction(1-5 int),
aspect_sentiments({{aspect:sentiment}}, aspects: {aspects}),
problem_type({problems.replace(' / ', '/')}),
action_priority({priorities.replace(' / ', '/')}),
operator_action({actions.replace(' / ', '/')}).
Output JSON only."""
    return f"""You are a restaurant review analyst. Given the review and known aspect sentiments, fill in these 3 fields:
- problem_type (pick one): {problems}
- action_priority: {priorities}
- operator_action (pick one): {actions}

Rules:
1. Output JSON only, no other text
2. Positive review with no complaint -> problem_type=none, action_priority=low, operator_action=no_action
3. Only use the values listed above"""


def build_persona_prompt(profile: DatasetProfile) -> str:
    """UI 专用：Streamlit demo 中「模拟微调模型」卡片使用的人设 prompt。
    真实 GPU 权重不在 Streamlit Cloud 上可用时，用这个 prompt 让 DeepSeek 模仿
    微调模型只输出 3 个运营字段的行为——仅用于展示，取值仍从 profile 生成以避免漂移。
    """
    problems = " / ".join(profile.problem_types)
    priorities = " / ".join(profile.priorities)
    actions = " / ".join(profile.operator_actions)
    if profile.lang == "zh":
        return f"""你是一个专门用于餐厅运营路由的轻量级模型（Qwen2.5-1.5B QLoRA微调版）。
给定餐厅评论，只输出以下 3 个字段的 JSON：
- problem_type: {problems}
- action_priority: {priorities}
- operator_action: {actions}
只输出 JSON，不含其他字段。"""
    return f"""You are a lightweight ops-routing model (fine-tuned Qwen2.5-1.5B QLoRA).
Given a restaurant review, output ONLY a JSON with exactly 3 fields:
- problem_type: {problems}
- action_priority: {priorities}
- operator_action: {actions}
Output JSON only. No other fields."""


@dataclass(frozen=True)
class FewShotExample:
    review: str
    output: dict


FEW_SHOT_BANK: dict[str, list[FewShotExample]] = {
    "asap": [
        FewShotExample(
            review="菜品非常新鲜，口味地道，服务也很好，就是价格稍微贵了点，但整体值得。",
            output={
                "sentiment": "positive", "rating_prediction": 4,
                "aspect_sentiments": {"food_taste": "positive", "service_attitude": "positive", "price_level": "negative"},
                "problem_type": "overpriced", "action_priority": "low", "operator_action": "review_pricing",
            },
        ),
        FewShotExample(
            review="等了将近一个小时才上菜，服务员态度也很差，菜的味道一般，不会再来了。",
            output={
                "sentiment": "negative", "rating_prediction": 1,
                "aspect_sentiments": {"service_wait_time": "negative", "service_attitude": "negative", "food_taste": "neutral"},
                "problem_type": "poor_service", "action_priority": "high", "operator_action": "train_service",
            },
        ),
        FewShotExample(
            review="环境不错，装修很有特色，菜量有点少，价格还可以接受，服务一般。",
            output={
                "sentiment": "neutral", "rating_prediction": 3,
                "aspect_sentiments": {"env_decoration": "positive", "food_portion": "negative", "price_level": "neutral", "service_attitude": "neutral"},
                "problem_type": "none", "action_priority": "low", "operator_action": "no_action",
            },
        ),
    ],
    "yelp": [
        FewShotExample(
            review="Great food, friendly staff, a bit pricey but totally worth it.",
            output={
                "sentiment": "positive", "rating_prediction": 4,
                "aspect_sentiments": {"product_quality": "positive", "customer_service": "positive", "value": "neutral"},
                "problem_type": "none", "action_priority": "low", "operator_action": "no_action",
            },
        ),
        FewShotExample(
            review="Waited over an hour, food was cold, staff were rude and dismissive.",
            output={
                "sentiment": "negative", "rating_prediction": 1,
                "aspect_sentiments": {"customer_service": "negative", "logistics": "negative", "product_quality": "negative"},
                "problem_type": "poor_service", "action_priority": "high", "operator_action": "train_service",
            },
        ),
        FewShotExample(
            review="Decent place, nothing special. Food okay, service average, fair prices.",
            output={
                "sentiment": "neutral", "rating_prediction": 3,
                "aspect_sentiments": {"product_quality": "neutral", "customer_service": "neutral", "value": "positive"},
                "problem_type": "none", "action_priority": "low", "operator_action": "no_action",
            },
        ),
    ],
}


def build_user_message(
    text: str,
    profile: DatasetProfile,
    few_shot: list[FewShotExample] | None = None,
    max_chars: int = 400,
) -> str:
    """6 字段任务的 user message（zero-shot 传 few_shot=None，few-shot 传 FEW_SHOT_BANK[profile.name]）。"""
    lang = profile.lang
    review_key = "评论" if lang == "zh" else "Review"
    output_key = "输出" if lang == "zh" else "Output"
    sep = "：" if lang == "zh" else ": "
    output_json_label = "输出 JSON：" if lang == "zh" else "Output JSON:"

    example_blocks = ""
    for ex in few_shot or []:
        example_blocks += f"{review_key}{sep}{ex.review}\n{output_key}{sep}{json.dumps(ex.output, ensure_ascii=False)}\n\n"

    return f"{example_blocks}{review_key}{sep}{text[:max_chars]}\n\n{output_json_label}"


def build_operational_user_message(
    text: str,
    aspect_sentiments: dict,
    profile: DatasetProfile,
    max_chars: int = 300,
) -> str:
    """3 字段任务的 user message（silver 标注 / 微调模型评测），带已知 aspect 情感作为上下文。"""
    aspect_str = json.dumps(aspect_sentiments, ensure_ascii=False)
    if profile.lang == "zh":
        return f"评论：{text[:max_chars]}\n\n已知 aspect 情感：{aspect_str}\n\n输出 JSON（仅 3 个字段）："
    return f"Review: {text[:max_chars]}\n\nKnown aspect sentiments: {aspect_str}\n\nOutput JSON (3 fields only):"
