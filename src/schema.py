"""
schema.py — 结构化输出的唯一事实来源（single source of truth）

DatasetProfile 定义每个数据集（ASAP / Yelp）合法的 aspect / problem_type /
operator_action 取值。Pydantic 模型在这些取值上做校验，取代原来分散在
02/03/05/06/app.py 五处、且已经出现漂移的 VALID_* 列表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ValidationInfo, field_validator


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    lang: str  # "zh" | "en"
    aspects: tuple[str, ...]
    problem_types: tuple[str, ...]
    operator_actions: tuple[str, ...]
    priorities: tuple[str, ...] = ("low", "medium", "high")
    sentiments: tuple[str, ...] = ("positive", "neutral", "negative")


ASAP_PROFILE = DatasetProfile(
    name="asap",
    lang="zh",
    aspects=(
        "location_traffic", "location_distance", "location_easy_to_find",
        "service_wait_time", "service_attitude", "service_parking", "service_speed",
        "price_level", "price_value", "price_discount",
        "env_decoration", "env_noise", "env_space", "env_cleanliness",
        "food_portion", "food_taste", "food_appearance", "food_recommendation",
    ),
    problem_types=(
        "taste_issue", "poor_service", "long_wait", "overpriced",
        "hygiene_issue", "location_issue", "packaging_issue", "none",
    ),
    operator_actions=(
        "improve_taste", "train_service", "reduce_wait",
        "review_pricing", "fix_hygiene", "no_action",
    ),
)

YELP_PROFILE = DatasetProfile(
    name="yelp",
    lang="en",
    aspects=(
        "product_quality", "logistics", "customer_service",
        "packaging", "value", "authenticity",
    ),
    problem_types=(
        "quality_issue", "slow_logistics", "poor_service",
        "overpriced", "fake_product", "packaging_damage", "none",
    ),
    operator_actions=(
        "fix_quality", "improve_logistics", "train_service",
        "review_pricing", "verify_authenticity", "no_action",
    ),
)

PROFILES = {"asap": ASAP_PROFILE, "yelp": YELP_PROFILE}


def _profile_of(info: ValidationInfo) -> Optional[DatasetProfile]:
    return (info.context or {}).get("profile") if info.context else None


# 模型给 aspect / sentiment 用的字段名，各家措辞不一样。放模块级而不是类属性——
# Pydantic 会把类里带下划线前缀的属性变成 ModelPrivateAttr，取出来就不是元组了。
_ASPECT_KEYS = ("aspect", "name", "category", "aspect_name")
_SENTIMENT_KEYS = ("sentiment", "polarity", "value", "label")


class OperationalFields(BaseModel):
    """3 个运营路由字段 — 微调模型 (Qwen2.5-1.5B QLoRA) 的完整输出。"""

    problem_type: str
    action_priority: str
    operator_action: str

    @field_validator("problem_type")
    @classmethod
    def _check_problem_type(cls, v: str, info: ValidationInfo) -> str:
        profile = _profile_of(info)
        if profile and v not in profile.problem_types:
            raise ValueError(f"problem_type must be one of {profile.problem_types}, got {v!r}")
        return v

    @field_validator("action_priority")
    @classmethod
    def _check_action_priority(cls, v: str, info: ValidationInfo) -> str:
        profile = _profile_of(info)
        if profile and v not in profile.priorities:
            raise ValueError(f"action_priority must be one of {profile.priorities}, got {v!r}")
        return v

    @field_validator("operator_action")
    @classmethod
    def _check_operator_action(cls, v: str, info: ValidationInfo) -> str:
        profile = _profile_of(info)
        if profile and v not in profile.operator_actions:
            raise ValueError(f"operator_action must be one of {profile.operator_actions}, got {v!r}")
        return v


class FullReviewAnalysis(OperationalFields):
    """6 字段完整输出 — zero-shot / few-shot baseline 的目标 schema。"""

    sentiment: str
    rating_prediction: int
    aspect_sentiments: dict[str, str] = {}

    @field_validator("sentiment")
    @classmethod
    def _check_sentiment(cls, v: str, info: ValidationInfo) -> str:
        profile = _profile_of(info)
        if profile and v not in profile.sentiments:
            raise ValueError(f"sentiment must be one of {profile.sentiments}, got {v!r}")
        return v

    @field_validator("rating_prediction", mode="before")
    @classmethod
    def _coerce_rating(cls, v):
        # 容错而非拒绝：与旧 compute_metrics 的 clamp 行为保持一致
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 3
        return max(1, min(5, v))

    @field_validator("aspect_sentiments", mode="before")
    @classmethod
    def _coerce_aspects(cls, v, info: ValidationInfo):
        """容错而非拒绝：把模型实际会吐出来的几种形状都归一成 {aspect: sentiment}。

        真实遇到过的形状（deepseek-v4-flash 就用第 2 种）：
          1. {"food_taste": "positive"}                              —— 期望形状
          2. [{"aspect": "food_taste", "sentiment": "positive"}, ...] —— list of dict
          3. ["food_taste", "service_attitude"]                      —— list of str
        未知 aspect / 非法 sentiment 一律丢弃，不让它污染下游指标。
        """
        if isinstance(v, list):
            coerced = {}
            for item in v:
                if isinstance(item, dict):
                    aspect = next((item[k] for k in _ASPECT_KEYS if k in item), None)
                    sentiment = next((item[k] for k in _SENTIMENT_KEYS if k in item), "neutral")
                    # 也可能是 {"food_taste": "positive"} 这种单键 dict
                    if aspect is None and len(item) == 1:
                        aspect, sentiment = next(iter(item.items()))
                    if isinstance(aspect, str):
                        coerced[aspect] = sentiment
                elif isinstance(item, str):
                    coerced[item] = "neutral"
            v = coerced
        if not isinstance(v, dict):
            return {}
        # 键/值可能不是字符串（嵌套 dict、数字…），先过滤掉再比对，避免 unhashable/类型错误
        clean = {a: s for a, s in v.items() if isinstance(a, str) and isinstance(s, str)}
        profile = _profile_of(info)
        if not profile:
            return clean
        return {a: s for a, s in clean.items() if a in profile.aspects and s in profile.sentiments}
