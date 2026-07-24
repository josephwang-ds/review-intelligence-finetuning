"""
guardrails.py — 输入/输出护栏

设计原则：规则集保持小而可辩护，而不是搭一个通用规则引擎。每一条规则都应该是
能在面试里讲清楚"为什么是这条、为什么这么定阈值"的，不是为了凑功能列表。

关于 prompt injection：check_input() 里的关键词匹配只是一个监控信号，不是真正的
防线——真正的防线是架构性的：评论文本永远只作为 user 角色内容传入，从不拼接进
system prompt；无论输入想让模型说什么，模型的输出都还要经过 structured_client.py
的 Pydantic schema 校验。关键词匹配能做的只是把可疑输入标记出来供人工复核，
不该被当成"挡住了攻击"的证明。
"""

from __future__ import annotations

import re

MAX_INPUT_CHARS = 2000

_INJECTION_PATTERNS_ZH = [
    "忽略之前的指令", "忽略上面的指令", "忽略上述指令", "忽略以上",
    "你现在是", "你现在扮演", "新的指令：", "系统提示", "系统指令",
]
_INJECTION_PATTERNS_EN = [
    "ignore previous instructions", "ignore all previous", "disregard the above",
    "you are now", "new instructions:", "system prompt:", "act as ",
]

_CN_ID_RE = re.compile(r"\b\d{17}[\dXx]\b")
_PHONE_CN_RE = re.compile(r"1[3-9]\d{9}")
_PHONE_GENERIC_RE = re.compile(r"\b\d{3}[-.\s]\d{3,4}[-.\s]\d{4}\b")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+")


def check_input(text: str) -> list[str]:
    """返回触发的规则名列表；空列表 = 干净。"""
    flags = []
    if len(text) > MAX_INPUT_CHARS:
        flags.append("too_long")
    lowered = text.lower()
    if any(p in text for p in _INJECTION_PATTERNS_ZH) or any(p in lowered for p in _INJECTION_PATTERNS_EN):
        flags.append("prompt_injection_suspected")
    return flags


def redact_pii(text: str) -> tuple[str, list[str]]:
    """脱敏后的文本才是真正发给 DeepSeek / 存进队列的内容，不只是展示层打码。"""
    redacted = text
    found: list[str] = []
    for pattern, label in [
        (_CN_ID_RE, "ID"),
        (_PHONE_CN_RE, "PHONE"),
        (_PHONE_GENERIC_RE, "PHONE"),
        (_EMAIL_RE, "EMAIL"),
    ]:
        if pattern.search(redacted):
            found.append(label)
            redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return redacted, sorted(set(found))


def check_output_consistency(result: dict) -> list[str]:
    """针对已通过 schema 校验的输出做业务逻辑一致性检查（对只有 3 个运营字段的
    OperationalFields 结果，sentiment 取不到值，两条规则自然不触发）。"""
    flags = []
    sentiment = result.get("sentiment")
    problem_type = result.get("problem_type")
    action_priority = result.get("action_priority")
    if sentiment == "negative" and problem_type == "none":
        flags.append("sentiment_problem_mismatch")
    if sentiment == "positive" and action_priority == "high":
        flags.append("sentiment_priority_mismatch")
    return flags
