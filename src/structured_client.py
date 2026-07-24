"""
structured_client.py — schema 校验的 LLM 调用层

与旧的 parse_llm_output()（regex 提取 JSON，提取失败就放弃）相比：
1. 用 DeepSeek 的 response_format=json_object 强制模型输出 JSON
2. 用 Pydantic 针对具体 DatasetProfile 校验取值（不止是"合法 JSON"，而是"合法的这份 schema"）
3. 校验失败时把错误信息喂回模型，做最多 max_repairs 次修复重试，而不是直接判失败
"""

from __future__ import annotations

import json
import re
import time
from typing import Literal, Optional, Type

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from prompts import PROMPT_VERSION, FewShotExample, build_operational_user_message, build_system_prompt, build_user_message
from schema import DatasetProfile, FullReviewAnalysis, OperationalFields

Mode = Literal["full", "operational"]

_MODEL_FOR_MODE: dict[str, Type[BaseModel]] = {
    "full": FullReviewAnalysis,
    "operational": OperationalFields,
}


def extract_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


def call_structured(
    client: OpenAI,
    profile: DatasetProfile,
    text: str,
    mode: Mode = "full",
    few_shot: Optional[list[FewShotExample]] = None,
    aspect_context: Optional[dict] = None,
    system_prompt: Optional[str] = None,
    model: str = "deepseek-chat",
    temperature: float = 0.1,
    max_tokens: int = 350,
    max_repairs: int = 1,
) -> tuple[Optional[BaseModel], dict]:
    """返回 (validated_model_or_None, meta)。

    meta = {latency_ms, valid, repaired, error, prompt_version}

    aspect_context: 传入已知 aspect_sentiments 时（silver 标注 / 微调模型评测），
      user message 会带上这份上下文；不传时走纯 review 文本格式（baseline / persona 模拟）。
    system_prompt: 默认由 profile + mode 生成；传入时覆盖（用于 app.py 的「模拟微调模型」人设 prompt）。
    """
    schema_model = _MODEL_FOR_MODE[mode]
    system_prompt = system_prompt or build_system_prompt(profile, mode=mode)
    if aspect_context is not None:
        user_message = build_operational_user_message(text, aspect_context, profile)
    else:
        user_message = build_user_message(text, profile, few_shot=few_shot)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    start = time.time()
    repaired = False
    last_error: Optional[str] = None

    for attempt in range(max_repairs + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
        except Exception as e:
            last_error = f"api_error: {e}"
            break

        parsed = extract_json(raw)
        if parsed is None:
            last_error = "json_parse_failed"
        else:
            try:
                result = schema_model.model_validate(parsed, context={"profile": profile})
                return result, {
                    "latency_ms": round((time.time() - start) * 1000),
                    "valid": True,
                    "repaired": repaired,
                    "error": None,
                    "prompt_version": PROMPT_VERSION,
                }
            except ValidationError as e:
                last_error = str(e)

        if attempt < max_repairs:
            repaired = True
            repair_note = (
                f"上一次输出未通过校验：{last_error}\n请只返回修正后的合法 JSON，不要其他文字。"
                if profile.lang == "zh"
                else f"Your previous output failed validation: {last_error}\nReturn only the corrected JSON, no other text."
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": repair_note})

    return None, {
        "latency_ms": round((time.time() - start) * 1000),
        "valid": False,
        "repaired": repaired,
        "error": last_error,
        "prompt_version": PROMPT_VERSION,
    }
