"""
api/main.py — FastAPI service wrapping the engine (schema/prompts/structured_client)
and guardrails built in Phases 1-3. This is the "how would this actually get
deployed" artifact — see ARCHITECTURE.md for the full design discussion.

Run:   uvicorn api.main:app --reload --port 8000
Docs:  http://localhost:8000/docs (auto-generated from the Pydantic models below)

Note: app.py (the Streamlit demo) does NOT call this service — it still calls
DeepSeek directly, exactly as before. This is an additive, independently
deployable artifact, not a migration. See ARCHITECTURE.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

import request_log
import review_queue
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from guardrails import check_input, check_output_consistency, redact_pii
from prompts import FEW_SHOT_BANK, PROMPT_VERSION
from router import route_and_predict
from schema import PROFILES
from structured_client import call_structured

# local_model 依赖 torch/transformers（requirements-api.txt 里的重依赖）。
# 只有 mode="operational" 的路由快路径需要它，mode="full" 完全用不上，
# 所以这里软导入：没装 torch 的环境（比如只跑 requirements.txt 的 CI）依然能
# 起服务、跑 /analyze 的完整分析，只是 operational 请求会全部走 API 兜底。
try:
    import local_model
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    local_model = None
    LOCAL_MODEL_AVAILABLE = False

app = FastAPI(title="Review Intelligence API", version=PROMPT_VERSION)

_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL) if DEEPSEEK_API_KEY else None


class AnalyzeRequest(BaseModel):
    text: str
    dataset: Literal["asap", "yelp"] = "asap"
    few_shot: bool = True
    mode: Literal["full", "operational"] = "full"


class AnalyzeResponse(BaseModel):
    sentiment: Optional[str] = None
    rating_prediction: Optional[int] = None
    aspect_sentiments: Optional[dict] = None
    problem_type: str
    action_priority: str
    operator_action: str
    valid: bool
    repaired: bool
    flags: list[str]
    latency_ms: int
    prompt_version: str
    route: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "deepseek_configured": _client is not None,
        "local_model_available": LOCAL_MODEL_AVAILABLE,
        "local_model_loaded": local_model.is_loaded() if LOCAL_MODEL_AVAILABLE else False,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if _client is None:
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY not configured")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    profile = PROFILES[req.dataset]

    # 护栏：脱敏后的文本才会真正发给模型/存进队列，不只是展示层打码
    safe_text, pii_types = redact_pii(req.text)
    flags = check_input(req.text) + [f"pii_{t.lower()}" for t in pii_types]

    few_shot_examples = FEW_SHOT_BANK[profile.name] if req.few_shot else None

    if req.mode == "operational":
        # torch 没装时传一个必然失败的 local fn，router 会自然走它已有的
        # few_shot_fallback_local_failed 兜底分支——不用为这种情况另开一条代码路径。
        def _unavailable(text, aspect_context=None):
            raise RuntimeError("local model unavailable: torch/transformers not installed")

        result, meta, route = route_and_predict(
            _client, profile, safe_text,
            local_predict_fn=local_model.predict_operational if LOCAL_MODEL_AVAILABLE else _unavailable,
            api_predict_fn=call_structured,
            few_shot_examples=few_shot_examples,
        )
        mode_label = route
        model_used = request_log.LOCAL_MODEL_NAME if route == "local_finetuned" else DEEPSEEK_MODEL
    else:
        result, meta = call_structured(_client, profile, safe_text, mode="full", few_shot=few_shot_examples)
        route = None
        mode_label = "few_shot" if req.few_shot else "zero_shot"
        model_used = DEEPSEEK_MODEL

    valid = result is not None
    payload = result.model_dump() if valid else {}
    if not valid:
        flags.append("schema_validation_failed")
    else:
        flags += check_output_consistency(payload)

    if flags:
        review_queue.enqueue(
            dataset=profile.name, method=f"api_{mode_label}",
            review_text=safe_text, prediction=payload, reasons=flags,
        )

    request_log.log_request(
        dataset=profile.name, mode=mode_label, text_length=len(req.text),
        latency_ms=meta["latency_ms"], valid=valid, repaired=meta["repaired"],
        flags=flags, model=model_used,
    )

    if not valid:
        raise HTTPException(status_code=502, detail=f"model output failed schema validation: {meta['error']}")

    return AnalyzeResponse(
        sentiment=payload.get("sentiment"),
        rating_prediction=payload.get("rating_prediction"),
        aspect_sentiments=payload.get("aspect_sentiments"),
        problem_type=payload["problem_type"],
        action_priority=payload["action_priority"],
        operator_action=payload["operator_action"],
        valid=valid,
        repaired=meta["repaired"],
        flags=flags,
        latency_ms=meta["latency_ms"],
        prompt_version=meta["prompt_version"],
        route=route,
    )


@app.get("/stats")
def stats():
    return {"review_queue": review_queue.queue_stats(), "requests": request_log.stats()}
