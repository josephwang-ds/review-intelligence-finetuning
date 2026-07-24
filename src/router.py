"""
router.py — model routing decision logic.

Deliberately has ZERO dependency on torch/transformers/openai: local_predict_fn
and api_predict_fn are injected by the caller (real functions in production,
fakes in tests) rather than imported here as defaults. That's what keeps this
file dependency-light and instantly unit-testable.

The routing decision has two real inputs, not a fabricated confidence score:
1. Whether the caller's request is even eligible for the cheap path — the local
   fine-tuned model only ever produces 3 fields (problem_type/action_priority/
   operator_action), so only "operational"-scope requests on the ASAP profile
   qualify (the model was only trained on ASAP).
2. Input length as a complexity proxy — short/simple reviews stay on the cheap
   specialist (its documented accuracy, 0.65-0.74, is fine for the easy majority);
   longer/more complex reviews escalate to the few-shot call even though only
   3 fields were asked for, because a harder review is more likely to need the
   larger model's reasoning.

Plus a reliability fallback: if the local path raises, returns None, or its
result fails validation, the router falls back to the few-shot path instead of
surfacing an error — a fast path that can fail has to degrade gracefully.
"""

from __future__ import annotations

from typing import Callable, Optional

from schema import DatasetProfile

LOCAL_MODEL_MAX_CHARS = 150


def route_and_predict(
    client,
    profile: DatasetProfile,
    text: str,
    local_predict_fn: Callable[..., tuple],
    api_predict_fn: Callable[..., tuple],
    few_shot_examples: Optional[list] = None,
    aspect_context: Optional[dict] = None,
) -> tuple:
    """Returns (result, meta, route_label)."""

    if profile.name != "asap":
        result, meta = api_predict_fn(client, profile, text, mode="full", few_shot=few_shot_examples)
        return result, meta, "few_shot_non_asap_profile"

    if len(text) > LOCAL_MODEL_MAX_CHARS:
        result, meta = api_predict_fn(client, profile, text, mode="full", few_shot=few_shot_examples)
        return result, meta, "few_shot_escalated_long_input"

    try:
        result, meta = local_predict_fn(text, aspect_context=aspect_context)
    except Exception as e:
        result, meta = None, {"valid": False, "error": f"local_predict_raised: {e}"}

    if result is not None and meta.get("valid"):
        return result, meta, "local_finetuned"

    # Local path unavailable or its output didn't validate — fall back rather than error out.
    result, meta = api_predict_fn(client, profile, text, mode="full", few_shot=few_shot_examples)
    return result, meta, "few_shot_fallback_local_failed"
