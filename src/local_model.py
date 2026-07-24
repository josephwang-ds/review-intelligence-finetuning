"""
local_model.py — real local inference for the QLoRA-fine-tuned Qwen2.5-1.5B model.

Different job from 06_evaluate_finetuned.py: that script does offline batch eval
on a GPU-appropriate box and reports accuracy; this module does lazy, single-request
inference for the router (src/router.py) to call at serving time. Loading logic is
the same transformers+peft fallback path 06_evaluate_finetuned.py already uses —
the unsloth branch there is skipped here entirely: unsloth isn't installed in this
environment and is CUDA-oriented, so it wouldn't help on Apple Silicon anyway.

The model loads lazily (first real call only) and is cached at module level —
importing this module, or even starting the API, never triggers the ~3GB base
model download by itself.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch

from prompts import PROMPT_VERSION, build_operational_user_message, build_system_prompt
from schema import ASAP_PROFILE, OperationalFields
from structured_client import extract_json

ROOT = Path(__file__).parent.parent
ADAPTER_PATH = ROOT / "models" / "qwen-asap-qlora" / "final"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_cache: dict = {}


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_local_model():
    """Lazy singleton. Downloads BASE_MODEL from HF on first call (cached by HF
    afterward), merges the local LoRA adapter, and caches the merged model."""
    if "model" in _cache:
        return _cache["model"], _cache["tokenizer"]

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _device()
    dtype = torch.float16 if device != "cpu" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype, trust_remote_code=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    model = model.merge_and_unload()
    model.eval()

    _cache["model"] = model
    _cache["tokenizer"] = tokenizer
    _cache["device"] = device
    return model, tokenizer


def is_loaded() -> bool:
    return "model" in _cache


def predict_operational(text: str, aspect_context: Optional[dict] = None) -> tuple[Optional[OperationalFields], dict]:
    """Same (result, meta) shape as structured_client.call_structured, so the
    router can treat this and the DeepSeek path interchangeably. Never raises —
    load/generation/validation failures come back as (None, meta-with-error) so
    the router can decide to fall back."""
    start = time.time()
    try:
        model, tokenizer = load_local_model()
        device = _cache["device"]
    except Exception as e:
        return None, {
            "latency_ms": round((time.time() - start) * 1000),
            "valid": False, "repaired": False, "error": f"model_load_failed: {e}",
            "prompt_version": PROMPT_VERSION,
        }

    system_prompt = build_system_prompt(ASAP_PROFILE, mode="operational")
    user_message = build_operational_user_message(text, aspect_context or {}, ASAP_PROFILE, max_chars=300)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids, max_new_tokens=64, temperature=0.1,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    except Exception as e:
        return None, {
            "latency_ms": round((time.time() - start) * 1000),
            "valid": False, "repaired": False, "error": f"generation_failed: {e}",
            "prompt_version": PROMPT_VERSION,
        }

    latency_ms = round((time.time() - start) * 1000)
    parsed = extract_json(raw)
    if parsed is None:
        return None, {
            "latency_ms": latency_ms, "valid": False, "repaired": False,
            "error": f"json_parse_failed: {raw!r}", "prompt_version": PROMPT_VERSION,
        }

    try:
        result = OperationalFields.model_validate(parsed, context={"profile": ASAP_PROFILE})
    except Exception as e:
        return None, {
            "latency_ms": latency_ms, "valid": False, "repaired": False,
            "error": str(e), "prompt_version": PROMPT_VERSION,
        }

    return result, {
        "latency_ms": latency_ms, "valid": True, "repaired": False, "error": None,
        "prompt_version": PROMPT_VERSION,
    }
