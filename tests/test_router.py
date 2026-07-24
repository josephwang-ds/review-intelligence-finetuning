"""Unit tests for router.py's decision logic — no torch, no network. Both
local_predict_fn and api_predict_fn are fakes injected by the test, so this
exercises exactly the routing decision, not the underlying model calls."""

from schema import ASAP_PROFILE, YELP_PROFILE, OperationalFields
from router import LOCAL_MODEL_MAX_CHARS, route_and_predict

SHORT_ASAP_TEXT = "服务态度差"
LONG_ASAP_TEXT = "a" * (LOCAL_MODEL_MAX_CHARS + 1)

_OPERATIONAL_RESULT = OperationalFields.model_validate(
    {"problem_type": "poor_service", "action_priority": "high", "operator_action": "train_service"},
    context={"profile": ASAP_PROFILE},
)
_FULL_RESULT = "fake_full_result"  # router treats it opaquely, doesn't need to be a real model


def _fake_local_ok(text, aspect_context=None):
    return _OPERATIONAL_RESULT, {"latency_ms": 50, "valid": True, "repaired": False, "error": None}


def _fake_local_fails(text, aspect_context=None):
    return None, {"latency_ms": 10, "valid": False, "repaired": False, "error": "model_load_failed: no gpu"}


def _fake_local_raises(text, aspect_context=None):
    raise RuntimeError("boom")


def _fake_api(client, profile, text, mode, few_shot=None):
    return _FULL_RESULT, {"latency_ms": 1500, "valid": True, "repaired": False, "error": None, "prompt_version": "v-test"}


def test_short_asap_text_routes_to_local():
    result, meta, route = route_and_predict(
        client=None, profile=ASAP_PROFILE, text=SHORT_ASAP_TEXT,
        local_predict_fn=_fake_local_ok, api_predict_fn=_fake_api,
    )
    assert route == "local_finetuned"
    assert result is _OPERATIONAL_RESULT


def test_long_asap_text_escalates():
    result, meta, route = route_and_predict(
        client=None, profile=ASAP_PROFILE, text=LONG_ASAP_TEXT,
        local_predict_fn=_fake_local_ok, api_predict_fn=_fake_api,
    )
    assert route == "few_shot_escalated_long_input"
    assert result == _FULL_RESULT


def test_yelp_text_always_escalates_regardless_of_length():
    result, meta, route = route_and_predict(
        client=None, profile=YELP_PROFILE, text=SHORT_ASAP_TEXT,
        local_predict_fn=_fake_local_ok, api_predict_fn=_fake_api,
    )
    assert route == "few_shot_non_asap_profile"
    assert result == _FULL_RESULT


def test_local_failure_falls_back_to_api():
    result, meta, route = route_and_predict(
        client=None, profile=ASAP_PROFILE, text=SHORT_ASAP_TEXT,
        local_predict_fn=_fake_local_fails, api_predict_fn=_fake_api,
    )
    assert route == "few_shot_fallback_local_failed"
    assert result == _FULL_RESULT


def test_local_exception_falls_back_to_api():
    result, meta, route = route_and_predict(
        client=None, profile=ASAP_PROFILE, text=SHORT_ASAP_TEXT,
        local_predict_fn=_fake_local_raises, api_predict_fn=_fake_api,
    )
    assert route == "few_shot_fallback_local_failed"
    assert result == _FULL_RESULT
