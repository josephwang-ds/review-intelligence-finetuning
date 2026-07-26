"""Unit tests for structured_client.call_structured — no network (the OpenAI
client is a fake), covering the failure/repair paths that matter most."""

import json

from schema import ASAP_PROFILE
from structured_client import call_structured, extract_json

VALID_PAYLOAD = {
    "sentiment": "negative", "rating_prediction": 1, "aspect_sentiments": {},
    "problem_type": "poor_service", "action_priority": "high", "operator_action": "train_service",
}


class _FakeMsg:
    def __init__(self, content): self.content = content


class _FakeChoice:
    def __init__(self, content): self.message = _FakeMsg(content)


class _FakeResp:
    def __init__(self, content): self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs["messages"])
        return _FakeResp(self.responses.pop(0))


class _FakeClient:
    def __init__(self, responses):
        self.chat = type("C", (), {"completions": _FakeCompletions(responses)})()


def test_valid_response_returns_immediately():
    client = _FakeClient([json.dumps(VALID_PAYLOAD)])
    result, meta = call_structured(client, ASAP_PROFILE, "测试", mode="full")
    assert result is not None and meta["valid"] and not meta["repaired"]
    assert len(client.chat.completions.calls) == 1


def test_invalid_enum_triggers_repair_then_succeeds():
    bad = json.dumps(dict(VALID_PAYLOAD, problem_type="NOT_A_REAL_VALUE"))
    client = _FakeClient([bad, json.dumps(VALID_PAYLOAD)])
    result, meta = call_structured(client, ASAP_PROFILE, "测试", mode="full", max_repairs=1)
    assert result is not None and meta["valid"] and meta["repaired"]
    assert len(client.chat.completions.calls) == 2


def test_gives_up_cleanly_after_exhausting_repairs():
    bad = json.dumps(dict(VALID_PAYLOAD, problem_type="STILL_BAD"))
    client = _FakeClient([bad, bad])
    result, meta = call_structured(client, ASAP_PROFILE, "测试", mode="full", max_repairs=1)
    assert result is None and not meta["valid"] and meta["error"]


def test_non_json_response_is_handled_not_raised():
    client = _FakeClient(["I'm sorry, I can't help with that.", json.dumps(VALID_PAYLOAD)])
    result, meta = call_structured(client, ASAP_PROFILE, "测试", mode="full", max_repairs=1)
    assert result is not None and meta["repaired"]


def test_unexpected_shape_does_not_escape_as_exception():
    """Regression: a before-validator hitting an unforeseen shape used to raise
    TypeError straight through Pydantic and crash the caller (this is exactly
    what took down the 200-sample benchmark run when the model changed)."""
    weird = json.dumps(dict(VALID_PAYLOAD, aspect_sentiments=[{"nested": {"a": "b"}}]))
    client = _FakeClient([weird])
    result, meta = call_structured(client, ASAP_PROFILE, "测试", mode="full", max_repairs=0)
    # Must return normally either way — the point is it doesn't raise.
    assert meta["valid"] in (True, False)


def test_api_exception_is_captured_in_meta():
    class _Boom:
        def create(self, **kwargs):
            raise RuntimeError("connection reset")

    client = type("C", (), {"chat": type("D", (), {"completions": _Boom()})()})()
    result, meta = call_structured(client, ASAP_PROFILE, "测试", mode="full")
    assert result is None and not meta["valid"] and "api_error" in meta["error"]


def test_extract_json_strips_markdown_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('here you go: {"a": 1} hope that helps') == {"a": 1}
    assert extract_json("no json at all") is None
