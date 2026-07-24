"""Golden regression set — runs the real few-shot pipeline (the production-recommended
path per app.py's own decision guidance) against ~30 curated cases and asserts
category-specific invariants. See eval/golden_set.jsonl for the cases themselves.

This is deliberately NOT the same thing as 03_run_baselines.py's 200-sample benchmark:
that measures aggregate model quality on a random sample; this is a small, curated,
adversarial/edge-case suite meant to catch a *behavior regression* when someone edits
prompts.py or a DatasetProfile in schema.py.
"""

import json
from pathlib import Path

import pytest

from guardrails import check_input, redact_pii
from prompts import FEW_SHOT_BANK
from schema import PROFILES
from structured_client import call_structured

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"


def _load_golden_set() -> list[dict]:
    cases = []
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


GOLDEN_CASES = _load_golden_set()


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
def test_golden_case(case, deepseek_client, golden_results):
    profile = PROFILES[case["dataset"]]
    category = case["category"]
    text = case["review_text"]
    passed, latency_ms, repaired, error = False, None, None, None

    try:
        if category == "prompt_injection":
            flags = check_input(text)
            assert "prompt_injection_suspected" in flags, f"expected injection heuristic to fire on: {text!r}"
            result, meta = call_structured(deepseek_client, profile, text, mode="full", few_shot=FEW_SHOT_BANK[profile.name])
            latency_ms, repaired = meta["latency_ms"], meta["repaired"]
            assert result is not None, "output must still pass schema validation despite the injection attempt"

        elif category == "pii_present":
            redacted, found = redact_pii(text)
            assert case["pii_type"] in found, f"expected {case['pii_type']} to be detected in: {text!r}"
            assert redacted != text, "expected redact_pii to actually change the text"
            result, meta = call_structured(deepseek_client, profile, redacted, mode="full", few_shot=FEW_SHOT_BANK[profile.name])
            latency_ms, repaired = meta["latency_ms"], meta["repaired"]
            assert result is not None

        elif category == "regression_location_issue":
            # Guards against the Phase 1 bug: 03_run_baselines.py's prompt used to be
            # missing location_issue/packaging_issue from problem_type entirely.
            assert "location_issue" in profile.problem_types
            result, meta = call_structured(deepseek_client, profile, text, mode="full", few_shot=FEW_SHOT_BANK[profile.name])
            latency_ms, repaired = meta["latency_ms"], meta["repaired"]
            assert result is not None

        elif category in ("clear_positive", "clear_negative"):
            result, meta = call_structured(deepseek_client, profile, text, mode="full", few_shot=FEW_SHOT_BANK[profile.name])
            latency_ms, repaired = meta["latency_ms"], meta["repaired"]
            assert result is not None
            assert result.sentiment == case["expected_sentiment"], f"expected {case['expected_sentiment']}, got {result.sentiment}"
            if category == "clear_negative":
                assert result.problem_type != "none", "a clear complaint should not resolve to problem_type=none"
            else:
                assert result.action_priority != "high", "a clearly positive review should not be flagged high priority"

        else:
            # mixed_neutral, sarcasm, short_input, long_input, non_target_language:
            # known-hard / ambiguous by construction — only require graceful, schema-valid output.
            result, meta = call_structured(deepseek_client, profile, text, mode="full", few_shot=FEW_SHOT_BANK[profile.name])
            latency_ms, repaired = meta["latency_ms"], meta["repaired"]
            assert result is not None, "edge case must still degrade gracefully to a schema-valid output"

        passed = True
    except AssertionError as e:
        error = str(e)
        raise
    finally:
        golden_results.append({
            "id": case["id"], "category": category, "passed": passed,
            "latency_ms": latency_ms, "repaired": repaired, "error": error,
        })
