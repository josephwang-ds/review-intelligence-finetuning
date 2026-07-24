"""Unit tests for request_log.py — uses a tmp_path DB, mirrors test_review_queue.py."""

import pytest

import request_log


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(request_log, "DB_PATH", tmp_path / "test_request_log.db")


def test_log_request_appears_in_recent():
    req_id = request_log.log_request(
        dataset="asap", mode="few_shot", text_length=42, latency_ms=1200,
        valid=True, repaired=False, flags=[], model="deepseek-chat",
    )
    recent = request_log.recent()
    assert any(r["id"] == req_id for r in recent)


def test_stats_aggregate_correctly():
    request_log.log_request(
        dataset="asap", mode="few_shot", text_length=10, latency_ms=1000,
        valid=True, repaired=False, flags=[], model="deepseek-chat",
    )
    request_log.log_request(
        dataset="yelp", mode="zero_shot", text_length=20, latency_ms=2000,
        valid=False, repaired=True, flags=["pii_phone"], model="deepseek-chat",
    )
    stats = request_log.stats()
    assert stats["n_requests"] == 2
    assert stats["avg_latency_ms"] == 1500.0
    assert stats["n_invalid"] == 1
    assert stats["n_repaired"] == 1


def test_stats_on_empty_db():
    stats = request_log.stats()
    assert stats == {"n_requests": 0, "avg_latency_ms": 0, "n_invalid": 0, "n_repaired": 0}


def _seed_mixed_rows():
    request_log.log_request(dataset="asap", mode="local_finetuned", text_length=10,
                             latency_ms=100, valid=True, repaired=False, flags=[],
                             model=request_log.LOCAL_MODEL_NAME)
    request_log.log_request(dataset="asap", mode="local_finetuned", text_length=12,
                             latency_ms=200, valid=True, repaired=False, flags=[],
                             model=request_log.LOCAL_MODEL_NAME)
    request_log.log_request(dataset="asap", mode="few_shot_escalated_long_input", text_length=300,
                             latency_ms=3000, valid=True, repaired=False, flags=[],
                             model="deepseek-chat")
    request_log.log_request(dataset="yelp", mode="few_shot_non_asap_profile", text_length=50,
                             latency_ms=4000, valid=True, repaired=False, flags=["pii_phone"],
                             model="deepseek-chat")


def test_route_distribution_counts_by_mode():
    _seed_mixed_rows()
    assert request_log.route_distribution() == {
        "local_finetuned": 2,
        "few_shot_escalated_long_input": 1,
        "few_shot_non_asap_profile": 1,
    }


def test_latency_stats_overall_and_by_model():
    _seed_mixed_rows()
    stats = request_log.latency_stats()
    assert stats["overall"]["n"] == 4
    assert stats["overall"]["avg_ms"] == 1825.0
    assert stats["overall"]["p50_ms"] == 1600.0
    assert stats["by_model"][request_log.LOCAL_MODEL_NAME] == {"n": 2, "avg_ms": 150.0, "p50_ms": 150.0, "p95_ms": 195.0}
    assert stats["by_model"]["deepseek-chat"]["n"] == 2


def test_cost_estimate_splits_local_vs_api():
    _seed_mixed_rows()
    cost = request_log.cost_estimate()
    assert cost["n_local"] == 2
    assert cost["n_api"] == 2
    assert cost["estimated_cost_usd"] == round(2 * 0.00005 + 2 * 0.001, 4)


def test_guardrail_trigger_rate():
    _seed_mixed_rows()
    assert request_log.guardrail_trigger_rate() == 0.25


def test_guardrail_trigger_rate_zero_on_empty_db():
    assert request_log.guardrail_trigger_rate() == 0.0
