"""Unit tests for review_queue.py — uses a tmp_path DB so the real
runtime/review_queue.db is never touched by the test suite."""

import pytest

import review_queue


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(review_queue, "DB_PATH", tmp_path / "test_review_queue.db")


def test_enqueue_appears_in_pending():
    item_id = review_queue.enqueue(
        dataset="asap", method="zero_shot", review_text="测试评论",
        prediction={"sentiment": "negative", "problem_type": "none"},
        reasons=["sentiment_problem_mismatch"],
    )
    pending = review_queue.list_pending()
    assert any(p["id"] == item_id for p in pending)


def test_resolve_corrected_removes_from_pending():
    item_id = review_queue.enqueue(
        dataset="asap", method="zero_shot", review_text="测试评论",
        prediction={"sentiment": "negative", "problem_type": "none"},
        reasons=["sentiment_problem_mismatch"],
    )
    review_queue.resolve(
        item_id, "corrected",
        corrected_json={"sentiment": "negative", "problem_type": "poor_service"},
        note="fixed by reviewer",
    )
    pending = review_queue.list_pending()
    assert not any(p["id"] == item_id for p in pending)


def test_resolve_rejects_invalid_status():
    item_id = review_queue.enqueue(
        dataset="asap", method="zero_shot", review_text="x",
        prediction={}, reasons=["too_long"],
    )
    with pytest.raises(ValueError):
        review_queue.resolve(item_id, "pending")


def test_queue_stats_counts_update_after_resolve():
    item_id = review_queue.enqueue(
        dataset="yelp", method="few_shot", review_text="x",
        prediction={}, reasons=["too_long"],
    )
    assert review_queue.queue_stats()["pending"] == 1
    review_queue.resolve(item_id, "approved")
    stats = review_queue.queue_stats()
    assert stats["pending"] == 0
    assert stats["approved"] == 1
