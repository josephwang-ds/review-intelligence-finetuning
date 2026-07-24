"""Structural tests for api/main.py — request validation only, never reaches the
point of calling DeepSeek, so these need no API key and no network."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
from main import app

client = TestClient(app)


def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["deepseek_configured"], bool)


def test_analyze_rejects_missing_text_field():
    resp = client.post("/analyze", json={"dataset": "asap"})
    assert resp.status_code == 422


def test_analyze_rejects_empty_text():
    resp = client.post("/analyze", json={"text": "   ", "dataset": "asap"})
    assert resp.status_code == 422


def test_analyze_rejects_unknown_dataset():
    resp = client.post("/analyze", json={"text": "还不错", "dataset": "google_reviews"})
    assert resp.status_code == 422
