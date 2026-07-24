"""Fixtures + reporting for the golden regression set.

Kept separate from tests/ on purpose: everything here hits the real DeepSeek
API, so it's gated on DEEPSEEK_API_KEY and skips cleanly without it instead
of failing a key-less CI run.
"""

import json
from pathlib import Path

import pytest
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"

_results: list[dict] = []


@pytest.fixture(scope="session")
def deepseek_client():
    if not DEEPSEEK_API_KEY:
        pytest.skip("requires DEEPSEEK_API_KEY to run the golden regression set against the real API")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


@pytest.fixture(scope="session")
def golden_results():
    """Shared list tests append {id, category, passed, latency_ms, repaired, error} to.
    Same object pytest_sessionfinish below writes out — no cross-file import needed."""
    return _results


def pytest_sessionfinish(session, exitstatus):
    if not _results:
        return
    REPORTS_DIR.mkdir(exist_ok=True)
    by_category: dict[str, dict] = {}
    for r in _results:
        c = by_category.setdefault(r["category"], {"n": 0, "n_passed": 0})
        c["n"] += 1
        c["n_passed"] += 1 if r["passed"] else 0
    summary = {
        "n": len(_results),
        "n_passed": sum(1 for r in _results if r["passed"]),
        "by_category": by_category,
        "results": _results,
    }
    with open(REPORTS_DIR / "eval_golden_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
