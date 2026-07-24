"""
request_log.py — raw request log for the FastAPI service (runtime/request_log.db)

Unlike review_queue.py (which only stores *flagged* items), this logs every
request that hits POST /analyze. That's what gives a real denominator for
metrics like "guardrail trigger rate as % of traffic" — Phase 3's eval harness
explicitly deferred that metric for exactly this reason (no total-request log
existed yet). This is the piece Phase 6's monitoring dashboard reads from.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "runtime" / "request_log.db"

# api/main.py stamps this into the `model` column when a request was actually
# served by the local fine-tuned model (src/local_model.py) rather than DeepSeek.
# Defined here (not in local_model.py) so this module — and anything that reads
# request_log.db, like the aggregations below — never has to import torch.
LOCAL_MODEL_NAME = "qwen2.5-1.5b-qlora-local"

# Mirrors 06_evaluate_finetuned.py's compute_breakeven() defaults (that function
# is the source of truth for these numbers). Not imported directly because that
# script's module-level setup pulls in torch/peft, which this lightweight
# SQLite module — and everything that depends on it, including the fast test
# suite — has no reason to carry.
API_COST_PER_QUERY_USD = 0.001
LOCAL_COST_PER_QUERY_USD = 0.00005

_SCHEMA = """
CREATE TABLE IF NOT EXISTS request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    dataset TEXT NOT NULL,
    mode TEXT NOT NULL,
    text_length INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    valid INTEGER NOT NULL,
    repaired INTEGER NOT NULL,
    flags_json TEXT NOT NULL,
    model TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def init_db() -> None:
    _connect().close()


def log_request(
    dataset: str, mode: str, text_length: int, latency_ms: int,
    valid: bool, repaired: bool, flags: list[str], model: str,
) -> int:
    conn = _connect()
    try:
        cur = conn.execute(
            """INSERT INTO request_log
               (created_at, dataset, mode, text_length, latency_ms, valid, repaired, flags_json, model)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(), dataset, mode, text_length, latency_ms,
                int(valid), int(repaired), json.dumps(flags, ensure_ascii=False), model,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def recent(limit: int = 50) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM request_log ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def stats() -> dict:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n, AVG(latency_ms) AS avg_latency_ms, "
            "SUM(1 - valid) AS n_invalid, SUM(repaired) AS n_repaired FROM request_log"
        ).fetchone()
    finally:
        conn.close()
    n = row["n"] or 0
    return {
        "n_requests": n,
        "avg_latency_ms": round(row["avg_latency_ms"], 1) if row["avg_latency_ms"] else 0,
        "n_invalid": row["n_invalid"] or 0,
        "n_repaired": row["n_repaired"] or 0,
    }


def route_distribution() -> dict[str, int]:
    """Count of requests per `mode` value. For mode="operational" requests this
    is the real route label from router.py (local_finetuned / few_shot_escalated_
    long_input / ...), not a generic "operational" bucket."""
    conn = _connect()
    try:
        rows = conn.execute("SELECT mode, COUNT(*) AS n FROM request_log GROUP BY mode").fetchall()
    finally:
        conn.close()
    return {r["mode"]: r["n"] for r in rows}


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _latency_summary(values: list[int]) -> dict:
    if not values:
        return {"n": 0, "avg_ms": 0, "p50_ms": 0, "p95_ms": 0}
    s = sorted(values)
    return {
        "n": len(s),
        "avg_ms": round(sum(s) / len(s), 1),
        "p50_ms": round(_percentile(s, 0.50), 1),
        "p95_ms": round(_percentile(s, 0.95), 1),
    }


def latency_stats() -> dict:
    """Overall latency + broken out by `model` (local vs. DeepSeek) — the concrete
    speed number behind Phase 5's cheap/expensive routing trade-off."""
    conn = _connect()
    try:
        rows = conn.execute("SELECT model, latency_ms FROM request_log").fetchall()
    finally:
        conn.close()

    by_model: dict[str, list[int]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r["latency_ms"])

    return {
        "overall": _latency_summary([r["latency_ms"] for r in rows]),
        "by_model": {model: _latency_summary(vals) for model, vals in by_model.items()},
    }


def cost_estimate() -> dict:
    """Real request counts × the same per-query cost assumptions already in the
    README's break-even analysis — turns that one-time calculation into a live number."""
    conn = _connect()
    try:
        rows = conn.execute("SELECT model, COUNT(*) AS n FROM request_log GROUP BY model").fetchall()
    finally:
        conn.close()

    n_local = sum(r["n"] for r in rows if r["model"] == LOCAL_MODEL_NAME)
    n_api = sum(r["n"] for r in rows if r["model"] != LOCAL_MODEL_NAME)
    estimated_cost_usd = n_local * LOCAL_COST_PER_QUERY_USD + n_api * API_COST_PER_QUERY_USD
    return {
        "n_local": n_local,
        "n_api": n_api,
        "estimated_cost_usd": round(estimated_cost_usd, 4),
    }


def guardrail_trigger_rate() -> float:
    """Flagged requests ÷ total requests. Phase 3's eval harness explicitly
    could not compute this — review_queue.db only stores flagged items, so
    there was no denominator until this table existed."""
    conn = _connect()
    try:
        rows = conn.execute("SELECT flags_json FROM request_log").fetchall()
    finally:
        conn.close()
    if not rows:
        return 0.0
    n_flagged = sum(1 for r in rows if json.loads(r["flags_json"]))
    return round(n_flagged / len(rows), 4)
