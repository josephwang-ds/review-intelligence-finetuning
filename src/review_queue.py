"""
review_queue.py — 人工复核队列（SQLite）

被 guardrails.py 标记的预测结果落在这里，pages/1_Review_Queue.py 提供人工
approve / correct / reject 的界面。correct 时写回的 corrected_json 就是
下一轮 QLoRA 微调可以直接用的 hard negative 样本——见 05_prepare_finetune_data.py。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "runtime" / "review_queue.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    dataset TEXT NOT NULL,
    method TEXT NOT NULL,
    review_text TEXT NOT NULL,
    prediction_json TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    corrected_json TEXT,
    reviewer_note TEXT,
    reviewed_at TEXT
);
"""

VALID_STATUSES = {"pending", "approved", "corrected", "rejected"}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def init_db() -> None:
    _connect().close()


def enqueue(dataset: str, method: str, review_text: str, prediction: dict, reasons: list[str]) -> int:
    conn = _connect()
    try:
        cur = conn.execute(
            """INSERT INTO review_queue
               (created_at, dataset, method, review_text, prediction_json, reasons_json, status)
               VALUES (?, ?, ?, ?, ?, ?, 'pending')""",
            (
                datetime.now(timezone.utc).isoformat(),
                dataset, method, review_text,
                json.dumps(prediction, ensure_ascii=False),
                json.dumps(reasons, ensure_ascii=False),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_pending() -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM review_queue WHERE status = 'pending' ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def resolve(item_id: int, status: str, corrected_json: Optional[dict] = None, note: Optional[str] = None) -> None:
    if status not in VALID_STATUSES - {"pending"}:
        raise ValueError(f"status must be one of {VALID_STATUSES - {'pending'}}, got {status!r}")
    conn = _connect()
    try:
        conn.execute(
            """UPDATE review_queue
               SET status = ?, corrected_json = ?, reviewer_note = ?, reviewed_at = ?
               WHERE id = ?""",
            (
                status,
                json.dumps(corrected_json, ensure_ascii=False) if corrected_json is not None else None,
                note,
                datetime.now(timezone.utc).isoformat(),
                item_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def queue_stats() -> dict:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM review_queue GROUP BY status"
        ).fetchall()
    finally:
        conn.close()
    stats = {s: 0 for s in VALID_STATUSES}
    stats.update({r["status"]: r["n"] for r in rows})
    return stats
