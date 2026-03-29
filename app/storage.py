from __future__ import annotations

from pathlib import Path
import json
import sqlite3
from contextlib import closing
from typing import Any


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS recommendation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                mode TEXT,
                request_json TEXT,
                response_json TEXT
            )
            '''
        )
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS feedback_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                item_id TEXT,
                event_type TEXT,
                value REAL,
                context_json TEXT
            )
            '''
        )
        conn.commit()


def log_recommendation(db_path: Path, user_id: str, mode: str, request_payload: dict[str, Any], response_payload: list[dict[str, Any]]) -> None:
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            '''
            INSERT INTO recommendation_logs (user_id, mode, request_json, response_json)
            VALUES (?, ?, ?, ?)
            ''',
            (str(user_id), mode, json.dumps(request_payload, ensure_ascii=False), json.dumps(response_payload, ensure_ascii=False)),
        )
        conn.commit()


def log_feedback(db_path: Path, user_id: str, item_id: str, event_type: str, value: float | None, context: dict[str, Any] | None) -> None:
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            '''
            INSERT INTO feedback_logs (user_id, item_id, event_type, value, context_json)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (str(user_id), str(item_id), event_type, value, json.dumps(context or {}, ensure_ascii=False)),
        )
        conn.commit()


def fetch_feedback_summary(db_path: Path) -> dict[str, Any]:
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            '''
            SELECT event_type, COUNT(*) as cnt
            FROM feedback_logs
            GROUP BY event_type
            ORDER BY cnt DESC
            '''
        ).fetchall()
        recent = conn.execute(
            '''
            SELECT created_at, user_id, item_id, event_type, value
            FROM feedback_logs
            ORDER BY created_at DESC
            LIMIT 10
            '''
        ).fetchall()
    return {
        "counts_by_event": [{"event_type": event_type, "count": count} for event_type, count in rows],
        "recent_feedback": [
            {"created_at": created_at, "user_id": user_id, "item_id": item_id, "event_type": event_type, "value": value}
            for created_at, user_id, item_id, event_type, value in recent
        ],
    }


def fetch_recommendation_summary(db_path: Path) -> dict[str, Any]:
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            '''
            SELECT mode, COUNT(*) as cnt
            FROM recommendation_logs
            GROUP BY mode
            ORDER BY cnt DESC
            '''
        ).fetchall()
        recent = conn.execute(
            '''
            SELECT created_at, user_id, mode
            FROM recommendation_logs
            ORDER BY created_at DESC
            LIMIT 10
            '''
        ).fetchall()
    return {
        "counts_by_mode": [{"mode": mode, "count": count} for mode, count in rows],
        "recent_requests": [{"created_at": created_at, "user_id": user_id, "mode": mode} for created_at, user_id, mode in recent],
    }
