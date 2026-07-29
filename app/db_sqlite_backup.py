import json
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "demo_history.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS search_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        mode TEXT,
        country TEXT,
        question_id TEXT,
        dimension TEXT,
        question TEXT,
        final_answer_selected TEXT,
        score_suggestion REAL,
        confidence REAL,
        evidence_sufficiency TEXT,
        justification_generated TEXT,
        retrieved_sources_json TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_search_result(result: dict) -> int:
    retrieved_sources_json = json.dumps(
        result.get("retrieved_sources", []),
        ensure_ascii=False
    )

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO search_history (
        mode,
        country,
        question_id,
        dimension,
        question,
        final_answer_selected,
        score_suggestion,
        confidence,
        evidence_sufficiency,
        justification_generated,
        retrieved_sources_json
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.get("mode", ""),
        result.get("country", ""),
        result.get("question_id", ""),
        result.get("dimension", ""),
        result.get("question", ""),
        result.get("final_answer_selected", ""),
        result.get("score_suggestion", None),
        result.get("confidence", None),
        result.get("evidence_sufficiency", ""),
        result.get("justification_generated", ""),
        retrieved_sources_json,
    ))

    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def list_saved_results(limit: int = 20) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT id, created_at, mode, country, question_id, dimension, final_answer_selected, score_suggestion
    FROM search_history
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))

    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_saved_result(record_id: int) -> dict | None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM search_history
    WHERE id = ?
    """, (record_id,))

    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    result = dict(row)
    result["retrieved_sources"] = json.loads(result.get("retrieved_sources_json", "[]") or "[]")
    return result