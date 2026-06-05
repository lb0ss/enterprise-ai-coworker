import sqlite3
import os
import time
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "analytics.db")


def init_db():
    """Create tables if they don't exist."""
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT,
                latency_ms INTEGER,
                token_count INTEGER,
                cache_hit BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER REFERENCES queries(id),
                rating INTEGER NOT NULL,  -- 1 = thumbs up, -1 = thumbs down
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_query(question: str, answer: str, latency_ms: int, token_count: int, cache_hit: bool) -> int:
    """Insert a query record and return its ID."""
    with get_db() as db:
        cursor = db.execute(
            "INSERT INTO queries (question, answer, latency_ms, token_count, cache_hit) VALUES (?, ?, ?, ?, ?)",
            (question, answer, latency_ms, token_count, cache_hit),
        )
        return cursor.lastrowid


def log_feedback(query_id: int, rating: int) -> None:
    """Store thumbs up (1) or thumbs down (-1) for a query."""
    with get_db() as db:
        db.execute(
            "INSERT INTO feedback (query_id, rating) VALUES (?, ?)",
            (query_id, rating),
        )


def get_analytics() -> dict:
    """Return aggregated stats."""
    with get_db() as db:
        total = db.execute("SELECT COUNT(*) as count FROM queries").fetchone()["count"]
        cache_hits = db.execute("SELECT COUNT(*) as count FROM queries WHERE cache_hit = 1").fetchone()["count"]
        avg_latency = db.execute("SELECT AVG(latency_ms) as avg FROM queries WHERE cache_hit = 0").fetchone()["avg"]
        total_tokens = db.execute("SELECT SUM(token_count) as total FROM queries").fetchone()["total"]
        thumbs_up = db.execute("SELECT COUNT(*) as count FROM feedback WHERE rating = 1").fetchone()["count"]
        thumbs_down = db.execute("SELECT COUNT(*) as count FROM feedback WHERE rating = -1").fetchone()["count"]

        recent = db.execute(
            "SELECT question, latency_ms, token_count, cache_hit, created_at FROM queries ORDER BY created_at DESC LIMIT 10"
        ).fetchall()

    return {
        "total_queries": total,
        "cache_hits": cache_hits,
        "cache_miss_rate": round((total - cache_hits) / total, 2) if total else 0,
        "avg_latency_ms": round(avg_latency) if avg_latency else 0,
        "total_tokens": total_tokens or 0,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "recent_queries": [dict(r) for r in recent],
    }
