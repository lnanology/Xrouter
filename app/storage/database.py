"""SQLite storage layer. All access goes through Repository classes — core
code never writes raw SQL (section 二十六). Swapping to PostgreSQL later
means reimplementing this module + repositories/, nothing else.

memory_entries (Phase 3: Memory/RAG groundwork, app/storage/repositories/
memory.py, app/retrieval/) backs XRouter's own long-term memory — distinct
from a single request's own conversation history, and from cache_entries
above (which is a response cache keyed by request hash, not a store of
facts XRouter chooses to remember)."""
from __future__ import annotations

import asyncio

import aiosqlite

from app.observability.logging import get_logger

logger = get_logger("storage")

SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,          -- public_id, e.g. ollama/llama3
    provider_id TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    quality_score REAL,
    speed_score REAL,
    reliability_score REAL,
    cost_score REAL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    task_type TEXT,
    complexity INTEGER,
    routing_policy TEXT,
    status TEXT,
    latency_ms REAL,
    ttft_ms REAL,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    retry_count INTEGER,
    fallback_count INTEGER,
    cache_hit INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS request_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    attempt_index INTEGER NOT NULL,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms REAL,
    error TEXT,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_health (
    provider_id TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms REAL,
    success_rate REAL,
    last_error TEXT,
    checked_at REAL NOT NULL,
    PRIMARY KEY (provider_id, checked_at)
);

CREATE TABLE IF NOT EXISTS quota_usage (
    provider_id TEXT NOT NULL,
    requests_total INTEGER,
    tokens_total INTEGER,
    risk TEXT,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (provider_id, recorded_at)
);

CREATE TABLE IF NOT EXISTS routing_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at REAL NOT NULL,
    metrics_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    ttft_ms REAL,
    total_latency_ms REAL,
    tokens_per_sec REAL,
    success INTEGER,
    recorded_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cache_entries (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at REAL NOT NULL,
    ttl_seconds REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload_json TEXT,
    recorded_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_entries (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    content TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = asyncio.Lock()

    async def init(self, timeout: float = 5.0) -> None:
        async def _init() -> None:
            async with aiosqlite.connect(self.db_path) as db:
                await db.executescript(SCHEMA)
                await db.commit()

        try:
            await asyncio.wait_for(_init(), timeout=timeout)
        except Exception as e:
            # A telemetry DB failure must never prevent the gateway from serving
            # requests — log and continue; repositories below are defensive too.
            logger.error("database init failed (telemetry will be degraded): %s", e)

    def connect(self):
        """Returns an un-opened aiosqlite connection context manager. Callers
        must use `async with db.connect() as conn:` (do NOT `await` this) —
        aiosqlite's Connection starts its background thread exactly once,
        either on `await connect(...)` or on `__aenter__`, never both."""
        return aiosqlite.connect(self.db_path)
