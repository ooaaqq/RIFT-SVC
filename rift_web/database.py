"""SQLite queue and immutable task metadata."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

JOB_KINDS = {"background", "deharmony", "dereverb", "rift"}
JOB_STATUSES = {"queued", "running", "succeeded", "failed", "cancelled"}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    title TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    input_bytes INTEGER NOT NULL,
                    audio_json TEXT NOT NULL,
                    params_json TEXT NOT NULL DEFAULT '{}',
                    public_error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    heartbeat_at TEXT,
                    CHECK (kind IN ('background','deharmony','dereverb','rift')),
                    CHECK (status IN ('queued','running','succeeded','failed','cancelled'))
                );
                CREATE INDEX IF NOT EXISTS jobs_queue
                    ON jobs(status, created_at, id);
                CREATE TABLE IF NOT EXISTS job_files (
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    bytes INTEGER NOT NULL,
                    media_type TEXT NOT NULL,
                    PRIMARY KEY (job_id, name)
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(jobs)")
            }
            if "params_json" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN params_json TEXT NOT NULL DEFAULT '{}'")

    def create_job(self, record: dict[str, object]) -> None:
        if record["kind"] not in JOB_KINDS:
            raise ValueError("unsupported job kind")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, username, title, kind, status, original_name, input_path,
                    input_sha256, input_bytes, audio_json, params_json, created_at
                ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["username"],
                    record["title"],
                    record["kind"],
                    record["original_name"],
                    record["input_path"],
                    record["input_sha256"],
                    record["input_bytes"],
                    record["audio_json"],
                    record.get("params_json", "{}"),
                    record["created_at"],
                ),
            )

    def list_jobs(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT * FROM jobs
                    ORDER BY
                        CASE WHEN status = 'queued' THEN 0 ELSE 1 END,
                        CASE WHEN status = 'queued' THEN created_at END ASC,
                        created_at DESC,
                        id DESC
                    LIMIT 200
                    """
                )
            )

    def finished_before(self, cutoff: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT * FROM jobs
                    WHERE finished_at IS NOT NULL AND finished_at < ?
                    ORDER BY finished_at
                    """,
                    (cutoff,),
                )
            )

    def get_job(self, job_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()

    def list_files(self, job_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM job_files WHERE job_id = ? ORDER BY name", (job_id,)
                )
            )

    def get_file(self, job_id: str, name: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM job_files WHERE job_id = ? AND name = ?",
                (job_id, name),
            ).fetchone()

    def cancel(self, job_id: str, username: str, admin: bool) -> bool:
        with self.connect() as connection:
            if admin:
                cursor = connection.execute(
                    """
                    UPDATE jobs SET status = 'cancelled', finished_at = ?
                    WHERE id = ? AND status = 'queued'
                    """,
                    (now_iso(), job_id),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE jobs SET status = 'cancelled', finished_at = ?
                    WHERE id = ? AND username = ? AND status = 'queued'
                    """,
                    (now_iso(), job_id, username),
                )
            return cursor.rowcount == 1

    def claim_next(self) -> sqlite3.Row | None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM jobs WHERE status = 'queued'
                ORDER BY created_at, id LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            timestamp = now_iso()
            cursor = connection.execute(
                """
                UPDATE jobs SET status = 'running', started_at = ?, heartbeat_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (timestamp, timestamp, row["id"]),
            )
            if cursor.rowcount != 1:
                return None
            return connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (row["id"],)
            ).fetchone()

    def heartbeat(self, job_id: str | None = None) -> None:
        timestamp = now_iso()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES ('dispatcher_heartbeat', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (timestamp,),
            )
            if job_id:
                connection.execute(
                    "UPDATE jobs SET heartbeat_at = ? WHERE id = ? AND status = 'running'",
                    (timestamp, job_id),
                )

    def recover_interrupted(self) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'failed', finished_at = ?,
                    public_error = '执行器曾中断；远端状态需要人工确认后再重新提交。'
                WHERE status = 'running'
                """,
                (now_iso(),),
            )
            return cursor.rowcount

    def complete(self, job_id: str, files: list[dict[str, object]]) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for file in files:
                connection.execute(
                    """
                    INSERT INTO job_files(job_id, name, path, sha256, bytes, media_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        file["name"],
                        file["path"],
                        file["sha256"],
                        file["bytes"],
                        file["media_type"],
                    ),
                )
            connection.execute(
                """
                UPDATE jobs SET status = 'succeeded', finished_at = ?, heartbeat_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (now_iso(), now_iso(), job_id),
            )

    def fail(self, job_id: str, public_error: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status = 'failed', finished_at = ?,
                    heartbeat_at = ?, public_error = ?
                WHERE id = ? AND status = 'running'
                """,
                (now_iso(), now_iso(), public_error[:500], job_id),
            )

    def metadata(self) -> dict[str, str]:
        with self.connect() as connection:
            return {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM metadata")
            }
