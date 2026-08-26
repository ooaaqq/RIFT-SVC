"""Runtime configuration loaded exclusively from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class Settings:
    state_directory: Path
    users_file: Path
    source_root: Path
    listen_host: str
    listen_port: int
    max_upload_bytes: int
    retention_days: int
    poll_seconds: int

    @classmethod
    def from_environment(cls) -> Settings:
        source_root = Path(
            os.environ.get("RIFT_WEB_SOURCE_ROOT", Path(__file__).resolve().parents[1])
        ).resolve()
        state_directory = Path(
            os.environ.get("RIFT_WEB_STATE_DIRECTORY", "var/rift-web")
        ).resolve()
        users_value = os.environ.get("RIFT_WEB_USERS_FILE")
        if not users_value:
            raise RuntimeError("RIFT_WEB_USERS_FILE is required")
        return cls(
            state_directory=state_directory,
            users_file=Path(users_value).resolve(),
            source_root=source_root,
            listen_host=os.environ.get("RIFT_WEB_LISTEN_HOST", "127.0.0.1"),
            listen_port=_positive_int("RIFT_WEB_LISTEN_PORT", 8766),
            max_upload_bytes=_positive_int(
                "RIFT_WEB_MAX_UPLOAD_BYTES", 500 * 1024 * 1024
            ),
            retention_days=_positive_int("RIFT_WEB_RETENTION_DAYS", 14),
            poll_seconds=_positive_int("RIFT_WEB_POLL_SECONDS", 5),
        )

    @property
    def database_path(self) -> Path:
        return self.state_directory / "queue.sqlite3"

    @property
    def jobs_directory(self) -> Path:
        return self.state_directory / "jobs"
