"""Delete expired private audio while retaining queue metadata and manifests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from rift_web.config import Settings
from rift_web.database import Database


def main() -> None:
    settings = Settings.from_environment()
    database = Database(settings.database_path)
    database.initialize()
    cutoff = datetime.now(UTC) - timedelta(days=settings.retention_days)
    removed = 0
    for job in database.finished_before(cutoff.isoformat()):
        paths = [Path(job["input_path"])]
        paths.extend(
            Path(record["path"])
            for record in database.list_files(job["id"])
            if record["media_type"].startswith("audio/")
        )
        for path in paths:
            if path.is_file():
                path.unlink()
                removed += 1
    print(f"Removed {removed} expired audio file(s)")


if __name__ == "__main__":
    main()
