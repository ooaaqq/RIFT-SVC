"""Single-process FIFO bridge from SQLite tasks to pinned Kaggle launchers."""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from rift_web.config import Settings
from rift_web.database import Database
from scripts.kaggle_common import publish_directory_atomic, sha256

PROFILE_BY_KIND = {
    "background": "big-beta7-bs-roformer-mag-max-spec",
    "deharmony": "anvuew-karaoke",
    "dereverb": "anvuew-dereverb-22.5050",
}


class Dispatcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings.database_path)
        self.stopping = False

    def stop(self, _signum: int, _frame: object) -> None:
        self.stopping = True

    def command_for(self, job: object, result_directory: Path) -> list[str]:
        input_path = str(job["input_path"])
        if job["kind"] == "rift":
            params = json.loads(job["params_json"] or "{}")
            command = [
                sys.executable,
                str(self.settings.source_root / "scripts/kaggle_rift.py"),
                input_path,
                "--output-dir",
                str(result_directory),
            ]
            flags = {
                "key_shift": "--key-shift",
                "steps": "--steps",
                "ds": "--ds",
                "spk": "--spk",
                "cfg_rescale": "--cfg-rescale",
                "robust_f0": "--robust-f0",
                "seed": "--seed",
            }
            for name, flag in flags.items():
                if name in params:
                    command.extend([flag, str(params[name])])
            return command
        return [
            sys.executable,
            str(self.settings.source_root / "scripts/kaggle_separate.py"),
            input_path,
            "--model",
            PROFILE_BY_KIND[job["kind"]],
            "--output-dir",
            str(result_directory),
        ]

    def run_job(self, job: object) -> None:
        job_directory = self.settings.jobs_directory / job["id"]
        work_directory = job_directory / "work"
        result_directory = work_directory / "results"
        output_directory = job_directory / "output"
        work_directory.mkdir(parents=True, exist_ok=True)
        log_path = work_directory / "dispatcher.log"
        if result_directory.exists():
            shutil.rmtree(result_directory)
        command = self.command_for(job, result_directory)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        kaggle_credentials = self.settings.state_directory / "kaggle" / "credentials.json"
        try:
            environment["KAGGLE_API_TOKEN"] = json.loads(
                kaggle_credentials.read_text(encoding="utf-8")
            )["access_token"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("Kaggle OAuth credentials are unavailable") from error
        with log_path.open("ab", buffering=0) as log:
            log.write((f"\n$ {' '.join(command)}\n").encode())
            process = subprocess.Popen(
                command,
                cwd=self.settings.source_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            while process.poll() is None:
                self.database.heartbeat(job["id"])
                time.sleep(10)
            if process.returncode != 0:
                raise RuntimeError(f"launcher exited with status {process.returncode}")

        manifests = sorted(result_directory.rglob("manifest.json"))
        audio_files = sorted(result_directory.rglob("*.wav"))
        if len(manifests) != 1 or not audio_files:
            raise RuntimeError(
                f"invalid result set: manifests={len(manifests)} audio={len(audio_files)}"
            )
        entries = [(path, Path(path.name)) for path in [*audio_files, manifests[0]]]
        published = publish_directory_atomic(output_directory, entries)
        files = []
        for path in published:
            files.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "media_type": mimetypes.guess_type(path.name)[0]
                    or "application/octet-stream",
                }
            )
        self.database.complete(job["id"], files)
        shutil.rmtree(result_directory, ignore_errors=True)

    def run(self) -> None:
        self.settings.state_directory.mkdir(parents=True, exist_ok=True)
        self.settings.jobs_directory.mkdir(parents=True, exist_ok=True)
        self.database.initialize()
        recovered = self.database.recover_interrupted()
        if recovered:
            print(f"Marked {recovered} interrupted job(s) failed", flush=True)
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        while not self.stopping:
            self.database.heartbeat()
            job = self.database.claim_next()
            if job is None:
                time.sleep(self.settings.poll_seconds)
                continue
            print(
                json.dumps(
                    {
                        "job": job["id"],
                        "username": job["username"],
                        "kind": job["kind"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            try:
                self.run_job(job)
            except Exception as error:  # noqa: BLE001 - isolate one queued task
                print(
                    f"Job {job['id']} failed: {type(error).__name__}: {error}",
                    flush=True,
                )
                self.database.fail(job["id"], "处理失败；请稍后重新提交或联系管理员。")


def main() -> None:
    Dispatcher(Settings.from_environment()).run()


if __name__ == "__main__":
    main()
