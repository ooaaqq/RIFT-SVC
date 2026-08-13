"""Small shared helpers for local Kaggle job launchers."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".flac", ".m4a", ".mp3", ".ogg", ".opus"}


def run(
    command: list[str],
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=capture,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_datasets() -> set[str]:
    result = run(
        [
            "kaggle",
            "datasets",
            "list",
            "--mine",
            "--page-size",
            "200",
            "--format",
            "json",
        ],
        capture=True,
    )
    return {item["ref"] for item in json.loads(result.stdout)}


def wait_for_dataset(dataset_ref: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = run(
            ["kaggle", "datasets", "status", dataset_ref],
            capture=True,
            check=False,
        )
        status = (result.stdout + result.stderr).strip().lower()
        print(f"Dataset status: {status or 'unknown'}", flush=True)
        if result.returncode == 0 and status == "ready":
            return
        if any(word in status for word in ("error", "failed", "invalid")):
            raise RuntimeError(f"dataset processing failed: {status}")
        time.sleep(10)
    raise TimeoutError(f"dataset did not become ready within {timeout} seconds")


def kernel_status(kernel_ref: str) -> str:
    result = run(["kaggle", "kernels", "status", kernel_ref], capture=True)
    text = result.stdout + result.stderr
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", text)
    if not match:
        raise RuntimeError(f"cannot parse Kaggle kernel status: {text.strip()}")
    return match.group(1)


def wait_for_kernel(kernel_ref: str, poll_interval: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    failure_states = {"ERROR", "CANCELLED", "FAILED"}
    while time.monotonic() < deadline:
        status = kernel_status(kernel_ref)
        print(f"Kernel status: {status}", flush=True)
        if status == "COMPLETE":
            return
        if status in failure_states:
            run(["kaggle", "kernels", "logs", kernel_ref], check=False)
            raise RuntimeError(f"Kaggle kernel ended with status {status}")
        time.sleep(poll_interval)
    run(["kaggle", "kernels", "logs", kernel_ref], check=False)
    raise TimeoutError(f"kernel did not complete within {timeout} seconds")


def probe_audio(path: Path) -> dict[str, object]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,sample_fmt,duration",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one audio stream in {path}")
    return streams[0]
