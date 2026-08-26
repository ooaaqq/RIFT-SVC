"""Small shared helpers for local Kaggle job launchers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock
from fractions import Fraction
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".flac", ".m4a", ".mp3", ".ogg", ".opus"}


@contextmanager
def exclusive_kaggle_channel(channel: str) -> Iterator[None]:
    """Prevent two local launchers from mutating one shared Kaggle channel."""
    digest = hashlib.sha256(channel.encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"rift-svc-kaggle-{digest}.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            flock(handle.fileno(), LOCK_EX | LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            owner = handle.read().strip() or "another local process"
            raise RuntimeError(
                f"Kaggle channel is already in use: {channel} ({owner})"
            ) from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}")
        handle.flush()
        yield
    finally:
        try:
            flock(handle.fileno(), LOCK_UN)
        finally:
            handle.close()


def publish_directory_atomic(
    destination: Path, files: Iterable[tuple[Path, Path]]
) -> list[Path]:
    """Copy a complete result set, then expose it with one directory rename."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing Run: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = list(files)
    relative_paths = [relative for _, relative in entries]
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("published file names must be unique")
    if any(
        relative.is_absolute() or not relative.parts or ".." in relative.parts
        for relative in relative_paths
    ):
        raise ValueError("published files must stay inside the Run directory")

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.publishing-", dir=destination.parent
        )
    )
    try:
        for source, relative in entries:
            target = staging.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return [destination.joinpath(*relative.parts) for relative in relative_paths]


def safe_audio_name(path: Path) -> str:
    """Return a Kaggle-safe basename while retaining the audio extension."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)
    stem = re.sub(r"_+", "_", stem).strip("._") or "audio"
    suffix = path.suffix.lower()
    return f"{stem}{suffix}"


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


def wait_for_dataset(
    dataset_ref: str,
    expected_files: set[str],
    timeout: int,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = run(
            ["kaggle", "datasets", "files", dataset_ref, "--format", "json"],
            capture=True,
            check=False,
        )
        if result.returncode == 0:
            available = {item["name"] for item in json.loads(result.stdout)}
            print(f"Dataset files: {sorted(available)}", flush=True)
            if expected_files <= available:
                return
        else:
            detail = (result.stdout + result.stderr).strip()
            print(f"Dataset files unavailable: {detail}", flush=True)
        time.sleep(10)
    raise TimeoutError(f"dataset did not become ready within {timeout} seconds")


def current_dataset_version(dataset_ref: str) -> int:
    """Return the current version so a kernel mounts the just-published data."""
    result = run(
        [
            "kaggle",
            "datasets",
            "status",
            dataset_ref,
            "--format",
            "json(current_version_number)",
        ],
        capture=True,
    )
    payload = json.loads(result.stdout)
    version = int(payload["current_version_number"])
    if version < 1:
        raise RuntimeError(
            f"invalid current dataset version for {dataset_ref}: {version}"
        )
    return version


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
        try:
            status = kernel_status(kernel_ref)
        except (RuntimeError, subprocess.CalledProcessError) as error:
            print(f"Kernel status unavailable: {error}", flush=True)
            time.sleep(poll_interval)
            continue
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
            "stream=codec_name,sample_rate,channels,sample_fmt,duration,duration_ts,time_base",
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


def frame_count_at_rate(probe: dict[str, object], sample_rate: int) -> int:
    """Convert a probed stream duration to an expected frame count."""
    duration_ts = probe.get("duration_ts")
    time_base = probe.get("time_base")
    if duration_ts not in (None, "N/A") and time_base not in (None, "N/A"):
        duration = int(str(duration_ts)) * Fraction(str(time_base))
    else:
        duration = Fraction(str(probe["duration"]))
    return round(duration * sample_rate)


def frame_delta(input_probe: dict[str, object], output_probe: dict[str, object]) -> int:
    """Return output frames minus expected input frames at the output rate."""
    output_rate = int(str(output_probe["sample_rate"]))
    expected = frame_count_at_rate(input_probe, output_rate)
    actual = frame_count_at_rate(output_probe, output_rate)
    return actual - expected
