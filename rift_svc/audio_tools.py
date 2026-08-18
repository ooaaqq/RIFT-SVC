"""Small shared primitives for human-facing audio production tools."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import soundfile as sf


def parse_time(value: str) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS into seconds."""
    parts = value.strip().split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"invalid time {value!r}") from exc
    if not 1 <= len(numbers) <= 3 or any(number < 0 for number in numbers):
        raise ValueError(f"invalid time {value!r}")
    if len(numbers) > 1 and any(number >= 60 for number in numbers[1:]):
        raise ValueError(f"invalid time {value!r}; minute/second fields must be below 60")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60.0 + number
    return seconds


def format_time(seconds: float) -> str:
    if seconds < 0 or not np.isfinite(seconds):
        raise ValueError(f"cannot format invalid time: {seconds}")
    minutes, remainder = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:012.9f}"
    return f"{minutes:02d}:{remainder:012.9f}"


def read_float_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError(f"{path} contains non-finite samples")
    return audio, sample_rate


def ensure_new_paths(paths: Iterable[Path | None]) -> None:
    destinations = [path for path in paths if path is not None]
    resolved = [path.resolve() for path in destinations]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output paths must be different")
    for path in destinations:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing path: {path}")


def _publish_temporary(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_float_wav_new(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    ensure_new_paths([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".wav", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        sf.write(temporary, audio, sample_rate, subtype="FLOAT", format="WAV")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _publish_temporary(temporary, path)


def write_json_new(path: Path, value: Any) -> None:
    ensure_new_paths([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _publish_temporary(temporary, path)


def write_text_new(path: Path, text: str) -> None:
    ensure_new_paths([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".txt", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _publish_temporary(temporary, path)


def concise_cli(main: Callable[[], None], program: str) -> NoReturn:
    """Run a CLI with concise expected errors while retaining argparse exits."""
    try:
        main()
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"{program}: error: {exc}") from None
    raise SystemExit(0)
