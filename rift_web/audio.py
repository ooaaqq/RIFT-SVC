"""Upload validation that never trusts extensions or browser MIME types."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ALLOWED_SUFFIXES = {".wav", ".flac", ".m4a", ".mp3", ".ogg", ".opus"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_audio(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "format=format_name,duration:stream=codec_name,sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError("文件必须只包含一条音频流")
    stream = streams[0]
    sample_rate = int(stream["sample_rate"])
    channels = int(stream["channels"])
    duration = float(stream.get("duration") or payload["format"]["duration"])
    if not 8000 <= sample_rate <= 384000:
        raise ValueError("不支持该采样率")
    if not 1 <= channels <= 8:
        raise ValueError("不支持该声道数")
    if not 0.1 <= duration <= 60 * 60:
        raise ValueError("音频时长必须在 0.1 秒到 60 分钟之间")
    return {
        "codec_name": stream.get("codec_name"),
        "sample_rate": sample_rate,
        "channels": channels,
        "duration": duration,
        "format_name": payload.get("format", {}).get("format_name"),
    }
