"""Human-first workspace helpers for song mixing projects.

The visible filesystem is the product: projects use official song and cover
names, audio is copied as ordinary files, and versions are named by the change
a listener can understand.  Hashing remains available for verification, not
as the storage model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SHELF_DIRS = (
    "00. Recent Listening",
    "10. In Progress",
    "20. On Hold",
    "30. Completed",
    "90. Shared Resources",
    "99. Tools and Guide",
)
PROJECT_DIRS = (
    "00. Listen",
    "10. Sources",
    "20. AI",
    "30. Mix",
    "40. Versions",
    "90. Deliverables",
)
PROJECT_STATUSES = {
    "in-progress": "10. In Progress",
    "on-hold": "20. On Hold",
    "completed": "30. Completed",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_text(path: Path, text: str) -> None:
    """Create a text file without replacing an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(text)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing path: {path}") from exc


def write_json_atomic(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite existing path: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_component(value: str, field: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."}:
        raise ValueError(f"{field} must not be empty")
    if "/" in value or "\0" in value or any(ord(char) < 32 for char in value):
        raise ValueError(f"{field} contains an invalid path character")
    return value


def project_name(number: int, title: str, cover: str) -> str:
    if not 0 <= number <= 999:
        raise ValueError("project number must be between 0 and 999")
    title = _safe_component(title, "title")
    cover = _safe_component(cover, "cover credit")
    return f"{number:03d}. {title}-{cover}"


def version_name(number: int, label: str) -> str:
    if not 0 <= number <= 99:
        raise ValueError("version number must be between 0 and 99")
    return f"{number:02d}. {_safe_component(label, 'version label')}"


def require_workspace(root: Path) -> Path:
    root = root.expanduser().resolve()
    missing = [name for name in SHELF_DIRS if not (root / name).is_dir()]
    if missing:
        raise ValueError(f"not a mixctl shelf workspace; missing: {', '.join(missing)}")
    return root


def require_project(project: Path) -> Path:
    project = project.expanduser().resolve()
    missing = [name for name in PROJECT_DIRS if not (project / name).is_dir()]
    if missing:
        raise ValueError(f"not a mixctl song project; missing: {', '.join(missing)}")
    return project


def create_workspace(root: Path) -> Path:
    root = root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"workspace directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for name in SHELF_DIRS:
        (root / name).mkdir()
    return root


def create_project(
    workspace: Path,
    number: int,
    title: str,
    cover: str,
    *,
    status: str = "in-progress",
) -> Path:
    root = require_workspace(workspace)
    try:
        shelf = PROJECT_STATUSES[status]
    except KeyError as exc:
        raise ValueError(f"unknown project status: {status}") from exc
    project = root / shelf / project_name(number, title, cover)
    if project.exists():
        raise FileExistsError(f"project already exists: {project}")
    for name in PROJECT_DIRS:
        (project / name).mkdir(parents=True, exist_ok=True)
    return project


def _project_destination(project: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("destination must be a relative path inside the project")
    destination = project.joinpath(*relative.parts)
    if destination == project:
        raise ValueError("destination must name a file")
    return destination


def copy_regular_file(source: Path, destination: Path) -> Path:
    """Create an independent regular file, using a reflink only as an optimization."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    if temporary.exists():
        raise FileExistsError(f"temporary path already exists: {temporary}")
    try:
        executable = shutil.which("cp")
        if executable:
            subprocess.run(
                [
                    executable,
                    "--reflink=auto",
                    "--preserve=mode,timestamps",
                    "--",
                    str(source),
                    str(temporary),
                ],
                check=True,
            )
        else:
            shutil.copy2(source, temporary)
        os.link(temporary, destination)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if destination.is_symlink() or not destination.is_file():
        raise RuntimeError(f"copy did not create a regular file: {destination}")
    return destination


def copy_project_file(project: Path, source: Path, destination: Path) -> Path:
    project = require_project(project)
    return copy_regular_file(source, _project_destination(project, destination))


def create_audition(project: Path, entries: list[tuple[str, Path]]) -> list[Path]:
    project = require_project(project)
    created = []
    for label, source in entries:
        label = _safe_component(label, "listening label")
        suffix = source.suffix.lower()
        if not suffix:
            raise ValueError(f"listening source has no extension: {source}")
        created.append(
            copy_regular_file(source, project / "00. Listen" / f"{label}{suffix}")
        )
    return created


def create_version(
    project: Path,
    number: int,
    label: str,
    entries: list[tuple[str, Path]],
) -> Path:
    project = require_project(project)
    version = project / "40. Versions" / version_name(number, label)
    if version.exists():
        raise FileExistsError(f"version already exists: {version}")
    temporary = version.parent / f".{version.name}.{os.getpid()}.building"
    if temporary.exists():
        raise FileExistsError(f"temporary version already exists: {temporary}")
    temporary.mkdir()
    try:
        for name, source in entries:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("version entry name must stay inside the version folder")
            copy_regular_file(source, temporary.joinpath(*relative.parts))
        temporary.replace(version)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return version


def ffprobe(path: Path) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("ffprobe is required but was not found on PATH")
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_fmt,sample_rate,channels,bits_per_sample,bits_per_raw_sample,duration,duration_ts,time_base",
            "-show_entries",
            "format=format_name,duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not value.get("streams"):
        raise ValueError(f"no audio stream found: {path}")
    return value


def probe_file(path: Path, *, with_hash: bool = False) -> dict[str, Any]:
    supplied_path = path.expanduser().absolute()
    resolved_path = supplied_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(supplied_path)
    report: dict[str, Any] = {
        "path": str(supplied_path),
        "resolved_path": str(resolved_path),
        "bytes": resolved_path.stat().st_size,
    }
    report.update(ffprobe(resolved_path))
    if with_hash:
        report["sha256"] = sha256(resolved_path)
    return report


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")
    return executable


def decode_file(path: Path) -> None:
    """Decode the complete first audio stream and fail on truncated/corrupt input."""
    subprocess.run(
        [
            _ffmpeg(),
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def measure_loudness(path: Path) -> dict[str, float]:
    """Measure integrated loudness, loudness range, and true peak with FFmpeg."""
    completed = subprocess.run(
        [
            _ffmpeg(),
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-filter:a",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    patterns = {
        "integrated_lufs": r"^\s*I:\s*([-+]?\w+(?:\.\d+)?)\s+LUFS\s*$",
        "loudness_range_lu": r"^\s*LRA:\s*([-+]?\w+(?:\.\d+)?)\s+LU\s*$",
        "true_peak_dbtp": r"^\s*Peak:\s*([-+]?\w+(?:\.\d+)?)\s+dBFS\s*$",
    }
    result: dict[str, float] = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, completed.stderr, flags=re.MULTILINE)
        if not matches:
            raise ValueError(f"could not read {name} from FFmpeg ebur128 output")
        result[name] = float(matches[-1])
    return result


def _audio_stream(report: dict[str, Any]) -> dict[str, Any]:
    return report["streams"][0]


def _audio_duration(report: dict[str, Any]) -> float:
    stream_duration = _audio_stream(report).get("duration")
    if stream_duration is not None:
        return float(stream_duration)
    return float(report["format"]["duration"])


def check_delivery(
    paths: Iterable[Path],
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
    target_lufs: float | None = None,
    lufs_tolerance: float = 1.0,
    max_true_peak: float | None = None,
    reference: Path | None = None,
    duration_tolerance_samples: int = 1,
) -> dict[str, Any]:
    """Fully decode and measure delivery files, applying optional acceptance limits."""
    results = []
    failures = []
    reference_report = probe_file(reference, with_hash=True) if reference else None
    reference_duration = (
        _audio_duration(reference_report) if reference_report is not None else None
    )
    for supplied in paths:
        try:
            result = probe_file(supplied, with_hash=True)
            resolved = Path(result["resolved_path"])
            decode_file(resolved)
            result["loudness"] = measure_loudness(resolved)
            stream = _audio_stream(result)
            problems = []
            if sample_rate is not None and int(stream["sample_rate"]) != sample_rate:
                problems.append(
                    f"sample rate is {stream['sample_rate']} Hz; expected {sample_rate} Hz"
                )
            if channels is not None and int(stream["channels"]) != channels:
                problems.append(
                    f"channels is {stream['channels']}; expected {channels}"
                )
            measured_lufs = result["loudness"]["integrated_lufs"]
            if (
                target_lufs is not None
                and abs(measured_lufs - target_lufs) > lufs_tolerance
            ):
                problems.append(
                    f"integrated loudness is {measured_lufs:.1f} LUFS; "
                    f"expected {target_lufs:.1f} ± {lufs_tolerance:.1f} LU"
                )
            measured_peak = result["loudness"]["true_peak_dbtp"]
            if max_true_peak is not None and measured_peak > max_true_peak:
                problems.append(
                    f"true peak is {measured_peak:.1f} dBTP; "
                    f"maximum is {max_true_peak:.1f} dBTP"
                )
            if reference_duration is not None:
                duration_delta_ms = (
                    _audio_duration(result) - reference_duration
                ) * 1000.0
                duration_delta_samples = round(
                    duration_delta_ms * int(stream["sample_rate"]) / 1000.0
                )
                result["reference_duration_delta_ms"] = duration_delta_ms
                result["reference_duration_delta_samples"] = duration_delta_samples
                if abs(duration_delta_samples) > duration_tolerance_samples:
                    problems.append(
                        f"duration differs from reference by {duration_delta_samples} "
                        f"samples ({duration_delta_ms:.3f} ms); tolerance is "
                        f"±{duration_tolerance_samples} samples"
                    )
            result["ok"] = not problems
            result["problems"] = problems
            results.append(result)
            if problems:
                failures.append({"path": str(supplied), "error": "; ".join(problems)})
        except (
            json.JSONDecodeError,
            OSError,
            RuntimeError,
            ValueError,
            subprocess.CalledProcessError,
        ) as exc:
            failures.append({"path": str(supplied), "error": str(exc)})
    return {
        "schema": 1,
        "generated_at": utc_now(),
        "ok": not failures,
        "reference": reference_report,
        "results": results,
        "failures": failures,
    }


def verify_files(paths: Iterable[Path]) -> dict[str, Any]:
    results = []
    failures = []
    for supplied in paths:
        try:
            result = probe_file(supplied, with_hash=True)
            result["ok"] = True
            results.append(result)
        except (
            json.JSONDecodeError,
            OSError,
            RuntimeError,
            ValueError,
            subprocess.CalledProcessError,
        ) as exc:
            failures.append({"path": str(supplied), "error": str(exc)})
    return {
        "schema": 1,
        "generated_at": utc_now(),
        "ok": not failures,
        "results": results,
        "failures": failures,
    }


def parse_entry(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("entry must be NAME=PATH") from exc
    if not label or not path:
        raise argparse.ArgumentTypeError("entry must be NAME=PATH")
    return label, Path(path)


def emit(value: Any, output: Path | None = None) -> None:
    if output is None:
        print(json.dumps(value, ensure_ascii=False, indent=2))
    else:
        write_json_atomic(output.resolve(), value)
        print(output.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mixctl")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a human-first shelf workspace")
    init.add_argument("workspace", type=Path)

    new = commands.add_parser("new", help="create a song project with its official identity")
    new.add_argument("workspace", type=Path)
    new.add_argument("--number", type=int, required=True)
    new.add_argument("--title", required=True)
    new.add_argument("--cover", required=True)
    new.add_argument("--status", choices=sorted(PROJECT_STATUSES), default="in-progress")

    copy = commands.add_parser("copy", help="copy one ordinary file into a project")
    copy.add_argument("project", type=Path)
    copy.add_argument("source", type=Path)
    copy.add_argument("destination", type=Path)

    listen = commands.add_parser("listen", help="copy selected files into 00. Listen")
    listen.add_argument("project", type=Path)
    listen.add_argument("--entry", type=parse_entry, action="append", required=True)

    version = commands.add_parser("version", help="create a meaningful version snapshot")
    version.add_argument("project", type=Path)
    version.add_argument("--number", type=int, required=True)
    version.add_argument("--label", required=True)
    version.add_argument("--entry", type=parse_entry, action="append", default=[])

    probe = commands.add_parser("probe", help="inspect one audio file")
    probe.add_argument("path", type=Path)
    probe.add_argument("--hash", action="store_true")
    probe.add_argument("--output", type=Path)

    preflight = commands.add_parser(
        "preflight", help="confirm that render inputs exist and contain readable audio"
    )
    preflight.add_argument("paths", type=Path, nargs="+")
    preflight.add_argument("--output", type=Path)

    delivery = commands.add_parser(
        "delivery", help="fully decode, measure, and validate delivery files"
    )
    delivery.add_argument("paths", type=Path, nargs="+")
    delivery.add_argument("--sample-rate", type=int)
    delivery.add_argument("--channels", type=int)
    delivery.add_argument("--target-lufs", type=float)
    delivery.add_argument("--lufs-tolerance", type=float, default=1.0)
    delivery.add_argument("--max-true-peak", type=float)
    delivery.add_argument("--reference", type=Path)
    delivery.add_argument("--duration-tolerance-samples", type=int, default=1)
    delivery.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            print(create_workspace(args.workspace))
        elif args.command == "new":
            print(
                create_project(
                    args.workspace,
                    args.number,
                    args.title,
                    args.cover,
                    status=args.status,
                )
            )
        elif args.command == "copy":
            print(copy_project_file(args.project, args.source, args.destination))
        elif args.command == "listen":
            for path in create_audition(args.project, args.entry):
                print(path)
        elif args.command == "version":
            print(create_version(args.project, args.number, args.label, args.entry))
        elif args.command == "probe":
            emit(probe_file(args.path, with_hash=args.hash), args.output)
        elif args.command == "preflight":
            report = verify_files(args.paths)
            emit(report, args.output)
            return 0 if report["ok"] else 1
        elif args.command == "delivery":
            if args.sample_rate is not None and args.sample_rate <= 0:
                raise ValueError("--sample-rate must be positive")
            if args.channels is not None and args.channels <= 0:
                raise ValueError("--channels must be positive")
            if args.lufs_tolerance < 0:
                raise ValueError("--lufs-tolerance cannot be negative")
            if args.duration_tolerance_samples < 0:
                raise ValueError("--duration-tolerance-samples cannot be negative")
            report = check_delivery(
                args.paths,
                sample_rate=args.sample_rate,
                channels=args.channels,
                target_lufs=args.target_lufs,
                lufs_tolerance=args.lufs_tolerance,
                max_true_peak=args.max_true_peak,
                reference=args.reference,
                duration_tolerance_samples=args.duration_tolerance_samples,
            )
            emit(report, args.output)
            return 0 if report["ok"] else 1
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"mixctl: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
