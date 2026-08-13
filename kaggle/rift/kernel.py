"""Kaggle script kernel for one RIFT-SVC inference job."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
RUNTIME_ROOT = Path("/kaggle/temp/rift-svc-runtime")
OUTPUT_ROOT = WORK_ROOT / "rift-output"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_job() -> tuple[Path, dict]:
    jobs = sorted(INPUT_ROOT.rglob("job.json"))
    if len(jobs) != 1:
        raise RuntimeError(f"expected exactly one job.json, found {jobs}")
    job_path = jobs[0]
    return job_path, json.loads(job_path.read_text(encoding="utf-8"))


def main() -> None:
    job_path, job = find_job()
    input_audio = job_path.parent / job["input_name"]
    if not input_audio.is_file():
        raise FileNotFoundError(input_audio)
    if sha256(input_audio) != job["input_sha256"]:
        raise RuntimeError("input SHA-256 mismatch")

    repo_dir = RUNTIME_ROOT / "repo"
    asset_dir = RUNTIME_ROOT / "assets"
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", job["repo_url"], str(repo_dir)])
    run(["git", "checkout", job["repo_ref"]], cwd=repo_dir)
    repo_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
    ).strip()

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "setuptools<81",
            "-r",
            str(repo_dir / "requirements-kaggle.txt"),
        ]
    )

    import soundfile as sf
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is unavailable; enable a GPU accelerator")

    download_command = [
        sys.executable,
        str(repo_dir / "scripts/download_inference_assets.py"),
        "--model-repo",
        job["model_repo"],
        "--model-filename",
        job["model_filename"],
        "--output-dir",
        str(asset_dir),
        "--expected-sha256",
        job["model_sha256"],
    ]
    if job.get("model_revision"):
        download_command.extend(["--model-revision", job["model_revision"]])
    if job.get("modules_revision"):
        download_command.extend(["--modules-revision", job["modules_revision"]])
    run(download_command)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_audio = OUTPUT_ROOT / job["output_name"]
    params = job["params"]
    command = [
        sys.executable,
        str(repo_dir / "infer.py"),
        "--model",
        str(asset_dir / "model" / job["model_filename"]),
        "--assets-dir",
        str(asset_dir / "pretrained"),
        "--input",
        str(input_audio),
        "--output",
        str(output_audio),
        "--speaker",
        params["speaker"],
        "--key-shift",
        str(params["key_shift"]),
        "--device",
        "cuda",
        "--infer-steps",
        str(params["infer_steps"]),
        "--ds-cfg-strength",
        str(params["ds_cfg_strength"]),
        "--spk-cfg-strength",
        str(params["spk_cfg_strength"]),
        "--cfg-rescale",
        str(params["cfg_rescale"]),
        "--robust-f0",
        str(params["robust_f0"]),
        "--seed",
        str(params["seed"]),
        "--output-subtype",
        "FLOAT",
    ]
    run(command, cwd=repo_dir)

    info = sf.info(output_audio)
    if info.samplerate != 44100 or info.channels != 1 or info.subtype != "FLOAT":
        raise RuntimeError(f"unexpected output format: {info}")

    manifest = {
        **job,
        "completed_at": datetime.now(UTC).isoformat(),
        "repo_commit": repo_commit,
        "output_sha256": sha256(output_audio),
        "output_bytes": output_audio.stat().st_size,
        "audio": {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "format": info.format,
            "subtype": info.subtype,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "packages": subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True
            ).splitlines(),
        },
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Output: {output_audio}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
