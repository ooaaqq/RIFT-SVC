"""Kaggle script kernel for one dual-T4 RIFT-SVC batch."""

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
MAX_GPU_WORKERS = 2
BATCH_WORKER_SOURCE = "__BATCH_WORKER_SOURCE__"


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


def worker_count(gpu_count: int, item_count: int) -> int:
    return min(MAX_GPU_WORKERS, gpu_count, item_count)


def main() -> None:
    job_path, job = find_job()
    items = job.get("items", [])
    if job.get("schema_version") != 2 or not items:
        raise RuntimeError("expected a non-empty schema-2 batch job")
    for item in items:
        input_audio = job_path.parent / item["input_name"]
        if not input_audio.is_file():
            raise FileNotFoundError(input_audio)
        if sha256(input_audio) != item["input_sha256"]:
            raise RuntimeError(f"input SHA-256 mismatch: {input_audio}")

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
    gpu_count = torch.cuda.device_count()
    workers = worker_count(gpu_count, len(items))
    if workers < 1:
        raise RuntimeError("no usable CUDA device was found")
    print(
        f"Batch items: {len(items)}; CUDA devices: {gpu_count}; workers: {workers}",
        flush=True,
    )

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
    worker_results = OUTPUT_ROOT / "worker-results.json"
    worker_script = RUNTIME_ROOT / "batch_worker.py"
    if BATCH_WORKER_SOURCE == "__BATCH_WORKER_SOURCE__":
        raise RuntimeError("batch worker source was not embedded by the launcher")
    worker_script.write_text(BATCH_WORKER_SOURCE, encoding="utf-8")
    run(
        [
            sys.executable,
            str(worker_script),
            "--repo-dir",
            str(repo_dir),
            "--asset-dir",
            str(asset_dir),
            "--job",
            str(job_path),
            "--input-dir",
            str(job_path.parent),
            "--output-dir",
            str(OUTPUT_ROOT),
            "--results",
            str(worker_results),
            "--workers",
            str(workers),
        ]
    )
    results = json.loads(worker_results.read_text(encoding="utf-8"))
    by_index = {result["index"]: result for result in results}
    if len(by_index) != len(items):
        raise RuntimeError(
            f"expected {len(items)} worker results, found {len(by_index)}"
        )

    outputs = []
    for item in items:
        result = by_index[item["index"]]
        output_audio = OUTPUT_ROOT / item["transport_output_name"]
        if result.get("error"):
            raise RuntimeError(
                f"batch item {item['index']} failed on {result.get('device')}: "
                f"{result['error']}"
            )
        if not output_audio.is_file():
            raise FileNotFoundError(output_audio)
        info = sf.info(output_audio)
        if info.samplerate != 44100 or info.channels != 1 or info.subtype != "FLOAT":
            raise RuntimeError(f"unexpected output format: {info}")
        outputs.append(
            {
                "index": item["index"],
                "transport_output_name": item["transport_output_name"],
                "output_name": item["output_name"],
                "sha256": sha256(output_audio),
                "bytes": output_audio.stat().st_size,
                "device": result["device"],
                "audio": {
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "duration": info.duration,
                    "format": info.format,
                    "subtype": info.subtype,
                },
            }
        )

    manifest = {
        **job,
        "completed_at": datetime.now(UTC).isoformat(),
        "repo_commit": repo_commit,
        "outputs": outputs,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_count": gpu_count,
            "worker_count": workers,
            "gpus": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
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
    print(f"Outputs: {len(outputs)}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
