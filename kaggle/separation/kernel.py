"""Kaggle script kernel for one pinned MSST separation profile."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
RUNTIME_ROOT = Path("/kaggle/temp/rift-separation-runtime")
OUTPUT_ROOT = WORK_ROOT / "separation-output"
MSST_REPO = "__MSST_REPO__"
MSST_COMMIT = "__MSST_COMMIT__"
PROFILES = "__SEPARATION_PROFILES__"

PIP_REQUIREMENTS = [
    "ml-collections>=0.1.1",
    "omegaconf>=2.2.3",
    "rotary-embedding-torch==0.3.5",
    "einops>=0.8.1",
    "beartype>=0.14",
    "librosa>=0.10",
    "soundfile>=0.12",
    "tqdm>=4.67",
    "matplotlib>=3.8",
    "huggingface-hub>=0.23",
]


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
    profile_name = job["profile"]
    if profile_name not in PROFILES:
        raise ValueError(f"unsupported separation profile: {profile_name}")
    profile = PROFILES[profile_name]
    if job.get("profile_config") != profile:
        raise RuntimeError("job profile does not match the pinned kernel profile")

    input_audio = job_path.parent / job["input_name"]
    if not input_audio.is_file() or sha256(input_audio) != job["input_sha256"]:
        raise RuntimeError("input audio is missing or its SHA-256 does not match")

    msst_dir = RUNTIME_ROOT / "msst"
    model_dir = RUNTIME_ROOT / "model"
    input_dir = RUNTIME_ROOT / "input"
    run(["git", "clone", MSST_REPO, str(msst_dir)])
    run(["git", "checkout", MSST_COMMIT], cwd=msst_dir)
    run([sys.executable, "-m", "pip", "install", "-q", *PIP_REQUIREMENTS])
    sys.path.insert(0, str(msst_dir))

    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    from inference import run_folder
    from utils.model_utils import load_start_checkpoint
    from utils.settings import get_model_from_config

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is unavailable; enable a GPU accelerator")

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=profile["model_repo"],
            filename=profile["checkpoint"],
            revision=profile["model_revision"],
            local_dir=str(model_dir),
        )
    )
    config_path = Path(
        hf_hub_download(
            repo_id=profile["model_repo"],
            filename=profile["config"],
            revision=profile["model_revision"],
            local_dir=str(model_dir),
        )
    )
    if checkpoint_path.stat().st_size < 100 * 1024 * 1024:
        raise RuntimeError(f"checkpoint looks incomplete: {checkpoint_path}")

    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / input_audio.name).symlink_to(input_audio)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model, config = get_model_from_config(profile["architecture"], str(config_path))
    if profile.get("normalize_small_config"):
        if "inference" not in config:
            config.inference = {}
        config.inference.dim_t = int(config.audio.dim_t)
        config.inference.num_overlap = 1
    config.inference.batch_size = 1

    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    args = SimpleNamespace(
        start_check_point=str(checkpoint_path),
        model_type="bs_roformer",
        load_only_compatible_weights=False,
        lora_checkpoint_loralib="",
        input_folder=str(input_dir),
        store_dir=str(OUTPUT_ROOT),
        extract_instrumental=True,
        disable_detailed_pbar=False,
        force_cpu=False,
        pcm_type="FLOAT",
        flac_file=False,
        use_tta=False,
        bigshifts=1,
        filename_template=f"{{file_name}}__{profile['label']}_{{instr}}",
        draw_spectro=0,
    )
    load_start_checkpoint(args, model, checkpoint, type_="inference")
    model = model.to(device).eval()
    run_folder(model, args, config, device, verbose=True)

    rename_to = profile.get("rename_instrumental")
    if rename_to:
        for path in sorted(OUTPUT_ROOT.rglob("*_instrumental.wav")):
            path.rename(
                path.with_name(
                    path.name.replace("_instrumental.wav", f"_{rename_to}.wav")
                )
            )

    output_files = sorted(OUTPUT_ROOT.rglob("*.wav"))
    if len(output_files) != 2:
        raise RuntimeError(f"expected two WAV outputs, found {output_files}")
    outputs = []
    for path in output_files:
        info = sf.info(path)
        if info.subtype != "FLOAT":
            raise RuntimeError(f"expected Float WAV output, got {info}")
        outputs.append(
            {
                "name": path.name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration": info.duration,
                "subtype": info.subtype,
            }
        )

    manifest = {
        **job,
        "completed_at": datetime.now(UTC).isoformat(),
        "msst_commit": MSST_COMMIT,
        "checkpoint_sha256": sha256(checkpoint_path),
        "outputs": outputs,
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
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
