#!/usr/bin/env python3
"""Live read-only monitor for a RIFT-SVC TensorBoard training run."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRIC_TAGS = {
    "MCD": "val/mcd",
    "SI-SNR": "val/si_snr",
    "PSNR": "val/psnr",
    "MSE": "val/mse",
}
CHECKPOINT_STEP_RE = re.compile(r"(?:model|full)-step=(\d+)\.ckpt$")


def latest_event_file(log_dir: Path) -> Path | None:
    files = list(log_dir.glob("events.out.tfevents.*"))
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def read_scalars(log_dir: Path) -> tuple[dict[str, list], str | None]:
    event_file = latest_event_file(log_dir)
    if event_file is None:
        return {}, None
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: accumulator.Scalars(tag)
        for tag in accumulator.Tags().get("scalars", [])
    }, event_file.name


def checkpoint_info(directory: Path, prefix: str) -> tuple[int | None, int, int]:
    latest_step = None
    count = 0
    total_bytes = 0
    for path in directory.glob(f"{prefix}-step=*.ckpt"):
        match = CHECKPOINT_STEP_RE.match(path.name)
        if not match:
            continue
        count += 1
        total_bytes += path.stat().st_size
        step = int(match.group(1))
        latest_step = step if latest_step is None else max(latest_step, step)
    return latest_step, count, total_bytes


def gpu_status() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        fields = [field.strip() for field in result.stdout.strip().split(",")]
        if len(fields) != 5:
            return result.stdout.strip()
        util, used, total, temp, power = fields
        return f"{util}% util | {used}/{total} MiB | {temp}°C | {power} W"
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable ({exc})"


def process_status(run_name: str) -> str:
    try:
        result = subprocess.run(
            ["pgrep", "-af", f"train.py.*{run_name}"],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        return f"running ({len(lines)} matching processes)" if lines else "not found"
    except OSError as exc:
        return f"unavailable ({exc})"


def disk_status(path: Path) -> str:
    usage = shutil.disk_usage(path)
    return f"{usage.free / 1024**3:.1f} GiB free / {usage.total / 1024**3:.1f} GiB"


def format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def render(args: argparse.Namespace) -> str:
    scalars, event_name = read_scalars(args.log_dir)
    train_values = scalars.get("train/loss", [])
    last_train = train_values[-1] if train_values else None
    first_train = train_values[0] if train_values else None

    current_step = int(last_train.step) if last_train else 0
    rate = None
    if first_train and last_train and last_train.wall_time > first_train.wall_time:
        elapsed = last_train.wall_time - first_train.wall_time
        if last_train.step > first_train.step:
            rate = (last_train.step - first_train.step) / elapsed
    eta = "-"
    if rate and current_step < args.max_steps:
        eta_seconds = (args.max_steps - current_step) / rate
        eta = f"{eta_seconds / 3600:.2f} h"

    metric_values: dict[str, dict[int, float]] = {}
    for name, tag in METRIC_TAGS.items():
        metric_values[name] = {
            int(item.step): float(item.value) for item in scalars.get(tag, [])
        }
    metric_steps = sorted(
        set.intersection(*(set(values) for values in metric_values.values()))
    ) if all(metric_values.values()) else []

    weight_step, weight_count, weight_bytes = checkpoint_info(args.weights_dir, "model")
    full_step, full_count, full_bytes = checkpoint_info(args.full_dir, "full")

    lines = [
        f"RIFT-SVC live monitor | {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"Run: {args.run_name}",
        f"Progress: {current_step:,}/{args.max_steps:,} | rate {rate:.2f} step/s | ETA {eta}"
        if rate
        else f"Progress: {current_step:,}/{args.max_steps:,} | rate - | ETA -",
        f"Train loss: {format_metric(float(last_train.value) if last_train else None)}"
        f" | process: {process_status(args.run_name)}",
        f"GPU: {gpu_status()}",
        f"Data disk: {disk_status(Path('/root/autodl-tmp'))}"
        f" | AutoFS: {disk_status(Path('/root/autodl-fs'))}",
        f"Weights: step {weight_step or '-'} | {weight_count} files"
        f" | {weight_bytes / 1024**3:.1f} GiB",
        f"Full resume: step {full_step or '-'} | {full_count} files"
        f" | {full_bytes / 1024**3:.1f} GiB",
        "",
        "Step | MCD ↓ | SI-SNR ↑ | PSNR ↑ | MSE ↓",
        "-----|-------|----------|--------|------",
    ]
    steps_to_show = (
        metric_steps
        if args.validation_rows <= 0
        else metric_steps[-args.validation_rows :]
    )
    for step in steps_to_show:
        lines.append(
            f"{step} | {metric_values['MCD'][step]:.6f}"
            f" | {metric_values['SI-SNR'][step]:.6f}"
            f" | {metric_values['PSNR'][step]:.6f}"
            f" | {metric_values['MSE'][step]:.6f}"
        )
    if not metric_steps:
        lines.append("waiting for complete validation metrics...")
    if event_name:
        lines.append(f"Event file: {event_name}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default="finetune_v3-dit-1024-16_scratch_40000steps-lr0.00003",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(
            "/root/autodl-fs/RIFT-SVC-runs/"
            "finetune_v3-dit-1024-16_scratch_40000steps-lr0.00003/logs"
        ),
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path(
            "/root/autodl-fs/RIFT-SVC-runs/"
            "finetune_v3-dit-1024-16_scratch_40000steps-lr0.00003/weights"
        ),
    )
    parser.add_argument(
        "--full-dir",
        type=Path,
        default=Path(
            "/root/autodl-fs/RIFT-SVC-runs/"
            "finetune_v3-dit-1024-16_scratch_40000steps-lr0.00003/full_resume"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument(
        "--validation-rows",
        type=int,
        default=0,
        help="number of recent rows; 0 displays all rows from step 0",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    try:
        while True:
            if sys.stdout.isatty():
                sys.stdout.write("\033[2J\033[H")
            print(render(args), flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nmonitor stopped")


if __name__ == "__main__":
    main()
