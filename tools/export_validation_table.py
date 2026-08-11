#!/usr/bin/env python3
"""Export validation metrics from TensorBoard event files as Markdown or CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRICS = {
    "MCD": "val/mcd",
    "SI-SNR": "val/si_snr",
    "PSNR": "val/psnr",
    "MSE": "val/mse",
}


def collect_metrics(log_dir: Path) -> dict[str, dict[int, float]]:
    event_files = sorted(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event files found under {log_dir}")

    values = {name: {} for name in METRICS}
    for event_file in event_files:
        try:
            accumulator = EventAccumulator(
                str(event_file), size_guidance={"scalars": 0}
            )
            accumulator.Reload()
        except Exception as exc:  # keep reading other event files
            print(f"warning: cannot read {event_file}: {exc}", file=sys.stderr)
            continue

        scalar_tags = set(accumulator.Tags().get("scalars", []))
        for name, tag in METRICS.items():
            if tag not in scalar_tags:
                continue
            for item in accumulator.Scalars(tag):
                values[name][int(item.step)] = float(item.value)

    if not all(values[name] for name in METRICS):
        missing = [name for name, items in values.items() if not items]
        raise RuntimeError(f"missing metric tags: {', '.join(missing)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/finetune_v3-dit-1024-16_30000steps-lr0.00003"),
    )
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument("--output", type=Path, help="write to a file instead of stdout")
    args = parser.parse_args()

    values = collect_metrics(args.log_dir)
    steps = sorted(set.intersection(*(set(items) for items in values.values())))

    if args.format == "csv":
        import io

        buffer = io.StringIO()
        csv_writer = csv.writer(buffer, lineterminator="\n")
        csv_writer.writerow(["Step", *METRICS])
        for step in steps:
            csv_writer.writerow([step, *(f"{values[name][step]:.6f}" for name in METRICS)])
        output = buffer.getvalue()
    else:
        rows = [
            "| Step | MCD ↓ | SI-SNR ↑ | PSNR ↑ | MSE ↓ |",
            "|---:|---:|---:|---:|---:|",
        ]
        for step in steps:
            rows.append(
                f"| {step} | {values['MCD'][step]:.6f} | "
                f"{values['SI-SNR'][step]:.6f} | {values['PSNR'][step]:.6f} | "
                f"{values['MSE'][step]:.6f} |"
            )
        output = "\n".join(rows) + "\n"

    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
