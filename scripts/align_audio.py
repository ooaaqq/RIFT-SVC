#!/usr/bin/env python3
"""CLI for detecting, checking, and applying audio alignment anchors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rift_svc.alignment import (
    Anchor,
    alignment_report,
    apply_alignment,
    detect_alignment,
    read_anchors,
)
from rift_svc.audio_tools import (
    concise_cli,
    ensure_new_paths,
    format_time,
    write_float_wav_new,
    write_json_new,
    write_text_new,
)


def anchor_file_text(anchors: list[Anchor], report: dict[str, Any]) -> str:
    lines = [
        "# Suggested affine alignment: SOURCE_TIME TARGET_TIME",
        f"# safe_to_apply: {str(report['safe_to_apply']).lower()}",
        f"# global correlation: {report['global_correlation']:.6f}",
        (
            "# fitted source seconds per target second: "
            f"{report['fit']['source_seconds_per_target_second']:.12f}"
        ),
    ]
    for reason in report["unsafe_reasons"]:
        lines.append(f"# unsafe: {reason}")
    lines.extend(
        f"{format_time(anchor.source)} {format_time(anchor.target)}"
        for anchor in anchors
    )
    lines.append("# Local measurements; inspect the JSON report before apply.")
    for result in report["local_windows"]:
        lines.append(
            f"# target={format_time(result['target_center'])} "
            f"source_minus_target={result['source_minus_target_ms']:+.3f}ms "
            f"corr={result['correlation']:.4f} "
            f"used={str(result['used_in_final_fit']).lower()}"
        )
    return "\n".join(lines) + "\n"


def print_alignment_report(report: dict[str, Any]) -> None:
    print(
        f"source/reference: {report['source_duration']:.6f}s / "
        f"{report['reference_duration']:.6f}s at {report['sample_rate']} Hz"
    )
    print(
        f"anchor offset: {report['anchor_offsets_ms'][0]:+.3f} to "
        f"{report['anchor_offsets_ms'][-1]:+.3f} ms; "
        f"range={report['offset_range_ms']:.3f} ms"
    )
    for index, segment in enumerate(report["segments"], 1):
        print(
            f"segment {index}: source {segment['source_start']:.3f}-"
            f"{segment['source_end']:.3f}s -> target "
            f"{segment['target_start']:.3f}-{segment['target_end']:.3f}s; "
            f"stretch={segment['stretch_percent']:+.4f}%"
        )


def add_common_paths(command: argparse.ArgumentParser) -> None:
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--reference", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect, check, or apply SOURCE_TIME TARGET_TIME alignment anchors."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    detect = commands.add_parser(
        "detect", help="detect fixed offset and affine drift, then suggest anchors"
    )
    add_common_paths(detect)
    detect.add_argument("--anchors-output", type=Path, required=True)
    detect.add_argument("--report", type=Path, required=True)
    detect.add_argument("--search-seconds", type=float, default=8.0)
    detect.add_argument("--local-window-seconds", type=float, default=40.0)
    detect.add_argument("--local-search-seconds", type=float, default=0.3)
    detect.add_argument("--min-correlation", type=float, default=0.5)
    detect.add_argument("--max-fit-residual-ms", type=float, default=15.0)
    detect.add_argument("--analysis-window-ms", type=float, default=25.0)
    detect.add_argument("--analysis-hop-ms", type=float, default=6.0)
    detect.add_argument("--edge-seconds", type=float, default=1.0)

    check = commands.add_parser("check", help="inspect an existing anchor file")
    add_common_paths(check)
    check.add_argument("--anchors-file", type=Path, required=True)
    check.add_argument("--report", type=Path)

    apply = commands.add_parser("apply", help="render an existing anchor file")
    add_common_paths(apply)
    apply.add_argument("--anchors-file", type=Path, required=True)
    apply.add_argument("--crossfade-ms", type=float, required=True)
    apply.add_argument("--max-stretch-percent", type=float, required=True)
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--report", type=Path)
    return parser


def run_detect(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive = {
        "--search-seconds": args.search_seconds,
        "--local-window-seconds": args.local_window_seconds,
        "--local-search-seconds": args.local_search_seconds,
        "--max-fit-residual-ms": args.max_fit_residual_ms,
        "--analysis-window-ms": args.analysis_window_ms,
        "--analysis-hop-ms": args.analysis_hop_ms,
    }
    if any(value <= 0 for value in positive.values()) or args.edge_seconds < 0:
        parser.error("analysis sizes must be positive and edge cannot be negative")
    if not -1.0 <= args.min_correlation <= 1.0:
        parser.error("--min-correlation must be between -1 and 1")
    ensure_new_paths([args.anchors_output, args.report])
    anchors, report = detect_alignment(
        args.input,
        args.reference,
        search_seconds=args.search_seconds,
        local_window_seconds=args.local_window_seconds,
        local_search_seconds=args.local_search_seconds,
        min_correlation=args.min_correlation,
        max_fit_residual_ms=args.max_fit_residual_ms,
        analysis_window_ms=args.analysis_window_ms,
        analysis_hop_ms=args.analysis_hop_ms,
        edge_seconds=args.edge_seconds,
    )
    write_text_new(args.anchors_output, anchor_file_text(anchors, report))
    write_json_new(args.report, report)
    print(f"anchors: {args.anchors_output}")
    print(f"report: {args.report}")
    print(
        f"global source-target offset: "
        f"{report['global_source_minus_target_ms']:+.3f} ms; "
        f"correlation={report['global_correlation']:.4f}"
    )
    print(
        f"affine ratio: {report['fit']['source_seconds_per_target_second']:.9f}; "
        f"fit residual RMS={report['fit']['residual_rms_ms']:.3f} ms; "
        f"safe_to_apply={str(report['safe_to_apply']).lower()}"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "detect":
        run_detect(args, parser)
        return

    destinations = [args.report, getattr(args, "output", None)]
    ensure_new_paths(destinations)
    anchors = read_anchors(args.anchors_file)
    if args.command == "check":
        report = alignment_report(args.input, args.reference, anchors)
    else:
        if args.crossfade_ms < 0 or args.max_stretch_percent <= 0:
            parser.error("crossfade must be non-negative and stretch limit positive")
        rendered, report = apply_alignment(
            args.input,
            args.reference,
            anchors,
            args.crossfade_ms,
            args.max_stretch_percent,
        )
        write_float_wav_new(args.output, rendered, report["sample_rate"])
        print(f"output: {args.output}")
    if args.report is not None:
        write_json_new(args.report, report)
        print(f"report: {args.report}")
    print_alignment_report(report)


if __name__ == "__main__":
    concise_cli(main, "align_audio")
