#!/usr/bin/env python3
"""Render true-stereo M7 impulse-response returns for vocal mix sends."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import oaconvolve

from rift_svc.audio_tools import (
    concise_cli,
    ensure_new_paths,
    read_float_audio,
    write_float_wav_new,
    write_json_new,
)

DEFAULT_IR_LIBRARY = Path(
    os.environ.get(
        "RIFT_M7_IR_DIR",
        "/home/elvedon/Music/露早/90. Shared Resources/M7/"
        "10. Samplicity-Bricasti-2023-10-44K-Left-Right",
    )
)
PRESET_FILES = {
    "small-vox-room": "3 Rooms 17 Small Vox Room, 44K",
    "studio-b-close": "3 Rooms 02 Studio B Close, 44K",
}


def resolve_ir_pair(
    preset: str,
    library: Path,
    ir_left: Path | None,
    ir_right: Path | None,
) -> tuple[Path, Path]:
    """Resolve an explicit pair or one of the two everyday vocal presets."""
    if (ir_left is None) != (ir_right is None):
        raise ValueError("--ir-left and --ir-right must be supplied together")
    if ir_left is not None and ir_right is not None:
        return ir_left, ir_right
    stem = PRESET_FILES[preset]
    return library / f"{stem} L.wav", library / f"{stem} R.wav"


def read_audio(path: Path, expected_sr: int, *, stereo_required: bool = False) -> np.ndarray:
    """Read a finite mono/stereo float array at the requested sample rate."""
    if not path.is_file():
        raise FileNotFoundError(path)
    audio, sample_rate = read_float_audio(path)
    if sample_rate != expected_sr:
        raise ValueError(f"{path} has {sample_rate} Hz; expected {expected_sr} Hz")
    channels = audio.shape[1]
    if channels not in (1, 2) or stereo_required and channels != 2:
        expected = "stereo" if stereo_required else "mono or stereo"
        raise ValueError(f"{path} has {channels} channels; expected {expected}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def read_ir(
    path: Path,
    expected_sr: int,
    fraction: float,
    fade_ms: float,
) -> np.ndarray:
    """Read an IR unchanged unless explicit trimming or fading is requested."""
    ir = read_audio(path, expected_sr, stereo_required=True)
    length = max(8, min(ir.shape[0], round(ir.shape[0] * fraction)))
    ir = ir[:length].copy()
    fade_length = min(round(fade_ms * expected_sr / 1000.0), length)
    if fade_length > 0:
        ir[-fade_length:] *= np.linspace(
            1.0, 0.0, fade_length, dtype=np.float32
        )[:, None]
    return ir


def add_predelay(ir: np.ndarray, predelay_samples: int) -> np.ndarray:
    if predelay_samples <= 0:
        return ir
    silence = np.zeros((predelay_samples, ir.shape[1]), dtype=np.float32)
    return np.concatenate((silence, ir), axis=0)


def true_stereo_convolve(
    audio: np.ndarray, ir_left: np.ndarray, ir_right: np.ndarray
) -> np.ndarray:
    """Convolve mono/stereo input through the left/right true-stereo IR pair."""
    if audio.shape[1] == 1:
        input_left = audio[:, 0] / np.sqrt(2.0)
        input_right = input_left
    else:
        input_left = audio[:, 0]
        input_right = audio[:, 1]

    wet_left = (
        oaconvolve(input_left, ir_left[:, 0], mode="full")
        + oaconvolve(input_right, ir_right[:, 0], mode="full")
    )
    wet_right = (
        oaconvolve(input_left, ir_left[:, 1], mode="full")
        + oaconvolve(input_right, ir_right[:, 1], mode="full")
    )
    return np.column_stack((wet_left, wet_right)).astype(np.float32, copy=False)


def render_return(
    name: str,
    audio: np.ndarray,
    ir_left: np.ndarray,
    ir_right: np.ndarray,
    wet_db: float,
    output: Path,
    sample_rate: int,
    keep_tail: bool,
) -> dict[str, object]:
    wet = np.nan_to_num(
        true_stereo_convolve(audio, ir_left, ir_right),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    full_output_samples = wet.shape[0]
    dry_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if dry_rms <= 1e-12:
        raise ValueError(f"{name} send is silent; cannot calibrate wet level")
    calibration_wet = wet[: audio.shape[0]]
    wet_rms_before = float(
        np.sqrt(np.mean(np.square(calibration_wet, dtype=np.float64)))
    )
    if wet_rms_before <= 1e-12:
        raise ValueError(f"{name} convolution is silent; check the selected IR pair")
    target_rms = dry_rms * 10.0 ** (wet_db / 20.0)
    scale = target_rms / wet_rms_before
    wet *= np.float32(scale)

    peak = float(np.max(np.abs(wet)))
    tail = wet[audio.shape[0] :]
    tail_peak = float(np.max(np.abs(tail), initial=0.0))
    tail_rms = float(
        np.sqrt(np.mean(np.square(tail, dtype=np.float64))) if tail.size else 0.0
    )
    output_audio = wet if keep_tail else wet[: audio.shape[0]]
    if not keep_tail and tail_peak > 1e-5:
        print(
            f"[{name}] warning: truncating IR tail with peak {tail_peak:.6f}; "
            "ensure the send contains enough trailing silence or use --keep-tail",
            file=sys.stderr,
            flush=True,
        )

    write_float_wav_new(output, output_audio, sample_rate)
    calibrated_rms = float(
        np.sqrt(
            np.mean(
                np.square(wet[: audio.shape[0]], dtype=np.float64)
            )
        )
    )
    output_rms = float(
        np.sqrt(np.mean(np.square(output_audio, dtype=np.float64)))
    )
    result = {
        "name": name,
        "output": str(output),
        "sample_rate": sample_rate,
        "input_seconds": audio.shape[0] / sample_rate,
        "output_seconds": output_audio.shape[0] / sample_rate,
        "full_convolution_seconds": full_output_samples / sample_rate,
        "tail_policy": "full" if keep_tail else "match-input",
        "input_channels": audio.shape[1],
        "target_wet_db_relative_rms": wet_db,
        "dry_rms": dry_rms,
        "wet_rms_before_scale": wet_rms_before,
        "calibration_window_seconds": audio.shape[0] / sample_rate,
        "calibrated_wet_rms": calibrated_rms,
        "output_wet_rms": output_rms,
        "scale": scale,
        "wet_peak": peak,
        "tail_peak": tail_peak,
        "tail_rms": tail_rms,
        "tail_truncated": not keep_tail and tail_peak > 1e-5,
    }
    print(
        f"[{name}] channels={audio.shape[1]} dry_rms={dry_rms:.6f} "
        f"wet_rms={calibrated_rms:.6f} "
        f"({20 * np.log10(calibrated_rms / dry_rms):.2f} dB) "
        f"peak={result['wet_peak']:.4f} -> {output}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render length-safe true-stereo returns from a selected M7 IR pair."
    )
    parser.add_argument("--main-send", type=Path, required=True)
    parser.add_argument("--harmony-send", type=Path)
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_FILES),
        default="small-vox-room",
        help="everyday vocal starting point (default: small-vox-room)",
    )
    parser.add_argument("--ir-library", type=Path, default=DEFAULT_IR_LIBRARY)
    parser.add_argument("--ir-left", type=Path)
    parser.add_argument("--ir-right", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--ir-fraction", type=float, default=1.0)
    parser.add_argument("--ir-fade-ms", type=float, default=0.0)
    parser.add_argument("--predelay-ms", type=float, required=True)
    parser.add_argument("--main-wet-db", type=float, required=True)
    parser.add_argument("--harmony-wet-db", type=float)
    parser.add_argument(
        "--keep-tail",
        action="store_true",
        help="keep the complete IR tail instead of matching the input sample count",
    )
    args = parser.parse_args()

    if not 0.0 < args.ir_fraction <= 1.0:
        parser.error("--ir-fraction must be greater than 0 and at most 1")
    if args.predelay_ms < 0.0:
        parser.error("--predelay-ms cannot be negative")
    if args.ir_fade_ms < 0.0:
        parser.error("--ir-fade-ms cannot be negative")
    if args.sample_rate <= 0:
        parser.error("--sample-rate must be positive")
    if (args.harmony_send is None) != (args.harmony_wet_db is None):
        parser.error("--harmony-send and --harmony-wet-db must be supplied together")

    try:
        ir_left_path, ir_right_path = resolve_ir_pair(
            args.preset, args.ir_library, args.ir_left, args.ir_right
        )
    except ValueError as exc:
        parser.error(str(exc))

    main_send = read_audio(args.main_send, args.sample_rate)
    harmony_send = (
        read_audio(args.harmony_send, args.sample_rate)
        if args.harmony_send is not None
        else None
    )
    for name, send in (("main", main_send), ("harmony", harmony_send)):
        if send is not None and float(np.max(np.abs(send), initial=0.0)) <= 1e-12:
            raise ValueError(f"{name} send is silent")
    if harmony_send is not None and main_send.shape[0] != harmony_send.shape[0]:
        raise ValueError(
            "main and harmony sends must have the same sample count; "
            f"got {main_send.shape[0]} and {harmony_send.shape[0]}"
        )
    output_paths = [
        args.output_dir / "m7-main-wet.wav",
        args.output_dir / "m7-return-summary.json",
    ]
    if harmony_send is not None:
        output_paths.append(args.output_dir / "m7-harmony-wet.wav")
    ensure_new_paths(output_paths)
    ir_left = add_predelay(
        read_ir(
            ir_left_path,
            args.sample_rate,
            args.ir_fraction,
            args.ir_fade_ms,
        ),
        round(args.predelay_ms * args.sample_rate / 1000.0),
    )
    ir_right = add_predelay(
        read_ir(
            ir_right_path,
            args.sample_rate,
            args.ir_fraction,
            args.ir_fade_ms,
        ),
        round(args.predelay_ms * args.sample_rate / 1000.0),
    )

    results = [
        render_return(
            "main",
            main_send,
            ir_left,
            ir_right,
            args.main_wet_db,
            args.output_dir / "m7-main-wet.wav",
            args.sample_rate,
            args.keep_tail,
        )
    ]
    if harmony_send is not None and args.harmony_wet_db is not None:
        results.append(
            render_return(
                "harmony",
                harmony_send,
                ir_left,
                ir_right,
                args.harmony_wet_db,
                args.output_dir / "m7-harmony-wet.wav",
                args.sample_rate,
                args.keep_tail,
            )
        )
    summary = {
        "preset": args.preset if args.ir_left is None else None,
        "ir_left": str(ir_left_path.resolve()),
        "ir_right": str(ir_right_path.resolve()),
        "ir_fraction": args.ir_fraction,
        "ir_fade_ms": args.ir_fade_ms,
        "predelay_ms": args.predelay_ms,
        "tail_policy": "full" if args.keep_tail else "match-input",
        "results": results,
    }
    summary_path = args.output_dir / "m7-return-summary.json"
    write_json_new(summary_path, summary)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    concise_cli(main, "render_m7_returns")
