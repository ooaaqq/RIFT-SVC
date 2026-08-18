"""Command-line entry point for RIFT-SVC inference."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from rift_svc.inference.audio import write_audio
from rift_svc.inference.runtime import InferenceRuntime


@click.command()
@click.option("--model", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--input", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--speaker", required=True)
@click.option("--assets-dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--key-shift", type=int, default=0, show_default=True)
@click.option("--device", default=None)
@click.option("--infer-steps", type=click.IntRange(min=2), default=32, show_default=True)
@click.option("--ds-cfg-strength", type=float, default=0.2, show_default=True)
@click.option("--spk-cfg-strength", type=float, default=0.8, show_default=True)
@click.option("--cfg-rescale", type=float, default=0.7, show_default=True)
@click.option("--cvec-downsample-rate", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--target-loudness", type=float, default=-18.0, show_default=True)
@click.option("--restore-loudness/--no-restore-loudness", default=True)
@click.option("--fade-duration", type=float, default=20.0, show_default=True)
@click.option("--robust-f0", type=click.IntRange(0, 2), default=0, show_default=True)
@click.option("--slicer-threshold", type=float, default=-30.0, show_default=True)
@click.option("--slicer-min-length", type=click.IntRange(min=1), default=3000, show_default=True)
@click.option("--slicer-min-interval", type=click.IntRange(min=1), default=100, show_default=True)
@click.option("--slicer-hop-size", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--slicer-max-sil-kept", type=click.IntRange(min=1), default=200, show_default=True)
@click.option("--use-fp16/--no-use-fp16", default=True)
@click.option("--seed", type=int, default=7, show_default=True)
@click.option(
    "--output-subtype",
    type=click.Choice(["PCM_24", "FLOAT", "PCM_16"], case_sensitive=True),
    default="FLOAT",
    show_default=True,
    help="SoundFile subtype for the intermediate inference output.",
)
def main(
    model,
    input,
    output,
    speaker,
    assets_dir,
    key_shift,
    device,
    infer_steps,
    ds_cfg_strength,
    spk_cfg_strength,
    cfg_rescale,
    cvec_downsample_rate,
    target_loudness,
    restore_loudness,
    fade_duration,
    robust_f0,
    slicer_threshold,
    slicer_min_length,
    slicer_min_interval,
    slicer_hop_size,
    slicer_max_sil_kept,
    use_fp16,
    seed,
    output_subtype,
):
    """Convert one audio file with a single reusable inference runtime."""
    logging.basicConfig(level=logging.INFO)
    runtime = InferenceRuntime(
        model,
        device=device,
        use_fp16=use_fp16,
        assets_dir=assets_dir,
    )
    click.echo(f"Converting on {runtime.device}; sample rate {runtime.sample_rate} Hz")
    audio = runtime.convert(
        input,
        speaker=speaker,
        key_shift=key_shift,
        infer_steps=infer_steps,
        ds_cfg_strength=ds_cfg_strength,
        spk_cfg_strength=spk_cfg_strength,
        cfg_rescale=cfg_rescale,
        cvec_downsample_rate=cvec_downsample_rate,
        target_loudness=target_loudness,
        restore_loudness=restore_loudness,
        fade_duration_ms=fade_duration,
        robust_f0=robust_f0,
        slicer_threshold=slicer_threshold,
        slicer_min_length=slicer_min_length,
        slicer_min_interval=slicer_min_interval,
        slicer_hop_size=slicer_hop_size,
        slicer_max_sil_kept=slicer_max_sil_kept,
        seed=seed,
    )
    write_audio(output, audio, runtime.sample_rate, subtype=output_subtype)
    click.echo(f"Wrote {output} ({len(audio) / runtime.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
