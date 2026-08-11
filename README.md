# RIFT-SVC Inference

This checkout contains the inference path for RIFT-SVC: ContentVec and RMVPE
feature extraction, RIFT Euler sampling, NSF-HiFiGAN vocoding, and lossless
audio output. Training, dataset preparation, monitoring, and GUI code are not
part of this repository.

## Requirements

- Python 3.11, 3.12, or 3.13
- PyTorch 2.9 or newer with a matching `torchaudio`
- CUDA is recommended for practical inference; CPU also works for validation

Install with `uv`:

```bash
uv sync --extra dev
source .venv/bin/activate
```

On Kaggle, keep Kaggle's preinstalled CUDA PyTorch packages and install the
small dependency list from `requirements-kaggle.txt` instead. See
[`kaggle/README.md`](kaggle/README.md).

## Inference assets

Download the official ContentVec, RMVPE, and NSF-HiFiGAN files into
`pretrained/`:

```bash
python scripts/download_inference_assets.py --modules-only --output-dir .
```

The fine-tuned checkpoint is supplied separately. It must be a full RIFT-SVC
checkpoint containing the model weights and the speaker/dataset metadata, for
example `ckpts/rift25k.ckpt`.

## CLI

```bash
python infer.py \
  --model ckpts/rift25k.ckpt \
  --input vocals.flac \
  --output outputs/vocals__steps64-ds0.2-spk0.8.flac \
  --speaker target \
  --device cuda \
  --infer-steps 64 \
  --ds-cfg-strength 0.2 \
  --spk-cfg-strength 0.8 \
  --cfg-rescale 0.7 \
  --robust-f0 1 \
  --seed 7
```

The runtime loads all models once, slices the input at silence, processes
segments sequentially, and overlap-adds them back into the original duration.
The speaker name must exist in the checkpoint metadata. `--key-shift` changes
the input pitch in semitones.

Useful controls:

- `--infer-steps`: Euler sampling steps; higher values cost more time.
- `--ds-cfg-strength`: content guidance, normally around `0.2`.
- `--spk-cfg-strength`: speaker guidance, normally around `0.8`.
- `--cfg-rescale`: guidance rescaling, default `0.7`.
- `--robust-f0 0|1|2`: raw, light, or stronger F0 repair.
- `--seed`: deterministic noise per segment for repeatable candidates.
- `--no-use-fp16`: disable CUDA half precision when debugging.

Run `python infer.py --help` for the complete option list.

## Audio output

WAV and FLAC output default to explicit 24-bit PCM, avoiding SoundFile's
implicit PCM16 default. Use float WAV for additional offline processing:

```bash
python infer.py ... --output outputs/vocals-float.wav --output-subtype FLOAT
```

The source audio should be a clean vocal stem. The loader resamples it to the
sample rate recorded in the checkpoint, mixes multi-channel input to mono,
and preserves the converted file's duration.
