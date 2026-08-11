# Kaggle inference

This is an inference-only workflow. The Kaggle notebook should use a GPU
accelerator and internet access, and should keep the HF token in Kaggle
Secrets as `HF_TOKEN`.

## Setup

Clone the code repository into `/kaggle/working`, then install only the
inference dependencies. `requirements-kaggle.txt` deliberately does not
install `torch`, `torchaudio`, or `torchcodec`; Kaggle provides the CUDA
PyTorch stack and `infer.py` uses SoundFile for audio I/O.

```bash
cd /kaggle/working
git clone https://github.com/OWNER/RIFT-SVC.git
cd RIFT-SVC
python -m pip install -q -r requirements-kaggle.txt
```

Download the private fine-tuned checkpoint and the official auxiliary models.
The checkpoint repository should contain only the selected inference model,
for example `rift25k.ckpt`; do not upload training data or resume files.

```bash
python scripts/download_inference_assets.py \
  --model-repo OWNER/rift-svc-luzao-25k \
  --output-dir /kaggle/working/rift-assets \
  --expected-sha256 EXPECTED_SHA256
```

The token is read by `huggingface_hub` from the Kaggle Secret/environment and
is never written into this repository or the command line. The download
script creates `inference-assets.json` and validates the checkpoint hash.

## Inference

Use batch size 1 on a T4 or P100. Adjust the guidance and robust-F0 values as
an explicit candidate matrix instead of silently changing them between runs.
The Runtime loads all neural networks once per notebook process, and `--seed`
makes candidates reproducible.

```bash
python infer.py \
  --model /kaggle/working/rift-assets/model/rift25k.ckpt \
  --assets-dir /kaggle/working/rift-assets/pretrained \
  --input /kaggle/input/INPUT_DATA/lead-vocal.wav \
  --output /kaggle/working/outputs/lead-vocal__steps64-ds0.2-spk0.8-rf1.flac \
  --speaker target \
  --device cuda \
  --infer-steps 64 \
  --batch-size 1 \
  --ds-cfg-strength 0.2 \
  --spk-cfg-strength 0.8 \
  --cfg-rescale 0.7 \
  --robust-f0 1 \
  --seed 7
```

The default FLAC output is explicit 24-bit PCM. For float intermediate work,
use a `.wav` path together with `--output-subtype FLOAT`. Before downloading
the result, verify that the output is non-empty, 44.1 kHz, and has the expected
duration. Keep outputs under `/kaggle/working`; Kaggle runtime storage is temporary.
