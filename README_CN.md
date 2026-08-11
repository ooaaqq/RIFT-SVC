# RIFT-SVC 推理

这个仓库只保留 RIFT-SVC 的推理链路：ContentVec、RMVPE 特征提取，RIFT
Euler 采样，NSF-HiFiGAN 声码，以及高精度音频输出。训练、数据集预处理、
监控和 GUI 不在当前仓库中。

## 环境要求

- Python 3.11、3.12 或 3.13
- PyTorch 2.9 以上，并使用匹配版本的 `torchaudio`
- 推荐使用 CUDA；CPU 可用于验证和短音频推理

使用 `uv` 安装：

```bash
uv sync --extra dev
source .venv/bin/activate
```

Kaggle 使用平台预装的 CUDA PyTorch，只安装
`requirements-kaggle.txt` 中的其他依赖，详见
[`kaggle/README.md`](kaggle/README.md)。

## 推理资源

将官方 ContentVec、RMVPE 和 NSF-HiFiGAN 文件下载到 `pretrained/`：

```bash
python scripts/download_inference_assets.py --modules-only --output-dir .
```

微调后的 checkpoint 单独提供。它必须是包含模型权重、说话人信息和数据集
元信息的完整 RIFT-SVC checkpoint，例如 `ckpts/rift25k.ckpt`。

## 命令行推理

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

运行时只加载一次所有模型，在静音处切分输入，按顺序处理各段，再将结果
交叠相加回原始时长。`--speaker` 必须存在于 checkpoint 元信息中；
`--key-shift` 以半音为单位调整输入音高。

常用参数：

- `--infer-steps`：Euler 采样步数，越高越慢。
- `--ds-cfg-strength`：内容引导强度，通常从 `0.2` 开始。
- `--spk-cfg-strength`：说话人引导强度，通常从 `0.8` 开始。
- `--cfg-rescale`：引导重缩放，默认 `0.7`。
- `--robust-f0 0|1|2`：原始、轻度或更强的 F0 修复。
- `--seed`：为每个分段固定噪声，便于重复试听。
- `--no-use-fp16`：调试时关闭 CUDA 半精度。

完整参数可运行 `python infer.py --help` 查看。

## 音频输出

WAV 和 FLAC 默认明确写成 24-bit PCM，避免 SoundFile 默认量化为 PCM16。
需要继续离线处理时可输出浮点 WAV：

```bash
python infer.py ... --output outputs/vocals-float.wav --output-subtype FLOAT
```

输入建议使用干净的人声 stem。程序会按照 checkpoint 记录的采样率重采样，
多声道输入会混合为单声道，并保持转换结果时长。
