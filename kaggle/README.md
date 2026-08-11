# Kaggle 推理

这是一个只做推理的 Kaggle 工作流。创建带 GPU 的 Notebook 并开启 Internet，
把 Hugging Face token 放在 Kaggle Secret 中；下载工具会通过环境变量读取它。

## 安装

```bash
cd /kaggle/working
git clone https://github.com/OWNER/RIFT-SVC.git
cd RIFT-SVC
python -m pip install -q -r requirements-kaggle.txt
```

`requirements-kaggle.txt` 不包含 `torch`、`torchaudio` 或 `torchcodec`，避免
覆盖 Kaggle 已配置好的 CUDA PyTorch。

下载官方推理资源和模型。模型仓库可以只包含最终使用的 checkpoint，例如
`rift25k.ckpt`；不需要上传数据或 resume checkpoint：

```bash
python scripts/download_inference_assets.py \
  --model-repo OWNER/rift-svc-luzao-25k \
  --output-dir /kaggle/working/rift-assets \
  --expected-sha256 EXPECTED_SHA256
```

脚本会生成 `inference-assets.json`，并在提供 hash 时校验模型完整性。

## 推理

```bash
python infer.py \
  --model /kaggle/working/rift-assets/model/rift25k.ckpt \
  --assets-dir /kaggle/working/rift-assets/pretrained \
  --input /kaggle/input/INPUT_DATA/lead-vocal.wav \
  --output /kaggle/working/outputs/lead-vocal__steps64-ds0.2-spk0.8-rf1.flac \
  --speaker target \
  --device cuda \
  --infer-steps 64 \
  --ds-cfg-strength 0.2 \
  --spk-cfg-strength 0.8 \
  --cfg-rescale 0.7 \
  --robust-f0 1 \
  --seed 7
```

模型在一个 Notebook 进程内只加载一次；CLI 每次转换一个文件。`--seed`
用于生成可复现的候选版本。默认 FLAC 是 24-bit PCM；继续做混音时可以将
`.wav` 与 `--output-subtype FLOAT` 一起使用。

输出写入 `/kaggle/working`，下载前请确认文件非空、采样率为 44.1 kHz、
时长符合输入。Kaggle 的工作目录是临时的，请将最终模型和结果另行保存。
