# RIFT-SVC 推理工作区

这是露早 1024-16、25K checkpoint 的私有推理仓库。只维护：

- RIFT-SVC 本地推理核心；
- Kaggle T4 自动推理与分离；
- 从音源、局部修复、混音到视频封装的制作工作流。

不维护训练、数据预处理、GUI 或通用多模型兼容层。模型公式、RMVPE、Mel/STFT
和 HiFi-GAN 结构保持与现有权重一致。

## 仓库结构

```text
infer.py                    本地单文件推理入口
rift_svc/                   RIFT、RMVPE、HiFi-GAN 与推理运行时
scripts/kaggle_rift.py      本地提交 RIFT 云端任务
scripts/kaggle_separate.py  本地提交 BS/MelBand-RoFormer 任务
kaggle/rift/kernel.py       Kaggle RIFT Script Kernel
kaggle/separation/kernel.py Kaggle 分离 Script Kernel
docs/workflow/              通用歌曲制作流程与歌曲文档模板
tests/                      推理行为和云端提交器的轻量测试
```

checkpoint、辅助模型、输入音频和输出文件均不进入 Git。

## 环境

首次进入仓库：

```bash
direnv allow
uv sync --extra dev
```

之后进入目录时，direnv 会自动加载 Nix Flake，并激活已有的 `.venv`。也可以手动：

```bash
nix develop
uv sync --extra dev
```

Flake 提供 Python 3.11、FFmpeg、Kaggle CLI、uv、Ruff 和编译依赖；uv 只管理
Python 包。

## 日常使用

RIFT 云端推理：

```bash
uv run python scripts/kaggle_rift.py vocals.wav \
  --output-dir /path/to/results
```

辅助分离：

```bash
uv run python scripts/kaggle_separate.py vocals.wav \
  --model anvuew-dereverb-22.5050 \
  --output-dir /path/to/results
```

两者都会更新私有输入 Dataset、提交固定 T4 Script Kernel、等待完成、下载结果，
并校验任务 ID、文件哈希、音频格式和时长。完整参数和故障恢复见
[`kaggle/README.md`](kaggle/README.md)。

## 本地推理

准备 `ckpts/rift25k.ckpt`，并下载辅助模型：

```bash
uv run python scripts/download_inference_assets.py --modules-only --output-dir .
```

执行基准推理：

```bash
uv run python infer.py \
  --model ckpts/rift25k.ckpt \
  --assets-dir pretrained \
  --input vocals.wav \
  --output outputs/vocals__rift25k-steps32-rf0-seed7.wav \
  --speaker target \
  --device cuda \
  --infer-steps 32 \
  --ds-cfg-strength 0.2 \
  --spk-cfg-strength 0.8 \
  --cfg-rescale 0.7 \
  --robust-f0 0 \
  --output-subtype FLOAT \
  --seed 7
```

## 制作与维护

歌曲制作从 [`docs/workflow/README.md`](docs/workflow/README.md) 开始；歌曲专属
素材、时间点、试听结论和版本记录保存在歌曲目录。

修改代码或更新依赖后运行：

```bash
uv run pytest
ruff check .
nix flake check --no-build
```

涉及模型加载、音频拼接或 Kaggle Kernel 的变更，还应使用短音频重新跑一次真实
T4 端到端任务。
