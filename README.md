# RIFT-SVC 推理工作区

这是露早 RIFT-SVC 的私有推理仓库，只维护三件事：

- 本地 RIFT 推理代码与测试；
- Kaggle GPU 自动推理；
- 歌曲制作、修复、混音和视频封装记录。

训练代码和通用 GUI 不在维护范围内。

## 本地环境

Nix Flake 提供 Python 3.11、编译环境、FFmpeg、Kaggle CLI、uv 和 jq；Python
依赖由 uv 管理。首次进入仓库：

```bash
direnv allow
uv sync --extra dev
```

以后进入目录时，direnv 会加载 Flake；如果 `.venv` 已存在，也会自动激活。
也可以手动使用：

```bash
nix develop
uv sync --extra dev
```

依赖允许持续追新。更新后至少运行：

```bash
uv lock --upgrade
uv run pytest
ruff check .
```

涉及 Torch、音频处理或推理逻辑时，还应在 Kaggle 跑一个短音频任务。

## Kaggle 自动推理

日常使用不需要打开 Kaggle 网页。登录 Kaggle CLI 后运行：

```bash
uv run python scripts/kaggle_rift.py vocals.wav \
  --output-dir /path/to/results
```

默认参数是 `steps=32`、`ds=0.2`、`spk=0.8`、`cfg-rescale=0.7`、
`robust-f0=0`、`seed=7`，输出为 44.1 kHz 单声道 Float WAV。脚本会：

1. 更新私有输入 Dataset；
2. 提交私有 GPU Script Kernel；
3. 轮询运行状态，失败时显示日志；
4. 下载并校验 WAV 和运行清单；
5. 保存到 `<output-dir>/<job-id>/`。

参数与首次配置见 [`kaggle/README.md`](kaggle/README.md)。

常用分离模型也走同一套 Kaggle CLI 工作流，例如：

```bash
uv run python scripts/kaggle_separate.py vocals.wav \
  --model anvuew-dereverb-22.5050 \
  --output-dir /path/to/results
```

## 本地推理

下载辅助模型：

```bash
uv run python scripts/download_inference_assets.py --modules-only --output-dir .
```

执行推理：

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

## 制作流程

通用制作流程见 [`docs/workflow/README.md`](docs/workflow/README.md)，每首歌的具体
素材、试听结论和版本记录保存在歌曲目录，不塞回代码仓库的主说明。
