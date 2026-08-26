# RIFT-SVC 推理工作区

这是露早 1024-16、25K checkpoint 的私有推理仓库。只维护：

- RIFT-SVC 本地推理核心；
- Kaggle T4 自动推理与分离；
- 好友内部使用的极简 Web 单步任务队列；
- 从音源、局部修复、混音到视频封装的制作工作流。

不维护训练、数据预处理或通用多模型兼容层。模型公式、RMVPE、Mel/STFT
和 HiFi-GAN 结构保持与现有权重一致。

## 仓库结构

```text
infer.py                    本地单文件推理入口
rift_svc/                   RIFT、RMVPE、HiFi-GAN 与推理运行时
rift_svc/audio_tools.py     制作脚本共享的音频 I/O、时间与原子写入
rift_svc/alignment.py       对齐检测、稳健拟合与保音高渲染核心
rift_svc/mixctl.py          歌曲书架、原子版本与交付检查工具
rift_web/                   匿名口令、共享队列、上传下载与串行 dispatcher
scripts/kaggle_rift.py      本地提交 RIFT 云端任务
scripts/kaggle_separate.py  本地提交 BS/MelBand-RoFormer 任务
kaggle/rift/kernel.py       Kaggle RIFT Script Kernel
kaggle/separation/kernel.py Kaggle 分离 Script Kernel
scripts/render_m7_returns.py 通用 M7 true-stereo IR 返回渲染
scripts/apply_breath_control.py 局部漏气高频动态压制
scripts/align_audio.py       对齐锚点检查与保音高分段校正
scripts/apply_gain_automation.py 人写时间区间的平滑音量自动化
scripts/analyze_reference_energy.py 只读原曲慢速能量分析
docs/workflow/README.md     唯一的歌曲制作工作流
tests/                      推理行为和云端提交器的轻量测试
```

模型 checkpoint、辅助模型、IR、输入音频和成品输出都不进入 Git。M7 等共享制作
素材保存在 `/home/elvedon/Music/露早/90. Shared Resources/`。

## 环境

首次进入仓库：

```bash
direnv allow
uv sync --extra dev --extra web
```

之后进入目录时，direnv 会自动加载 Nix Flake，并激活已有的 `.venv`。也可以手动：

```bash
nix develop
uv sync --extra dev --extra web
```

Flake 提供 Python 3.11、FFmpeg、Kaggle CLI、uv、Ruff 和编译依赖；uv 只管理
Python 包。

本地与 Kaggle RIFT 默认都使用 `seed=7`，中间输出均为 44.1 kHz mono Float WAV；
PCM24 只在后续交付量化时使用。

## 日常使用

RIFT 云端批量推理：

```bash
uv run python scripts/kaggle_rift.py vocal-a.wav vocal-b.wav vocal-c.wav \
  --output-dir /path/to/results
```

辅助分离：

```bash
uv run python scripts/kaggle_separate.py vocals.wav \
  --model anvuew-dereverb-22.5050 \
  --output-dir /path/to/results
```

RIFT 批次会让双 T4 各加载一套常驻推理运行时并动态领取输入；辅助分离仍按所选公开
模型提交。两者都会校验任务 ID、文件哈希、音频格式和时长。完整参数和故障恢复见
[`kaggle/README.md`](kaggle/README.md)。

## 好友任务队列

Web 队列一次只接受一个操作：提取干声、去和声、去混响或 RIFT。提交时选择的模型
会随任务保存，并由 dispatcher 映射到对应的固定 Kaggle profile。任务名、用户名、
类型和状态公开可见；输入、输出和 manifest 只有提交者与管理员可访问。后续处理必须
人工下载、试听、归档并重新上传，不自动串联步骤。

Nix package 提供三个入口：

```text
rift-web          FastAPI 与极简前端
rift-dispatcher   SQLite FIFO 到 Kaggle 的单通道执行器
rift-cleanup      删除到期音频但保留任务记录和 manifest
```

生产环境由 NixOS fleet 提供 `RIFT_WEB_USERS_FILE`、状态目录和 Kaggle credential；
凭据、任务输入和输出不进入 Git 或 Nix store。

本地修改前端或 API 时可直接运行开发预览：

```bash
./scripts/run-local-web.sh
```

浏览 `http://127.0.0.1:8767`，使用 `local-admin` 或 `local-friend`。本地预览只运行
Web API，不启动 Kaggle dispatcher；状态写入被 Git 忽略的 `var/rift-web-local/`。

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

歌曲制作只维护一份 [`docs/workflow/README.md`](docs/workflow/README.md)。歌曲专属
处理通过版本名、脚本和同目录输出表达，不要求维护额外的素材、问题或版本文档。
工作区入口为：

```bash
uv run python -m rift_svc.mixctl --help
```

修改代码或更新依赖后运行：

```bash
uv run pytest
ruff check .
nix flake check --no-build
```

涉及模型加载、音频拼接或 Kaggle Kernel 的变更，还应使用短音频重新跑一次真实
T4 端到端任务。
