# Kaggle CLI 推理

这套流程用于从本地提交一条音频到 Kaggle GPU，并在完成后自动下载结果，不需要
打开 Notebook 页面。

## 固定资源

默认使用以下私有 Kaggle 资源：

- 输入 Dataset：`eeviriyi/rift-svc-cli-input`
- 推理 Kernel：`eeviriyi/rift-svc-cli-inference`
- 模型：`ooaaqq/rift-svc-luzao-25k/rift25k.ckpt`

输入 Dataset 是临时任务通道。每次提交只保留当前音频和 `job.json`，默认删除它的
旧版本；本地原文件和已经下载的结果不受影响。

## 首次检查

进入 Flake 环境并确认 Kaggle 已登录：

```bash
kaggle kernels list --mine --page-size 5
```

模型仓库公开时不需要 `HF_TOKEN`。若以后改为私有仓库，再在 Kaggle Secret 中添加
只读 `HF_TOKEN`。

Kernel 默认拉取 GitHub `master`。修改 RIFT 核心推理代码后，应先提交并推送；只修改
本地控制脚本或 Kernel 启动代码时不受这个限制。

先检查任务内容，不上传：

```bash
uv run python scripts/kaggle_rift.py vocals.wav --dry-run
```

## 提交并下载

```bash
uv run python scripts/kaggle_rift.py vocals.wav \
  --output-dir /home/elvedon/Music/露早/歌曲/RIFT
```

常用参数：

```bash
uv run python scripts/kaggle_rift.py vocals.wav \
  --output-dir ./results \
  --steps 64 \
  --robust-f0 1 \
  --spk 0.8 \
  --seed 7
```

输出目录包含动态命名的 Float WAV 和 `manifest.json`。清单记录输入输出 hash、
推理参数、代码 commit、模型信息以及云端 Python、Torch、CUDA、GPU 和依赖版本。

## 运行结构

```text
本地音频
  -> 私有 Kaggle Dataset 最新版本
  -> 私有 GPU Script Kernel
  -> /kaggle/working/rift-output
  -> Kaggle CLI 下载
  -> 本地 <output-dir>/<job-id>
```

仓库、checkpoint、ContentVec、RMVPE 和 HiFi-GAN 都放在 `/kaggle/temp`，不会作为
Kernel 输出下载。`/kaggle/working` 只保留最终 WAV 和 manifest。

两类 Script Kernel 都固定使用网页中的 `GPU T4 ×2`（API 名称
`NvidiaTeslaT4`），并沿用 Kaggle 预装的 CUDA PyTorch。当前推理只使用其中一张
T4，不为 P100 降级或重装 PyTorch。

## 排错

脚本会在 Kernel 失败或超时后自动打印日志。也可以手动检查：

```bash
kaggle kernels status eeviriyi/rift-svc-cli-inference
kaggle kernels logs eeviriyi/rift-svc-cli-inference
```

中断本地脚本不会取消已经提交的 Kaggle 任务。任务仍可继续运行，完成后可手动取回：

```bash
kaggle kernels output eeviriyi/rift-svc-cli-inference \
  -p ./kaggle-output -o
```

## 辅助分离 Notebook

分离任务也可以完全通过 CLI 提交：

```bash
uv run python scripts/kaggle_separate.py input.wav \
  --model anvuew-karaoke \
  --output-dir ./results
```

支持的模型：

- `anvuew-dereverb-22.5050`：Anvuew BS-RoFormer Dereverb 22.5050；
- `anvuew-karaoke`：Anvuew Karaoke BS-RoFormer；
- `becruily-frazer-karaoke`：Becruily & Frazer Karaoke BS-RoFormer；
- `small-karaoke-gaboxaufr`：GaboxR67 Small Karaoke MelBand-RoFormer。

每次输出两个 Float WAV stem 和 `manifest.json`，并固定 MSST commit、模型仓库
revision、权重文件和配置文件。对应的固定私有资源是：

- 输入 Dataset：`eeviriyi/rift-separation-cli-input`
- Script Kernel：`eeviriyi/rift-separation-cli`

现有 `.ipynb` 暂时保留作已验证行为参考。等新 Script Kernel 至少实际跑通 Anvuew
Dereverb、Anvuew Karaoke 和一个 MelBand profile 后，再统一删除旧 Notebook。
