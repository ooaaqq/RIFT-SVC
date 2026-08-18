# Kaggle CLI inference

本地脚本负责上传、提交、等待、下载和校验，不需要打开 Kaggle 网页。计算单位是：

```text
一个模型 + 一组参数 + 多条输入 = 一个 Batch Run
```

## RIFT dual-T4 batch

默认资源：

- Dataset：`eeviriyi/rift-svc-cli-input`
- Kernel：`eeviriyi/rift-svc-cli-inference`
- 模型：`ooaaqq/rift-svc-luzao-25k/rift25k.ckpt`

一次提交多条准备好的人声：

```bash
uv run python scripts/kaggle_rift.py \
  '/path/歌名A-歌手A-bs124-vocal-l1.wav' \
  '/path/歌名B-歌手B-bs124-vocal-l1.wav' \
  '/path/歌名C-歌手C-bs124-vocal-l1-anvuew-karaoke-lead.wav' \
  --output-dir '/path/to/song/20. AI'
```

先生成批次计划而不上传：

```bash
uv run python scripts/kaggle_rift.py vocal-a.wav vocal-b.wav --dry-run
```

Kaggle 的两张 T4 各启动一个常驻 worker，分别绑定 `cuda:0`、`cuda:1`。两个 worker
从动态队列领取文件，每张卡只加载一次 RIFT、RMVPE、ContentVec 和 vocoder。只有一条
输入时只启动一个 worker；不会为了占用第二张卡切割单首音频。

一个批次的全部输出会先下载到隐藏临时目录，校验并复制完成后再整体发布为参数目录：

```text
RIFT25K-k0-s32-ds0p2-spk0p8-cfg0p7-rf0-seed7/
```

输入文件名中的处理链保持不变，只追加 `rift25k`。例如：

```text
歌名-歌手-bs124-vocal-l1-rift25k.wav
歌名-歌手-bs124-vocal-l1-anvuew-karaoke-lead-rift25k.wav
```

云端 manifest 用于验证批次 ID、输出数量、SHA-256、采样帧、采样率和声道；验证成功
后本地工作区只保留普通 Float WAV。输出相对输入重采样后的长度默认最多只容许 2 个
采样点误差。参数目录已存在时整批拒绝，因此同一 Run 的输入应一次批量提交。

## Public separation models

公开去和声和去混响模型通过独立 Kernel 提交：

```bash
uv run python scripts/kaggle_separate.py input.wav \
  --model anvuew-karaoke \
  --output-dir ./results
```

当前 profiles：

- `anvuew-dereverb-22.5050`
- `anvuew-karaoke`
- `becruily-frazer-karaoke`
- `small-karaoke-gaboxaufr`

它们是歌曲按需选择的独立模型，不属于 MVSEP 的固定后处理阶段。

## Resource lifecycle

输入 Dataset 是当前任务的临时通道，默认删除旧版本；本地原文件与已下载结果不受
影响。默认 Git、checkpoint 仓库和 ContentVec/RMVPE/vocoder modules 都固定为确切
revision；修改核心推理代码或模型资产后要先提交并推送，再显式更新固定 revision。
只修改本地提交器或上传的 Kernel 文件不受此限制。

RIFT 和公开分离各自共享一组 Dataset/Kernel。本机启动第二个占用相同远端通道的任务
会立即拒绝；锁覆盖上传、运行、下载和原子发布全过程。不同机器仍不能同时操作同一组
Dataset/Kernel。

## Troubleshooting

```bash
kaggle kernels status eeviriyi/rift-svc-cli-inference
kaggle kernels logs eeviriyi/rift-svc-cli-inference
kaggle kernels output eeviriyi/rift-svc-cli-inference \
  -p ./kaggle-output -o
```

中断本地脚本不会取消已经提交的 Kaggle Batch Run。
