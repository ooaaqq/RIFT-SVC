# RIFT 推理

## 基准参数

先用固定参数跑完整歌曲：

```text
checkpoint          rift25k
speaker             target
key shift           0
infer steps         32
ds cfg strength     0.2
speaker cfg         0.8
cfg rescale         0.7
robust-f0           0
seed                7
output              44.1 kHz Float WAV
```

示例：

```bash
python infer.py \
  --model ckpts/rift25k.ckpt \
  --input vocals.wav \
  --output outputs/vocals__rift25k-k0-steps32-ds0.2-spk0.8-cfg0.7-rf0-seed7.wav \
  --speaker target \
  --key-shift 0 \
  --device cuda \
  --infer-steps 32 \
  --ds-cfg-strength 0.2 \
  --spk-cfg-strength 0.8 \
  --cfg-rescale 0.7 \
  --robust-f0 0 \
  --output-subtype FLOAT \
  --seed 7
```

## 局部参数矩阵

只有基准版出现明确问题时，再截取带上下文的局部测试：

```text
steps64 + robust-f0 1
steps64 + robust-f0 2
同参数更换 seed
```

重点比较 F0 抖动、低音丢失、转音电音、咬字和音色一致性。高 steps 或
robust-f0 并不天然更好；若修正音高但损伤咬字，应保留基准版并局部处理。

## 输出与复现

- 中间推理输出使用 Float WAV，避免重复量化。
- 文件名包含 checkpoint、key、steps、CFG、robust-f0 和 seed。
- 保存实际命令或 Notebook 参数。
- 全曲基准与局部候选分开存放。
- 不因局部问题重新替换整首人声。
