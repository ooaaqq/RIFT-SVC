# 母带与视频封装

## 音频交付

保留：

```text
44.1 kHz / 24-bit WAV   后续制作与归档
44.1 kHz / 24-bit FLAC  无损试听与发布母版
```

交付前验证：

- 编码、采样率、位深、声道和时长。
- integrated LUFS、LRA 和峰值。
- 局部修复区间与首尾解码。
- 最终文件 SHA-256。

## Bilibili Hi-Res 封装

当前验证可用的封装规格：

```text
视频             原 4K H.264 直接复制
音频             96 kHz / 24-bit ALAC
容器             MP4
音频语言         zho
faststart         启用
```

示例：

```bash
ffmpeg -i input-video.mp4 -i final-44k24.wav \
  -map 0:v:0 -map 1:a:0 \
  -c:v copy \
  -af 'aresample=96000:resampler=soxr:precision=28,aformat=sample_fmts=s32p:channel_layouts=stereo' \
  -c:a alac \
  -metadata:s:a:0 language=zho \
  -metadata:s:a:0 handler_name=SoundHandler \
  -movflags +faststart -shortest \
  output__Bilibili-HiRes-96k24-ALAC.mp4
```

44.1 kHz 升采样到 96 kHz不会产生新的真实高频信息，只用于匹配平台封装规格。
平台是否显示 Hi-Res、转码后是否出现刺耳失真，必须以上传后的实际播放结果
判断。
