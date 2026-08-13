from __future__ import annotations

import torch
from torch import nn

from rift_svc.core_utils import lens_to_mask


def _combine_cfg_predictions(
    predictions: torch.Tensor,
    *,
    use_ds_cfg: bool,
    use_spk_cfg: bool,
    ds_cfg_strength: float,
    spk_cfg_strength: float,
    cfg_rescale: float,
) -> torch.Tensor:
    standard = predictions[:, 0]
    guided = standard
    condition_index = 1

    if use_ds_cfg:
        guided = guided + (standard - predictions[:, condition_index]) * ds_cfg_strength
        condition_index += 1
    if use_spk_cfg:
        guided = (
            guided + (standard - predictions[:, condition_index]) * spk_cfg_strength
        )

    if cfg_rescale > 1e-5:
        reduce_dims = tuple(range(1, guided.ndim))
        standard_std = standard.std(dim=reduce_dims, keepdim=True)
        guided_std = guided.std(dim=reduce_dims, keepdim=True)
        scale = torch.where(
            guided_std > 1e-5,
            standard_std / guided_std,
            torch.ones_like(guided_std),
        )
        rescaled = guided * scale
        guided = cfg_rescale * rescaled + (1 - cfg_rescale) * guided

    return guided


class RF(nn.Module):
    """Inference-time Euler sampler for a trained RIFT transformer."""

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer
        self.mel_min = -12
        self.mel_max = 2

    @torch.no_grad()
    def sample(
        self,
        src_mel: torch.Tensor,
        spk_id: torch.Tensor,
        f0: torch.Tensor,
        rms: torch.Tensor,
        cvec: torch.Tensor,
        *,
        frame_len: torch.Tensor | None = None,
        steps: int = 32,
        bad_cvec: torch.Tensor | None = None,
        ds_cfg_strength: float = 0.0,
        spk_cfg_strength: float = 0.0,
        cfg_rescale: float = 0.7,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Sample a mel spectrogram using the repository's Euler path."""
        if steps < 2:
            raise ValueError("steps must be at least 2")

        batch, sequence_length, mel_channels = src_mel.shape
        device = src_mel.device
        if frame_len is None:
            frame_len = torch.full(
                (batch,), sequence_length, dtype=torch.long, device=device
            )
        mask = lens_to_mask(frame_len, length=sequence_length)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        sampled = torch.randn(
            batch,
            sequence_length,
            mel_channels,
            device=device,
            dtype=src_mel.dtype,
            generator=generator,
        )
        sampled = sampled.masked_fill(~mask.unsqueeze(-1), 0)

        use_ds_cfg = ds_cfg_strength > 1e-5
        use_spk_cfg = spk_cfg_strength > 1e-5
        if use_ds_cfg and bad_cvec is None:
            raise ValueError("bad_cvec is required when ds_cfg_strength is positive")

        condition_count = 1 + int(use_ds_cfg) + int(use_spk_cfg)
        if condition_count > 1:
            speaker_cond = spk_id.repeat_interleave(condition_count, dim=0)
            f0_cond = f0.repeat_interleave(condition_count, dim=0)
            rms_cond = rms.repeat_interleave(condition_count, dim=0)
            mask_cond = mask.repeat_interleave(condition_count, dim=0)
            cvec_cond = torch.stack(
                [cvec]
                + ([bad_cvec] if use_ds_cfg else [])
                + ([cvec] if use_spk_cfg else []),
                dim=1,
            ).reshape(-1, sequence_length, cvec.shape[-1])
            drop_speaker_cond = torch.zeros(
                (batch, condition_count), dtype=torch.bool, device=device
            )
            if use_spk_cfg:
                drop_speaker_cond[:, -1] = True
            drop_speaker_cond = drop_speaker_cond.reshape(-1)

        def predict(time: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
            if condition_count == 1:
                return self.transformer(
                    x=current,
                    spk=spk_id,
                    f0=f0,
                    rms=rms,
                    cvec=cvec,
                    time=time,
                    mask=mask,
                )

            current = current.repeat_interleave(condition_count, dim=0)
            predictions = self.transformer(
                x=current,
                spk=speaker_cond,
                f0=f0_cond,
                rms=rms_cond,
                cvec=cvec_cond,
                time=time,
                mask=mask_cond,
                drop_speaker=drop_speaker_cond,
            )
            predictions = predictions.reshape(
                batch, condition_count, sequence_length, mel_channels
            )
            return _combine_cfg_predictions(
                predictions,
                use_ds_cfg=use_ds_cfg,
                use_spk_cfg=use_spk_cfg,
                ds_cfg_strength=ds_cfg_strength,
                spk_cfg_strength=spk_cfg_strength,
                cfg_rescale=cfg_rescale,
            )

        time = torch.linspace(
            0,
            1,
            steps,
            device=device,
        )
        for index in range(steps - 1):
            sampled = sampled + (time[index + 1] - time[index]) * predict(
                time[index], sampled
            )

        sampled = self.denorm_mel(sampled)
        return torch.where(mask.unsqueeze(-1), sampled, src_mel)

    def denorm_mel(self, mel: torch.Tensor) -> torch.Tensor:
        return (mel + 1) / 2 * (self.mel_max - self.mel_min) + self.mel_min
