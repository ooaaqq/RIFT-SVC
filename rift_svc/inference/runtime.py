"""The single-file RIFT-SVC inference runtime."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import torch
from torch.amp import autocast
from tqdm import tqdm

from rift_svc.core_utils import linear_interpolate_tensor
from rift_svc.dit import DiT
from rift_svc.feature_extractors import (
    HubertModelWithFinalProj,
    RMSExtractor,
    get_mel_spectrogram,
)
from rift_svc.inference.audio import add_segment, load_audio
from rift_svc.inference.pitch import (
    f0_ensemble,
    f0_ensemble_light,
    get_f0_pm,
    get_f0_pw,
    post_process_f0,
)
from rift_svc.inference.slicer import Slicer
from rift_svc.nsf_hifigan import NsfHifiGAN
from rift_svc.rf import RF
from rift_svc.rmvpe import RMVPE


def extract_state_dict(ckpt):
    """Extract model weights and inference metadata from a RIFT-SVC checkpoint."""
    state_dict = {
        key.removeprefix("model."): value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("model.")
    }
    if not state_dict:
        raise ValueError("checkpoint does not contain a model state_dict")

    config = ckpt["hyper_parameters"]["cfg"]
    return (
        state_dict,
        config["spk2idx"],
        config["model"],
        config["dataset"],
    )


def load_models(model_path, device, use_fp16=True, assets_dir=None):
    """Load the fine-tuned model and the three auxiliary inference models."""
    device = torch.device(device)
    assets_dir = (
        Path(assets_dir)
        if assets_dir is not None
        else Path(__file__).resolve().parents[2] / "pretrained"
    )
    asset_paths = {
        "vocoder": assets_dir
        / "nsf_hifigan_44.1k_hop512_128bin_2024.02"
        / "model.ckpt",
        "rmvpe": assets_dir / "rmvpe" / "model.pt",
        "content_vec": assets_dir / "content-vec-best",
    }
    missing_assets = [
        str(path) for path in asset_paths.values() if not path.exists()
    ]
    if missing_assets:
        raise FileNotFoundError(
            "Missing inference assets. Run scripts/download_inference_assets.py or "
            f"provide --assets-dir. Missing: {', '.join(missing_assets)}"
        )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict, spk2idx, model_cfg, dataset_cfg = extract_state_dict(checkpoint)

    transformer = DiT(num_speaker=len(spk2idx), **model_cfg)
    svc_model = RF(transformer)
    svc_model.load_state_dict(state_dict)
    svc_model = svc_model.to(device).eval()

    use_fp16 = bool(use_fp16 and device.type == "cuda")
    if use_fp16:
        svc_model.half()

    vocoder = NsfHifiGAN(str(asset_paths["vocoder"]), device=device).to(device)
    rmvpe = RMVPE(
        model_path=str(asset_paths["rmvpe"]),
        hop_length=160,
        device=device,
    )
    hubert = HubertModelWithFinalProj.from_pretrained(
        str(asset_paths["content_vec"])
    ).to(device)
    rms_extractor = RMSExtractor().to(device)

    if use_fp16:
        vocoder.half()
        hubert.half()
        rms_extractor.half()

    return svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg


def extract_features(
    audio_segment,
    sample_rate,
    hop_length,
    rmvpe,
    hubert,
    rms_extractor,
    device,
    key_shift=0,
    ds_cfg_strength=0.0,
    cvec_downsample_rate=2,
    target_loudness=-18.0,
    robust_f0=0,
    use_fp16=True,
):
    """Extract the mel, ContentVec, F0 and RMS inputs for one segment."""
    meter = pyln.Meter(sample_rate, block_size=0.1)
    original_loudness = meter.integrated_loudness(audio_segment)
    normalized_audio = audio_segment
    if np.isfinite(original_loudness):
        normalized_audio = pyln.normalize.loudness(
            audio_segment, original_loudness, target_loudness
        )

    max_amp = np.max(np.abs(normalized_audio), initial=0.0)
    if max_amp > 1.0:
        normalized_audio = normalized_audio * (0.99 / max_amp)

    audio_tensor = (
        torch.from_numpy(np.asarray(normalized_audio, dtype=np.float32))
        .unsqueeze(0)
        .to(device)
    )
    audio_16khz = (
        torch.from_numpy(
            librosa.resample(
                np.asarray(normalized_audio, dtype=np.float32),
                orig_sr=sample_rate,
                target_sr=16000,
            )
        )
        .float()
        .unsqueeze(0)
        .to(device)
    )

    if use_fp16:
        audio_tensor = audio_tensor.half()
        audio_16khz = audio_16khz.half()

    mel = get_mel_spectrogram(
        audio_tensor,
        sampling_rate=sample_rate,
        n_fft=2048,
        num_mels=128,
        hop_size=hop_length,
        win_size=2048,
        fmin=40,
        fmax=16000,
    ).transpose(1, 2)

    device_type = "cuda" if device.type == "cuda" else "cpu"
    with autocast(device_type=device_type, enabled=use_fp16):
        cvec = hubert(audio_16khz)["last_hidden_state"].squeeze(0)
    cvec = linear_interpolate_tensor(cvec, mel.shape[1])[None, :]

    cvec_ds = None
    if ds_cfg_strength > 0:
        cvec_ds = cvec[:, ::2, :]
        cvec_ds = linear_interpolate_tensor(
            cvec_ds[0], max(1, cvec_ds.shape[1] // cvec_downsample_rate)
        )
        cvec_ds = linear_interpolate_tensor(cvec_ds, mel.shape[1])[None, :]

    if robust_f0 > 0:
        time_step = hop_length / sample_rate
        f0_min, f0_max = 40, 1100
        with autocast(device_type=device_type, enabled=use_fp16):
            rmvpe_f0 = rmvpe.infer_from_audio(
                audio_tensor, sample_rate=sample_rate, device=device
            )
        rmvpe_f0 = post_process_f0(
            rmvpe_f0,
            sample_rate,
            hop_length,
            mel.shape[1],
            silence_front=0.0,
            cut_last=False,
        )
        pw_f0 = get_f0_pw(
            normalized_audio, sample_rate, time_step, f0_min, f0_max
        )
        pmac_f0 = get_f0_pm(
            normalized_audio, sample_rate, time_step, f0_min, f0_max
        )
        if robust_f0 == 1:
            with autocast(device_type=device_type, enabled=use_fp16):
                rms_np = rms_extractor(audio_tensor).squeeze().cpu().numpy()
            f0 = f0_ensemble_light(rmvpe_f0, pw_f0, pmac_f0, rms=rms_np)
        else:
            f0 = f0_ensemble(rmvpe_f0, pw_f0, pmac_f0)
    else:
        with autocast(device_type=device_type, enabled=use_fp16):
            f0 = rmvpe.infer_from_audio(
                audio_tensor, sample_rate=sample_rate, device=device
            )
        f0 = post_process_f0(
            f0,
            sample_rate,
            hop_length,
            mel.shape[1],
            silence_front=0.0,
            cut_last=False,
        )

    if key_shift:
        f0 = f0 * 2 ** (key_shift / 12)
    f0 = torch.from_numpy(f0).float().to(device)[None, :]
    rms = rms_extractor(audio_tensor)
    return mel, cvec, cvec_ds, f0, rms, original_loudness


def run_inference(
    model,
    mel,
    cvec,
    f0,
    rms,
    cvec_ds,
    spk_id,
    infer_steps,
    ds_cfg_strength,
    spk_cfg_strength,
    cfg_rescale,
    frame_length,
    use_fp16,
    seed,
):
    device_type = "cuda" if mel.device.type == "cuda" else "cpu"
    with autocast(device_type=device_type, enabled=use_fp16):
        return model.sample(
            src_mel=mel,
            spk_id=spk_id,
            f0=f0,
            rms=rms,
            cvec=cvec,
            steps=infer_steps,
            bad_cvec=cvec_ds,
            ds_cfg_strength=ds_cfg_strength,
            spk_cfg_strength=spk_cfg_strength,
            cfg_rescale=cfg_rescale,
            frame_len=frame_length,
            seed=seed,
        )


def generate_audio(
    vocoder,
    mel_out,
    f0,
    original_loudness=None,
    restore_loudness=True,
    use_fp16=True,
    expected_length=None,
    sample_rate=44100,
):
    """Vocode one segment and restore its original integrated loudness."""
    device_type = "cuda" if mel_out.device.type == "cuda" else "cpu"
    with autocast(device_type=device_type, enabled=use_fp16):
        audio_out = vocoder(mel_out.transpose(1, 2), f0)
    audio_out = audio_out.squeeze().float().cpu().numpy()

    if expected_length is not None:
        audio_out = audio_out[:expected_length]
        if len(audio_out) < expected_length:
            audio_out = np.pad(
                audio_out, (0, expected_length - len(audio_out)), mode="constant"
            )

    if restore_loudness and original_loudness is not None and np.isfinite(
        original_loudness
    ):
        meter = pyln.Meter(sample_rate, block_size=0.1)
        output_loudness = meter.integrated_loudness(audio_out)
        if np.isfinite(output_loudness):
            audio_out = pyln.normalize.loudness(
                audio_out, output_loudness, original_loudness
            )
            max_amp = np.max(np.abs(audio_out), initial=0.0)
            if max_amp > 1.0:
                audio_out = audio_out * (0.99 / max_amp)

    return np.nan_to_num(audio_out).astype(np.float32, copy=False)


def process_segment(
    audio_segment,
    *,
    svc_model,
    vocoder,
    rmvpe,
    hubert,
    rms_extractor,
    speaker_id,
    sample_rate,
    hop_length,
    device,
    key_shift=0,
    infer_steps=32,
    ds_cfg_strength=0.0,
    spk_cfg_strength=0.0,
    cfg_rescale=0.7,
    cvec_downsample_rate=2,
    target_loudness=-18.0,
    restore_loudness=True,
    robust_f0=0,
    use_fp16=True,
    seed=None,
):
    mel, cvec, cvec_ds, f0, rms, original_loudness = extract_features(
        audio_segment,
        sample_rate,
        hop_length,
        rmvpe,
        hubert,
        rms_extractor,
        device,
        key_shift,
        ds_cfg_strength,
        cvec_downsample_rate,
        target_loudness,
        robust_f0,
        use_fp16,
    )
    speaker = torch.tensor([speaker_id], dtype=torch.long, device=device)
    frame_length = torch.tensor([mel.shape[1]], device=device)
    mel_out = run_inference(
        svc_model,
        mel,
        cvec,
        f0,
        rms,
        cvec_ds,
        speaker,
        infer_steps,
        ds_cfg_strength,
        spk_cfg_strength,
        cfg_rescale,
        frame_length,
        use_fp16,
        seed,
    )
    return generate_audio(
        vocoder,
        mel_out,
        f0,
        original_loudness,
        restore_loudness,
        use_fp16,
        expected_length=len(audio_segment),
        sample_rate=sample_rate,
    )


class InferenceRuntime:
    """Load the models once and convert one audio file at a time."""

    def __init__(
        self,
        model_path,
        *,
        device=None,
        use_fp16=True,
        assets_dir=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        (
            self.svc_model,
            self.vocoder,
            self.rmvpe,
            self.hubert,
            self.rms_extractor,
            self.spk2idx,
            dataset_cfg,
        ) = load_models(
            model_path,
            self.device,
            self.use_fp16,
            assets_dir,
        )
        self.sample_rate = int(dataset_cfg.get("sample_rate", 44100))
        self.hop_length = 512

    def convert(
        self,
        input_path,
        *,
        speaker,
        key_shift=0,
        infer_steps=32,
        ds_cfg_strength=0.2,
        spk_cfg_strength=0.8,
        cfg_rescale=0.7,
        cvec_downsample_rate=2,
        target_loudness=-18.0,
        restore_loudness=True,
        fade_duration_ms=20.0,
        robust_f0=0,
        slicer_threshold=-30.0,
        slicer_min_length=3000,
        slicer_min_interval=100,
        slicer_hop_size=10,
        slicer_max_sil_kept=200,
        seed=None,
    ):
        if speaker not in self.spk2idx:
            valid = ", ".join(self.spk2idx)
            raise ValueError(f"unknown speaker {speaker!r}; valid speakers: {valid}")
        if infer_steps < 2:
            raise ValueError("infer_steps must be at least 2")
        if robust_f0 not in (0, 1, 2):
            raise ValueError("robust_f0 must be 0, 1, or 2")
        if fade_duration_ms < 0:
            raise ValueError("fade_duration_ms must not be negative")

        audio = load_audio(input_path, self.sample_rate)
        slicer = Slicer(
            sr=self.sample_rate,
            threshold=slicer_threshold,
            min_length=slicer_min_length,
            min_interval=slicer_min_interval,
            hop_size=slicer_hop_size,
            max_sil_kept=slicer_max_sil_kept,
        )
        segments = slicer.slice(audio)
        if not segments:
            raise ValueError("no non-empty audio segments were found")

        fade_samples = round(fade_duration_ms * self.sample_rate / 1000)
        result = np.zeros(
            len(audio) + max(0, fade_samples),
            dtype=np.float32,
        )
        with torch.inference_mode():
            for index, (start_sample, chunk) in enumerate(
                tqdm(segments, desc="Processing segments")
            ):
                converted = process_segment(
                    chunk,
                    svc_model=self.svc_model,
                    vocoder=self.vocoder,
                    rmvpe=self.rmvpe,
                    hubert=self.hubert,
                    rms_extractor=self.rms_extractor,
                    speaker_id=self.spk2idx[speaker],
                    sample_rate=self.sample_rate,
                    hop_length=self.hop_length,
                    device=self.device,
                    key_shift=key_shift,
                    infer_steps=infer_steps,
                    ds_cfg_strength=ds_cfg_strength,
                    spk_cfg_strength=spk_cfg_strength,
                    cfg_rescale=cfg_rescale,
                    cvec_downsample_rate=cvec_downsample_rate,
                    target_loudness=target_loudness,
                    restore_loudness=restore_loudness,
                    robust_f0=robust_f0,
                    use_fp16=self.use_fp16,
                    seed=None if seed is None else seed + index,
                )
                add_segment(
                    result,
                    index=index,
                    total_segments=len(segments),
                    start_sample=start_sample,
                    audio_out=converted,
                    expected_length=len(chunk),
                    fade_samples=fade_samples,
                )
        return result[: len(audio)]
