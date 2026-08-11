"""Reusable RIFT-SVC inference runtime."""

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
from rift_svc.inference.audio import add_segment, assemble_segments, load_audio
from rift_svc.inference.pitch import (
    f0_ensemble,
    f0_ensemble_light,
    get_f0_pm,
    get_f0_pw,
    post_process_f0,
)
from rift_svc.nsf_hifigan import NsfHifiGAN
from rift_svc.rf import RF
from rift_svc.rmvpe import RMVPE
from slicer import Slicer


def extract_state_dict(ckpt):
    state_dict = ckpt['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_k = k.replace('model.', '')
            new_state_dict[new_k] = v
    spk2idx = ckpt['hyper_parameters']['cfg']['spk2idx']
    model_cfg = ckpt['hyper_parameters']['cfg']['model']
    dataset_cfg = ckpt['hyper_parameters']['cfg']['dataset']
    return new_state_dict, spk2idx, model_cfg, dataset_cfg


def load_models(model_path, device, use_fp16=True, assets_dir=None):
    """Load all required models and return them"""
    device = torch.device(device)
    assets_dir = (
        Path(assets_dir)
        if assets_dir
        else Path(__file__).resolve().parents[2] / "pretrained"
    )
    asset_paths = {
        "vocoder": assets_dir / "nsf_hifigan_44.1k_hop512_128bin_2024.02" / "model.ckpt",
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

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict, spk2idx, dit_cfg, dataset_cfg = extract_state_dict(ckpt)

    transformer = DiT(num_speaker=len(spk2idx), **dit_cfg)
    svc_model = RF(transformer=transformer)
    svc_model.load_state_dict(state_dict)
    svc_model = svc_model.to(device)

    use_fp16 = bool(use_fp16 and device.type == 'cuda')
    if use_fp16:
        svc_model = svc_model.half()

    svc_model.eval()

    vocoder = NsfHifiGAN(str(asset_paths["vocoder"]), device=device).to(device)
    rmvpe = RMVPE(model_path=str(asset_paths["rmvpe"]), hop_length=160, device=device)
    hubert = HubertModelWithFinalProj.from_pretrained(str(asset_paths["content_vec"])).to(device)
    rms_extractor = RMSExtractor().to(device)

    if use_fp16:
        vocoder = vocoder.half()
        hubert = hubert.half()
        rms_extractor = rms_extractor.half()

    return svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg


def extract_features(audio_segment, sample_rate, hop_length, rmvpe, hubert, rms_extractor,
                     device, key_shift=0, ds_cfg_strength=0.0, cvec_downsample_rate=2, target_loudness=-18.0,
                     robust_f0=0, use_fp16=True):
    """Extract all required features from an audio segment"""
    meter = pyln.Meter(sample_rate, block_size=0.1)
    original_loudness = meter.integrated_loudness(audio_segment)
    normalized_audio = pyln.normalize.loudness(audio_segment, original_loudness, target_loudness)

    max_amp = np.max(np.abs(normalized_audio))
    if max_amp > 1.0:
        normalized_audio = normalized_audio * (0.99 / max_amp)

    audio_tensor = torch.from_numpy(normalized_audio).float().unsqueeze(0).to(device)
    audio_16khz = torch.from_numpy(librosa.resample(normalized_audio, orig_sr=sample_rate, target_sr=16000)).float().unsqueeze(0).to(device)

    if use_fp16 and device.type != 'cpu':
        audio_tensor = audio_tensor.half()
        audio_16khz = audio_16khz.half()

    mel = get_mel_spectrogram(
        audio_tensor,
        sampling_rate=sample_rate,
        n_fft=2048,
        num_mels=128,
        hop_size=512,
        win_size=2048,
        fmin=40,
        fmax=16000
    ).transpose(1, 2)

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    with autocast(device_type=device_type, enabled=use_fp16):
        cvec = hubert(audio_16khz)["last_hidden_state"].squeeze(0)
    cvec = linear_interpolate_tensor(cvec, mel.shape[1])[None, :]

    if ds_cfg_strength > 0:
        cvec_ds = cvec.clone()
        cvec_ds = cvec_ds[0, ::2, :]
        cvec_ds = linear_interpolate_tensor(cvec_ds, cvec_ds.shape[0]//cvec_downsample_rate)
        cvec_ds = linear_interpolate_tensor(cvec_ds, mel.shape[1])[None, :]
    else:
        cvec_ds = None

    if robust_f0 > 0:
        time_step = hop_length / sample_rate
        f0_min = 40
        f0_max = 1100

        with autocast(device_type=device_type, enabled=use_fp16):
            rmvpe_f0 = rmvpe.infer_from_audio(audio_tensor, sample_rate=sample_rate, device=device)
        rmvpe_f0 = post_process_f0(rmvpe_f0, sample_rate, hop_length, mel.shape[1], silence_front=0.0, cut_last=False)
        pw_f0 = get_f0_pw(normalized_audio, sample_rate, time_step, f0_min, f0_max)
        pmac_f0 = get_f0_pm(normalized_audio, sample_rate, time_step, f0_min, f0_max)

        if robust_f0 == 1:
            with autocast(device_type=device_type, enabled=use_fp16):
                rms_np = rms_extractor(audio_tensor).squeeze().cpu().numpy()
            f0 = f0_ensemble_light(rmvpe_f0, pw_f0, pmac_f0, rms=rms_np)
        else:
            f0 = f0_ensemble(rmvpe_f0, pw_f0, pmac_f0)
    else:
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type=device_type, enabled=use_fp16):
            f0 = rmvpe.infer_from_audio(audio_tensor, sample_rate=sample_rate, device=device)
        f0 = post_process_f0(f0, sample_rate, hop_length, mel.shape[1], silence_front=0.0, cut_last=False)

    if key_shift != 0:
        f0 = f0 * 2 ** (key_shift / 12)
    f0 = torch.from_numpy(f0).float().to(device)[None, :]

    rms = rms_extractor(audio_tensor)

    return mel, cvec, cvec_ds, f0, rms, original_loudness


def run_inference(
    model, mel, cvec, f0, rms, cvec_ds, spk_id,
    infer_steps, ds_cfg_strength, spk_cfg_strength,
    skip_cfg_strength, cfg_skip_layers, cfg_rescale,
    frame_lengths=None, use_fp16=True, noise=None, generator=None
):
    """Run the actual inference through the model with optional batch processing"""
    device_type = 'cuda' if mel.device.type == 'cuda' else 'cpu'

    with autocast(device_type=device_type, enabled=use_fp16):
        mel_out, _ = model.sample(
            src_mel=mel,
            spk_id=spk_id,
            f0=f0,
            rms=rms,
            cvec=cvec,
            steps=infer_steps,
            bad_cvec=cvec_ds,
            ds_cfg_strength=ds_cfg_strength,
            spk_cfg_strength=spk_cfg_strength,
            skip_cfg_strength=skip_cfg_strength,
            cfg_skip_layers=cfg_skip_layers,
            cfg_rescale=cfg_rescale,
            frame_len=frame_lengths,
            noise=noise,
            generator=generator,
        )

    return mel_out


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
    """Generate audio from mel spectrogram using vocoder"""
    device_type = 'cuda' if mel_out.device.type == 'cuda' else 'cpu'
    with autocast(device_type=device_type, enabled=use_fp16):
        audio_out = vocoder(mel_out.transpose(1, 2), f0)
    audio_out = audio_out.squeeze().cpu().numpy()

    if expected_length is not None:
        if len(audio_out) > expected_length:
            audio_out = audio_out[:expected_length]
        elif len(audio_out) < expected_length:
            audio_out = np.pad(audio_out, (0, expected_length - len(audio_out)), 'constant')

    if restore_loudness and original_loudness is not None:
        meter = pyln.Meter(sample_rate, block_size=0.1)
        audio_out_loudness = meter.integrated_loudness(audio_out)
        audio_out = pyln.normalize.loudness(audio_out, audio_out_loudness, original_loudness)

        max_amp = np.max(np.abs(audio_out))
        if max_amp > 1.0:
            audio_out = audio_out * (0.99 / max_amp)

    return audio_out


def process_segment(
    audio_segment,
    svc_model, vocoder, rmvpe, hubert, rms_extractor,
    speaker_id, sample_rate, hop_length, device,
    key_shift=0,
    infer_steps=32,
    ds_cfg_strength=0.0,
    spk_cfg_strength=0.0,
    skip_cfg_strength=0.0,
    cfg_skip_layers=None,
    cfg_rescale=0.7,
    cvec_downsample_rate=2,
    target_loudness=-18.0,
    restore_loudness=True,
    robust_f0=0,
    use_fp16=True,
    seed=None,
):
    """Process a single audio segment and return the converted audio"""
    mel, cvec, cvec_ds, f0, rms, original_loudness = extract_features(
        audio_segment, sample_rate, hop_length, rmvpe, hubert, rms_extractor,
        device, key_shift, ds_cfg_strength, cvec_downsample_rate, target_loudness,
        robust_f0, use_fp16
    )

    spk_id = torch.LongTensor([speaker_id]).to(device)

    frame_length = torch.tensor([mel.shape[1]], device=device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    mel_out = run_inference(
        model=svc_model,
        mel=mel,
        cvec=cvec,
        f0=f0,
        rms=rms,
        cvec_ds=cvec_ds,
        spk_id=spk_id,
        infer_steps=infer_steps,
        ds_cfg_strength=ds_cfg_strength,
        spk_cfg_strength=spk_cfg_strength,
        skip_cfg_strength=skip_cfg_strength,
        cfg_skip_layers=cfg_skip_layers,
        cfg_rescale=cfg_rescale,
        frame_lengths=frame_length,
        use_fp16=use_fp16,
        generator=generator,
    )

    audio_out = generate_audio(
        vocoder, mel_out, f0,
        original_loudness if restore_loudness else None,
        restore_loudness, use_fp16,
        expected_length=len(audio_segment),
        sample_rate=sample_rate,
    )

    return audio_out


def batch_process_segments(
    segments_with_pos,
    svc_model, vocoder, rmvpe, hubert, rms_extractor,
    speaker_id, sample_rate, hop_length, device,
    key_shift=0,
    infer_steps=32,
    ds_cfg_strength=0.0,
    spk_cfg_strength=0.0,
    skip_cfg_strength=0.0,
    cfg_skip_layers=None,
    cfg_rescale=0.7,
    cvec_downsample_rate=2,
    target_loudness=-18.0,
    restore_loudness=True,
    robust_f0=0,
    use_fp16=True,
    batch_size=1,
    gr_progress=None,
    progress_desc=None,
    seed=None,
):
    """Process audio segments in batches for faster inference"""
    if batch_size <= 1:
        results = []
        for i, (start_sample, chunk) in enumerate(tqdm(segments_with_pos, desc="Processing segments")):
            if gr_progress is not None:
                gr_progress(0.2 + (0.7 * (i / len(segments_with_pos))), desc=progress_desc.format(i+1, len(segments_with_pos)))
            audio_out = process_segment(
                chunk, svc_model, vocoder, rmvpe, hubert, rms_extractor,
                speaker_id, sample_rate, hop_length, device,
                key_shift, infer_steps, ds_cfg_strength, spk_cfg_strength,
                skip_cfg_strength, cfg_skip_layers, cfg_rescale,
                cvec_downsample_rate, target_loudness, restore_loudness,
                robust_f0,
                use_fp16,
                None if seed is None else seed + i,
            )
            results.append((start_sample, audio_out, len(chunk)))
        return results

    sorted_with_idx = sorted(enumerate(segments_with_pos), key=lambda x: len(x[1][1]))
    sorted_segments = []

    for orig_idx, (pos, chunk) in sorted_with_idx:
        sorted_segments.append((orig_idx, pos, chunk))

    batched_segments = [sorted_segments[i:i + batch_size] for i in range(0, len(sorted_segments), batch_size)]

    all_results = []

    for batch_idx, batch in enumerate(tqdm(batched_segments, desc="Processing batches")):
        if gr_progress is not None:
            gr_progress(
                0.2 + (0.7 * (batch_idx / len(batched_segments))),
                desc=progress_desc.format(batch_idx+1, len(batched_segments)))

        batch_original_indices = [orig_idx for orig_idx, _, _ in batch]
        batch_start_samples = [pos for _, pos, _ in batch]
        batch_chunks = [chunk for _, _, chunk in batch]
        batch_lengths = [len(chunk) for chunk in batch_chunks]

        batch_features = []
        for chunk in batch_chunks:
            mel, cvec, cvec_ds, f0, rms, original_loudness = extract_features(
                chunk, sample_rate, hop_length, rmvpe, hubert, rms_extractor,
                device, key_shift, ds_cfg_strength, cvec_downsample_rate, target_loudness,
                robust_f0, use_fp16
            )
            batch_features.append({
                'mel': mel,
                'cvec': cvec,
                'cvec_ds': cvec_ds,
                'f0': f0,
                'rms': rms,
                'original_loudness': original_loudness,
                'length': mel.shape[1]
            })

        max_length = max(feat['length'] for feat in batch_features)

        padded_mels = []
        padded_cvecs = []
        padded_f0s = []
        padded_rmss = []
        frame_lengths = []
        original_loudness_values = []

        if ds_cfg_strength > 0:
            padded_cvec_ds = []

        for feat in batch_features:
            curr_len = feat['length']
            frame_lengths.append(curr_len)

            padded_mels.append(pad_tensor_to_length(feat['mel'], max_length))
            padded_cvecs.append(pad_tensor_to_length(feat['cvec'], max_length))
            padded_f0s.append(pad_tensor_to_length(feat['f0'], max_length))
            padded_rmss.append(pad_tensor_to_length(feat['rms'], max_length))

            if ds_cfg_strength > 0:
                padded_cvec_ds.append(pad_tensor_to_length(feat['cvec_ds'], max_length))

            original_loudness_values.append(feat['original_loudness'])

        batched_mel = torch.cat(padded_mels, dim=0)
        batched_cvec = torch.cat(padded_cvecs, dim=0)
        batched_f0 = torch.cat(padded_f0s, dim=0)
        batched_rms = torch.cat(padded_rmss, dim=0)

        if ds_cfg_strength > 0:
            batched_cvec_ds = torch.cat(padded_cvec_ds, dim=0)
        else:
            batched_cvec_ds = None

        batched_noise = None
        if seed is not None:
            noises = []
            for feature, original_index in zip(batch_features, batch_original_indices):
                generator = torch.Generator(device=device).manual_seed(seed + original_index)
                noise = torch.randn(
                    1,
                    feature['length'],
                    feature['mel'].shape[-1],
                    device=device,
                    dtype=batched_mel.dtype,
                    generator=generator,
                )
                noises.append(pad_tensor_to_length(noise, max_length))
            batched_noise = torch.cat(noises, dim=0)

        frame_lengths = torch.tensor(frame_lengths, device=device)

        batch_spk_id = torch.LongTensor([speaker_id] * len(batch)).to(device)

        with torch.no_grad():
            mel_out = run_inference(
                model=svc_model,
                mel=batched_mel,
                cvec=batched_cvec,
                f0=batched_f0,
                rms=batched_rms,
                cvec_ds=batched_cvec_ds,
                spk_id=batch_spk_id,
                infer_steps=infer_steps,
                ds_cfg_strength=ds_cfg_strength,
                spk_cfg_strength=spk_cfg_strength,
                skip_cfg_strength=skip_cfg_strength,
                cfg_skip_layers=cfg_skip_layers,
                cfg_rescale=cfg_rescale,
                frame_lengths=frame_lengths,
                use_fp16=use_fp16,
                noise=batched_noise,
            )

            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=use_fp16):
                audio_out = vocoder(mel_out.transpose(1, 2), batched_f0)

            for i in range(len(batch)):
                expected_audio_length = batch_lengths[i]

                curr_audio = audio_out[i].squeeze().cpu().numpy()

                if len(curr_audio) > expected_audio_length:
                    curr_audio = curr_audio[:expected_audio_length]
                elif len(curr_audio) < expected_audio_length:
                    curr_audio = np.pad(curr_audio, (0, expected_audio_length - len(curr_audio)), 'constant')

                if restore_loudness:
                    meter = pyln.Meter(sample_rate, block_size=0.1)
                    curr_loudness = meter.integrated_loudness(curr_audio)
                    curr_audio = pyln.normalize.loudness(curr_audio, curr_loudness, original_loudness_values[i])

                    max_amp = np.max(np.abs(curr_audio))
                    if max_amp > 1.0:
                        curr_audio = curr_audio * (0.99 / max_amp)

                expected_length = batch_lengths[i]

                all_results.append((batch_idx, i, batch_start_samples[i], curr_audio, expected_length, batch_original_indices[i]))

    all_results.sort(key=lambda x: x[5])

    return [(pos, audio, length) for _, _, pos, audio, length, _ in all_results]


def pad_tensor_to_length(tensor, length):
    """Pad a tensor to the specified length along the sequence dimension (dim=1)"""
    curr_len = tensor.shape[1]
    if curr_len >= length:
        return tensor

    pad_len = length - curr_len

    if tensor.dim() == 2:
        padding = (0, pad_len)
    elif tensor.dim() == 3:
        padding = (0, 0, 0, pad_len)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")

    padded = torch.nn.functional.pad(tensor, padding, "constant", 0)
    return padded


class InferenceRuntime:
    """Reusable, process-local inference service.

    All neural networks are loaded once in ``__init__``.  ``convert`` only
    handles audio-specific state, so a notebook can process many files without
    repeatedly constructing ContentVec, RMVPE, and HiFi-GAN.
    """

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
            self.dataset_cfg,
        ) = load_models(
            model_path,
            self.device,
            self.use_fp16,
            assets_dir,
        )
        self.sample_rate = int(self.dataset_cfg.get("sample_rate", 44100))
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
        skip_cfg_strength=0.0,
        cfg_skip_layers=None,
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
        batch_size=1,
        seed=None,
    ):
        if speaker not in self.spk2idx:
            valid = ", ".join(self.spk2idx)
            raise ValueError(f"unknown speaker {speaker!r}; valid speakers: {valid}")
        if infer_steps < 1:
            raise ValueError("infer_steps must be at least 1")
        if robust_f0 not in (0, 1, 2):
            raise ValueError("robust_f0 must be 0, 1, or 2")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
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
        if batch_size <= 1:
            result = np.zeros(
                len(audio) + max(0, fade_samples), dtype=np.float32
            )
            with torch.no_grad():
                for index, (start_sample, chunk) in enumerate(
                    tqdm(segments, desc="Processing segments")
                ):
                    converted = process_segment(
                        chunk,
                        self.svc_model,
                        self.vocoder,
                        self.rmvpe,
                        self.hubert,
                        self.rms_extractor,
                        self.spk2idx[speaker],
                        self.sample_rate,
                        self.hop_length,
                        self.device,
                        key_shift,
                        infer_steps,
                        ds_cfg_strength,
                        spk_cfg_strength,
                        skip_cfg_strength,
                        cfg_skip_layers,
                        cfg_rescale,
                        cvec_downsample_rate,
                        target_loudness,
                        restore_loudness,
                        robust_f0,
                        self.use_fp16,
                        None if seed is None else seed + index,
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

        with torch.no_grad():
            processed = batch_process_segments(
                segments,
                self.svc_model,
                self.vocoder,
                self.rmvpe,
                self.hubert,
                self.rms_extractor,
                self.spk2idx[speaker],
                self.sample_rate,
                self.hop_length,
                self.device,
                key_shift,
                infer_steps,
                ds_cfg_strength,
                spk_cfg_strength,
                skip_cfg_strength,
                cfg_skip_layers,
                cfg_rescale,
                cvec_downsample_rate,
                target_loudness,
                restore_loudness,
                robust_f0,
                self.use_fp16,
                batch_size,
                seed=seed,
            )
        return assemble_segments(processed, len(audio), fade_samples)
