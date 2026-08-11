"""Pitch extraction helpers used only by the inference pipeline."""

from __future__ import annotations

import numpy as np
import parselmouth as pm
import pyworld as pw


def post_process_f0(
    f0: np.ndarray,
    sample_rate: int,
    hop_length: int,
    n_frames: int,
    silence_front: float = 0.0,
    cut_last: bool = True,
) -> np.ndarray:
    start_frame = int(silence_front * sample_rate / hop_length)
    f0 = np.asarray(f0, dtype=np.float64).copy()
    unvoiced = f0 == 0
    if np.any(~unvoiced):
        f0[unvoiced] = np.interp(
            np.where(unvoiced)[0],
            np.where(~unvoiced)[0],
            f0[~unvoiced],
        )
    else:
        f0 = np.zeros_like(f0)

    origin_time = 0.01 * np.arange(len(f0))
    target_time = hop_length / sample_rate * np.arange(
        max(0, n_frames - start_frame)
    )
    f0 = np.interp(target_time, origin_time, f0)
    voiced = np.interp(
        target_time, origin_time, unvoiced.astype(float)
    ) > 0.5
    f0[voiced] = 0
    f0 = np.pad(f0, (start_frame, 0), mode="constant")
    return f0[:-1] if cut_last else f0


def slide_nanmedian(signals: np.ndarray, win_length: int = 3) -> np.ndarray:
    signals = np.asarray(signals)
    if signals.size == 0:
        return signals.copy()
    filtered = np.empty_like(signals)
    half = win_length // 2
    for index in range(signals.shape[0]):
        start = max(0, index - half)
        end = min(signals.shape[0], index + half + 1)
        filtered[index] = np.nanmedian(signals[start:end])
    return filtered


def _fit_length(
    values: np.ndarray,
    target_length: int,
    fill_value: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) >= target_length:
        return values[:target_length]
    return np.pad(
        values,
        (0, target_length - len(values)),
        mode="constant",
        constant_values=fill_value,
    )


def _stack_aligned_f0(
    rmvpe_f0: np.ndarray,
    pw_f0: np.ndarray,
    pmac_f0: np.ndarray,
) -> np.ndarray:
    target_length = len(rmvpe_f0)
    return np.stack(
        [
            _fit_length(pw_f0, target_length, np.nan),
            _fit_length(pmac_f0, target_length, np.nan),
            _fit_length(rmvpe_f0, target_length, np.nan),
        ],
        axis=0,
    )


def get_f0_pw(
    audio: np.ndarray,
    sr: int,
    time_step: float,
    f0_min: float,
    f0_max: float,
) -> np.ndarray:
    raw_f0, times = pw.dio(
        audio.astype(np.double),
        sr,
        f0_floor=f0_min,
        f0_ceil=f0_max,
        frame_period=time_step * 1000,
    )
    f0 = pw.stonemask(audio.astype(np.double), raw_f0, times, sr)
    f0[f0 == 0] = np.nan
    return slide_nanmedian(f0, 3)


def get_f0_pm(
    audio: np.ndarray,
    sr: int,
    time_step: float,
    f0_min: float,
    f0_max: float,
) -> np.ndarray:
    pitch = pm.Sound(audio, sampling_frequency=sr).to_pitch_ac(
        time_step=time_step,
        voicing_threshold=0.6,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max,
        very_accurate=True,
        octave_jump_cost=0.5,
    )
    f0 = pitch.selected_array["frequency"]
    f0[f0 == 0] = np.nan
    return slide_nanmedian(f0, 3)


def f0_ensemble(rmvpe_f0: np.ndarray, pw_f0: np.ndarray, pmac_f0: np.ndarray) -> np.ndarray:
    stack_f0 = _stack_aligned_f0(rmvpe_f0, pw_f0, pmac_f0)
    median_f0 = np.nanmedian(stack_f0, axis=0)
    nan_count = np.sum(np.isnan(stack_f0), axis=0)
    median_f0[nan_count >= 2] = np.nan

    smoothed = slide_nanmedian(median_f0, 41)
    deviation = np.abs(median_f0 - smoothed)
    median_f0[deviation > 96] = smoothed[deviation > 96]

    one_missing = nan_count == 1
    if np.any(one_missing):
        available_min = np.nanmin(stack_f0[:, one_missing], axis=0)
        available_max = np.nanmax(stack_f0[:, one_missing], axis=0)
        median_f0[one_missing] = np.where(
            np.abs(available_min - smoothed[one_missing])
            < np.abs(available_max - smoothed[one_missing]),
            available_min,
            available_max,
        )

    median_f0 = slide_nanmedian(median_f0, 3)
    median_f0[nan_count >= 2] = np.nan
    median_f0[np.isnan(median_f0)] = 0
    return median_f0


def f0_ensemble_light(
    rmvpe_f0: np.ndarray,
    pw_f0: np.ndarray,
    pmac_f0: np.ndarray,
    rms: np.ndarray | None = None,
    rms_threshold: float = 0.05,
) -> np.ndarray:
    stack_f0 = _stack_aligned_f0(rmvpe_f0, pw_f0, pmac_f0)
    corrected_f0 = stack_f0[2].copy()
    correction_mask = (rmvpe_f0 == 0) & np.any(
        (~np.isnan(stack_f0[:2])) & (stack_f0[:2] > 0), axis=0
    )
    if rms is not None:
        rms = _fit_length(rms, len(corrected_f0), 0.0)
        correction_mask &= rms > rms_threshold

    for index in np.where(correction_mask)[0]:
        valid = stack_f0[:, index]
        valid = valid[~np.isnan(valid) & (valid > 0)]
        if len(valid):
            corrected_f0[index] = np.median(valid)
    corrected_f0[np.isnan(corrected_f0)] = 0
    return corrected_f0
