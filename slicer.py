"""Silence-based audio slicer used by the inference runtime."""

from __future__ import annotations

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class Slicer:
    """Split mono or channel-first audio while retaining original sample offsets."""

    def __init__(
        self,
        sr: int,
        threshold: float = -30.0,
        min_length: int = 3000,
        min_interval: int = 100,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ) -> None:
        if not min_length >= min_interval >= hop_size:
            raise ValueError("min_length >= min_interval >= hop_size is required")
        if max_sil_kept < hop_size:
            raise ValueError("max_sil_kept >= hop_size is required")

        min_interval_samples = sr * min_interval / 1000
        self.sr = sr
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval_samples), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval_samples / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform: np.ndarray, begin: int, end: int) -> np.ndarray:
        start_sample = begin * self.hop_size
        if waveform.ndim > 1:
            end_sample = min(waveform.shape[1], end * self.hop_size)
            return waveform[:, start_sample:end_sample]
        end_sample = min(waveform.shape[0], end * self.hop_size)
        return waveform[start_sample:end_sample]

    def slice(self, waveform: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Return `(start_sample, chunk)` pairs for each non-silent region."""
        samples = librosa.to_mono(waveform) if waveform.ndim > 1 else waveform
        if samples.shape[0] <= self.min_length:
            return [(0, waveform)]

        rms_list = librosa.feature.rms(
            y=samples,
            frame_length=self.win_size,
            hop_length=self.hop_size,
        ).squeeze(0)
        silence_tags: list[tuple[int, int]] = []
        silence_start = None
        clip_start = 0
        for index, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = index
                continue
            if silence_start is None:
                continue

            is_leading_silence = silence_start == 0 and index > self.max_sil_kept
            need_middle_slice = (
                index - silence_start >= self.min_interval
                and index - clip_start >= self.min_length
            )
            if not is_leading_silence and not need_middle_slice:
                silence_start = None
                continue

            silence_length = index - silence_start
            if silence_length <= self.max_sil_kept:
                position = rms_list[silence_start : index + 1].argmin() + silence_start
                silence_tags.append(
                    (0, int(position))
                    if silence_start == 0
                    else (int(position), int(position))
                )
                clip_start = int(position)
            elif silence_length <= self.max_sil_kept * 2:
                position = rms_list[
                    index - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                position += index - self.max_sil_kept
                position_left = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                position_right = (
                    rms_list[index - self.max_sil_kept : index + 1].argmin()
                    + index
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    silence_tags.append((0, int(position_right)))
                    clip_start = int(position_right)
                else:
                    silence_tags.append(
                        (
                            int(min(position_left, position)),
                            int(max(position_right, position)),
                        )
                    )
                    clip_start = int(max(position_right, position))
            else:
                position_left = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                position_right = (
                    rms_list[index - self.max_sil_kept : index + 1].argmin()
                    + index
                    - self.max_sil_kept
                )
                silence_tags.append(
                    (0, int(position_right))
                    if silence_start == 0
                    else (int(position_left), int(position_right))
                )
                clip_start = int(position_right)
            silence_start = None

        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            position = (
                rms_list[silence_start : silence_end + 1].argmin() + silence_start
            )
            silence_tags.append((int(position), total_frames + 1))

        if not silence_tags:
            return [(0, waveform)]

        chunks: list[tuple[int, np.ndarray]] = []
        if silence_tags[0][0] > 0:
            chunks.append((0, self._apply_slice(waveform, 0, silence_tags[0][0])))

        for index in range(1, len(silence_tags)):
            start_frame = silence_tags[index - 1][1]
            end_frame = silence_tags[index][0]
            if start_frame < end_frame:
                chunks.append(
                    (
                        start_frame * self.hop_size,
                        self._apply_slice(waveform, start_frame, end_frame),
                    )
                )

        waveform_length = waveform.shape[-1]
        if silence_tags[-1][1] * self.hop_size < waveform_length:
            start_frame = silence_tags[-1][1]
            chunks.append(
                (
                    start_frame * self.hop_size,
                    self._apply_slice(waveform, start_frame, total_frames),
                )
            )

        for index, (start_sample, chunk) in enumerate(chunks):
            start_seconds = start_sample / self.sr
            duration_seconds = chunk.shape[-1] / self.sr
            end_seconds = start_seconds + duration_seconds
            start_minutes, start_remainder = divmod(start_seconds, 60)
            end_minutes, end_remainder = divmod(end_seconds, 60)
            logger.info(
                "Chunk %d: Start=%02d:%05.2f, End=%02d:%05.2f, Duration=%.2fs",
                index,
                int(start_minutes),
                start_remainder,
                int(end_minutes),
                end_remainder,
                duration_seconds,
            )

        return chunks
