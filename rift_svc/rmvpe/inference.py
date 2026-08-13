import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from .constants import (
    CONST,
    MEL_FMAX,
    MEL_FMIN,
    N_CLASS,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_LENGTH,
)
from .model import E2E0
from .spec import MelSpectrogram


def to_local_average_f0(hidden, center=None, thred=0.03):
    idx = torch.arange(N_CLASS, device=hidden.device)[None, None, :]
    idx_cents = idx * 20 + CONST
    if center is None:
        center = torch.argmax(hidden, dim=2, keepdim=True)
    start = torch.clip(center - 4, min=0)
    end = torch.clip(center + 5, max=N_CLASS)
    weights = hidden * ((idx >= start) & (idx < end))
    weight_sum = torch.sum(weights, dim=2)
    cents = torch.sum(weights * idx_cents, dim=2) / (weight_sum + (weight_sum == 0))
    f0 = 10 * 2 ** (cents / 1200)
    f0 = f0 * ~(hidden.max(dim=2)[0] < thred)
    return f0.squeeze(0).cpu().numpy()


class RMVPE:
    def __init__(self, model_path, hop_length=160, device="cpu"):
        model = E2E0(4, 1, (2, 2))
        ckpt = torch.load(model_path, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()

        self.model = model.to(device)
        self.mel_extractor = MelSpectrogram(
            N_MELS,
            SAMPLE_RATE,
            WINDOW_LENGTH,
            hop_length,
            None,
            MEL_FMIN,
            MEL_FMAX,
        ).to(device)
        self.resample_kernel = {}

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            padded_frames = 32 * ((n_frames - 1) // 32 + 1)
            mel = F.pad(mel, (0, padded_frames - n_frames), mode="constant")
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        return to_local_average_f0(hidden, thred=thred)

    def infer_from_audio(
        self,
        audio,
        sample_rate=SAMPLE_RATE,
        device=None,
        thred=0.03,
    ):
        if sample_rate == SAMPLE_RATE:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(
                    sample_rate,
                    SAMPLE_RATE,
                    lowpass_filter_width=128,
                )
            target_device = audio.device if device is None else device
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(
                target_device
            )
            audio_res = self.resample_kernel[key_str](audio)

        mel = self.mel_extractor(audio_res, center=True)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden, thred=thred)
        return f0
