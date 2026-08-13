import torch

from .models import load_model


class NsfHifiGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = load_model(model_path, device=device)

    def forward(self, mel: torch.Tensor, f0: torch.Tensor):
        with torch.no_grad():
            return self.model(mel, f0)
