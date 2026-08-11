"""Training-only helpers kept out of the inference import graph."""

from __future__ import annotations

import time

import torch
from pytorch_lightning.callbacks import TQDMProgressBar


def l2_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    gradients = [
        parameter.grad.detach().flatten()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else None
        return torch.zeros((), device=device)
    return torch.cat(gradients).norm(2)


class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.total_steps = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.start_time = time.time()
        self.total_steps = trainer.max_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        current_step = trainer.global_step
        elapsed_time = time.time() - self.start_time
        average_step_time = elapsed_time / current_step if current_step > 0 else 0
        remaining_steps = self.total_steps - current_step
        remaining_time = average_step_time * remaining_steps

        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours}:{minutes:02d}:{seconds:02d}"

        self.train_progress_bar.set_postfix(
            {
                "loss": f"{outputs['loss'].item():.4f}",
                "elapsed_time": f"{format_time(elapsed_time)}/{format_time(remaining_time)}",
                "remaining_steps": f"{remaining_steps}/{self.total_steps}",
            }
        )


def load_state_dict(model, state_dict, strict=False):
    """Load a checkpoint state dict with or without a ``model.`` prefix."""
    if any(key.startswith("model.") for key in state_dict):
        state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
        }
    return model.load_state_dict(state_dict, strict=strict)
