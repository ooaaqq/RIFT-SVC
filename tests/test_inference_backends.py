import hashlib
from types import SimpleNamespace

import torch

from rift_svc.nsf_hifigan import vocoder as vocoder_module
from rift_svc.rmvpe import inference as rmvpe_module
from rift_svc.rmvpe.model import E2E0


def test_rmvpe_checkpoint_structure_is_stable():
    keys = "\n".join(E2E0(4, 1, (2, 2)).state_dict())

    assert len(keys.splitlines()) == 801
    assert hashlib.sha256(keys.encode()).hexdigest() == (
        "6aed931f9325f8959d9b2092745eb24922337cb96c26160f8e079f8e8a045913"
    )


def test_rmvpe_loads_checkpoint_strictly(monkeypatch):
    calls = {}

    class FakeModel:
        def load_state_dict(self, state_dict, strict=True):
            calls["state_dict"] = state_dict
            calls["strict"] = strict

        def eval(self):
            return self

        def to(self, device):
            return self

    class FakeMel:
        def __init__(self, *args):
            pass

        def to(self, device):
            return self

    monkeypatch.setattr(rmvpe_module, "E2E0", lambda *args: FakeModel())
    monkeypatch.setattr(rmvpe_module, "MelSpectrogram", FakeMel)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model": {"w": 1}})

    rmvpe_module.RMVPE("model.pt")

    assert calls == {"state_dict": {"w": 1}, "strict": True}


def test_hifigan_loads_generator_during_construction(monkeypatch):
    load_count = 0

    class FakeGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, mel, f0):
            return mel[:, :1, :]

    def fake_load_model(model_path, device):
        nonlocal load_count
        load_count += 1
        return FakeGenerator(), SimpleNamespace()

    monkeypatch.setattr(vocoder_module, "load_model", fake_load_model)

    vocoder = vocoder_module.NsfHifiGAN("model.ckpt", device="cpu")
    assert load_count == 1

    mel = torch.zeros(1, 128, 4)
    f0 = torch.zeros(1, 4)
    vocoder(mel, f0)
    vocoder(mel, f0)

    assert load_count == 1


def test_hifigan_half_converts_generator(monkeypatch):
    generator = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        vocoder_module,
        "load_model",
        lambda model_path, device: (generator, SimpleNamespace()),
    )

    vocoder = vocoder_module.NsfHifiGAN("model.ckpt", device="cpu").half()

    assert next(vocoder.model.parameters()).dtype == torch.float16
