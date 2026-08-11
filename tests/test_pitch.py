from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def test_f0_ensembles_align_mismatched_extractor_lengths(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "parselmouth", types.ModuleType("parselmouth"))
    monkeypatch.setitem(sys.modules, "pyworld", types.ModuleType("pyworld"))
    pitch = importlib.import_module("rift_svc.inference.pitch")

    result = pitch.f0_ensemble(
        np.array([220.0, 0.0, 230.0]),
        np.array([220.0, 221.0, 230.0, 231.0]),
        np.array([220.0]),
    )

    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_light_f0_ensemble_pads_short_rms() -> None:
    pitch = importlib.import_module("rift_svc.inference.pitch")

    result = pitch.f0_ensemble_light(
        np.array([220.0, 0.0, 230.0]),
        np.array([220.0, 221.0]),
        np.array([220.0, 221.0, 230.0]),
        rms=np.array([1.0]),
    )

    assert result.shape == (3,)
    assert np.all(np.isfinite(result))
