from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("kernel_name", ["rift", "separation"])
def test_pascal_gpu_installs_compatible_torch(kernel_name: str) -> None:
    kernel_path = Path(__file__).parents[1] / f"kaggle/{kernel_name}/kernel.py"
    spec = importlib.util.spec_from_file_location(f"{kernel_name}_kernel", kernel_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with (
        patch.object(module.subprocess, "check_output", return_value="6.0\n"),
        patch.object(module, "run") as run,
    ):
        module.install_pascal_torch_if_needed()

    command = run.call_args.args[0]
    assert "torch==2.6.0" in command
    if kernel_name == "rift":
        assert "torchaudio==2.6.0" in command
