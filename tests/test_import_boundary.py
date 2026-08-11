from __future__ import annotations

import subprocess
import sys


def test_inference_import_does_not_load_training_stack() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import rift_svc; "
                "from rift_svc.inference.output import write_audio; "
                "assert 'pytorch_lightning' not in sys.modules; "
                "assert 'wandb' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""
