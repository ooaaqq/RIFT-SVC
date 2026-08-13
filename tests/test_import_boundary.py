from __future__ import annotations

import subprocess
import sys


def test_inference_import_has_no_model_side_effects() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import rift_svc; "
                "from rift_svc.inference.audio import write_audio; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""
