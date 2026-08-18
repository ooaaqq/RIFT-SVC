from __future__ import annotations

import argparse
import json
import queue
from pathlib import Path

from kaggle.rift.batch_worker import collect_worker_results, run_batch


def test_worker_results_are_consumed_before_process_join() -> None:
    class ResultQueue:
        consumed = False

        def get(self, timeout):
            assert timeout > 0
            self.consumed = True
            return {"index": 1}

        def get_nowait(self):
            raise queue.Empty

    class Process:
        def __init__(self, result_queue):
            self.result_queue = result_queue

        def is_alive(self):
            return not self.result_queue.consumed

        def join(self):
            assert self.result_queue.consumed

    result_queue = ResultQueue()

    assert collect_worker_results([Process(result_queue)], result_queue) == [
        {"index": 1}
    ]


def test_two_workers_share_one_dynamic_batch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    inference = repo / "rift_svc/inference"
    inference.mkdir(parents=True)
    (repo / "rift_svc/__init__.py").write_text("")
    (inference / "__init__.py").write_text("")
    (inference / "runtime.py").write_text(
        """
import time
from pathlib import Path

class InferenceRuntime:
    def __init__(self, model, *, device, use_fp16, assets_dir):
        self.device = device
        self.sample_rate = 44100

    def convert(self, input_path, **params):
        time.sleep(0.05)
        return Path(input_path).read_bytes()
"""
    )
    (inference / "audio.py").write_text(
        """
from pathlib import Path

def write_audio(path, audio, sample_rate, subtype):
    Path(path).write_bytes(audio)
"""
    )

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    asset_dir = tmp_path / "assets"
    input_dir.mkdir()
    output_dir.mkdir()
    (asset_dir / "model").mkdir(parents=True)
    (asset_dir / "pretrained").mkdir()
    (asset_dir / "model/rift25k.ckpt").write_bytes(b"model")

    items = []
    for index in range(1, 5):
        input_name = f"input-{index:04d}.wav"
        (input_dir / input_name).write_bytes(f"audio-{index}".encode())
        items.append(
            {
                "index": index,
                "input_name": input_name,
                "transport_output_name": f"output-{index:04d}.wav",
            }
        )
    job = {
        "model_filename": "rift25k.ckpt",
        "items": items,
        "params": {
            "speaker": "target",
            "key_shift": 0,
            "infer_steps": 32,
            "ds_cfg_strength": 0.2,
            "spk_cfg_strength": 0.8,
            "cfg_rescale": 0.7,
            "robust_f0": 0,
            "seed": 7,
        },
    }
    job_path = tmp_path / "job.json"
    results_path = tmp_path / "results.json"
    job_path.write_text(json.dumps(job))

    results = run_batch(
        argparse.Namespace(
            repo_dir=repo,
            asset_dir=asset_dir,
            job=job_path,
            input_dir=input_dir,
            output_dir=output_dir,
            results=results_path,
            workers=2,
        )
    )

    assert {result["index"] for result in results} == {1, 2, 3, 4}
    assert {result["device"] for result in results} == {"cuda:0", "cuda:1"}
    for index in range(1, 5):
        assert (output_dir / f"output-{index:04d}.wav").read_bytes() == (
            f"audio-{index}".encode()
        )
    assert json.loads(results_path.read_text()) == results
