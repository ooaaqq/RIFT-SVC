"""Run a dynamic multi-file RIFT queue with one persistent runtime per GPU."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    return parser.parse_args()


def gpu_worker(
    worker_index: int,
    repo_dir: str,
    asset_dir: str,
    job: dict,
    input_dir: str,
    output_dir: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    device = f"cuda:{worker_index}"
    try:
        sys.path.insert(0, repo_dir)
        from rift_svc.inference.audio import write_audio
        from rift_svc.inference.runtime import InferenceRuntime

        runtime = InferenceRuntime(
            str(Path(asset_dir) / "model" / job["model_filename"]),
            device=device,
            use_fp16=True,
            assets_dir=Path(asset_dir) / "pretrained",
        )
    except Exception as exc:  # noqa: BLE001 - report worker initialization failures
        result_queue.put(
            {
                "fatal": True,
                "device": device,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return

    params = job["params"]
    while True:
        item = task_queue.get()
        if item is None:
            return
        try:
            audio = runtime.convert(
                Path(input_dir) / item["input_name"],
                speaker=params["speaker"],
                key_shift=params["key_shift"],
                infer_steps=params["infer_steps"],
                ds_cfg_strength=params["ds_cfg_strength"],
                spk_cfg_strength=params["spk_cfg_strength"],
                cfg_rescale=params["cfg_rescale"],
                robust_f0=params["robust_f0"],
                seed=params["seed"],
            )
            output = Path(output_dir) / item["transport_output_name"]
            write_audio(output, audio, runtime.sample_rate, subtype="FLOAT")
            result_queue.put({"index": item["index"], "device": device, "error": None})
        except Exception as exc:  # noqa: BLE001 - keep the remaining batch alive
            result_queue.put(
                {
                    "index": item["index"],
                    "device": device,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )


def collect_worker_results(
    processes: list[mp.Process], result_queue: mp.Queue
) -> list[dict]:
    """Drain results while workers run so their queue feeder cannot block exit."""
    results = []
    while any(process.is_alive() for process in processes):
        try:
            results.append(result_queue.get(timeout=0.2))
        except queue.Empty:
            continue
    for process in processes:
        process.join()
    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            return results


def run_batch(args: argparse.Namespace) -> list[dict]:
    job = json.loads(args.job.read_text(encoding="utf-8"))
    items = job.get("items", [])
    if not items:
        raise ValueError("batch job has no items")
    if not 1 <= args.workers <= 2:
        raise ValueError("workers must be 1 or 2")

    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    for item in items:
        task_queue.put(item)
    for _ in range(args.workers):
        task_queue.put(None)

    processes = [
        context.Process(
            target=gpu_worker,
            args=(
                worker_index,
                str(args.repo_dir),
                str(args.asset_dir),
                job,
                str(args.input_dir),
                str(args.output_dir),
                task_queue,
                result_queue,
            ),
        )
        for worker_index in range(args.workers)
    ]
    for process in processes:
        process.start()
    results = collect_worker_results(processes, result_queue)
    failed_processes = [process.exitcode for process in processes if process.exitcode]
    if failed_processes:
        results.append(
            {"fatal": True, "error": f"worker exit codes: {failed_processes}"}
        )
    args.results.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    completed = {result.get("index") for result in results if "index" in result}
    fatals = [result for result in results if result.get("fatal")]
    if fatals or len(completed) != len(items):
        raise RuntimeError(
            f"batch incomplete: completed={len(completed)}/{len(items)} fatal={fatals}"
        )
    errors = [result for result in results if result.get("error")]
    if errors:
        raise RuntimeError(f"batch items failed: {errors}")
    return results


def main() -> None:
    run_batch(parse_args())


if __name__ == "__main__":
    main()
