#!/usr/bin/env python3
"""Download and validate the minimal RIFT-SVC inference asset set."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODULE_PATTERNS = [
    "content-vec-best/**",
    "rmvpe/**",
    "nsf_hifigan_44.1k_hop512_128bin_2024.02/**",
]
DEFAULT_MODULES_REVISION = "03c1662ba24a76fa3a653c33bc983ce6422620b4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-repo", help="HF model repository")
    parser.add_argument("--model-filename", default="rift25k.ckpt")
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--modules-repo",
        default="Pur1zumu/RIFT-SVC-modules",
        help="HF repository containing the official inference modules",
    )
    parser.add_argument("--modules-revision", default=DEFAULT_MODULES_REVISION)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--modules-only",
        action="store_true",
        help="download only the official ContentVec/RMVPE/HiFi-GAN assets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_repo and not args.modules_only:
        raise SystemExit("--model-repo is required unless --modules-only is used")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = args.output_dir / "model"
    model_path = None
    model_hash = None
    if args.model_repo:
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = Path(
            hf_hub_download(
                repo_id=args.model_repo,
                filename=args.model_filename,
                revision=args.model_revision,
                local_dir=model_dir,
            )
        )
        model_hash = sha256(model_path)
        if args.expected_sha256 and model_hash != args.expected_sha256.lower():
            raise SystemExit(
                f"checkpoint SHA-256 mismatch: expected {args.expected_sha256}, got {model_hash}"
            )

    assets_dir = args.output_dir / "pretrained"
    snapshot_download(
        repo_id=args.modules_repo,
        revision=args.modules_revision,
        local_dir=assets_dir,
        allow_patterns=MODULE_PATTERNS,
    )

    required = [
        assets_dir / "content-vec-best" / "config.json",
        assets_dir / "content-vec-best" / "pytorch_model.bin",
        assets_dir / "rmvpe" / "model.pt",
        assets_dir
        / "nsf_hifigan_44.1k_hop512_128bin_2024.02"
        / "model.ckpt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "inference asset download is incomplete; missing: " + ", ".join(missing)
        )

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_repo": args.model_repo,
        "model_revision": args.model_revision,
        "model_filename": args.model_filename,
        "model_path": str(model_path) if model_path else None,
        "model_sha256": model_hash,
        "modules_repo": args.modules_repo,
        "modules_revision": args.modules_revision,
        "assets_dir": str(assets_dir),
    }
    manifest_path = args.output_dir / "inference-assets.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    if model_path:
        print(f"model: {model_path}")
        print(f"model_sha256: {model_hash}")
    print(f"assets: {assets_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
