#!/usr/bin/env python3
"""Fit the configured train-only core20 normalization artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import training


def run(config_path: Path) -> dict[str, object]:
    config = training.load_resolved_config(config_path)
    manifest_path = Path(config["data"]["manifest"])
    output_path = Path(config["data"]["normalization"])
    payload = training.fit_normalization(
        manifest_path,
        config["data"]["train_scenes"],
        output_path,
        epsilon=config["normalization"]["epsilon"],
    )
    return {
        "status": "generated",
        "normalization_version": payload["normalization_version"],
        "fit_role": payload["fit_role"],
        "train_scene_count": len(payload["train_scenes"]),
        "train_sample_count": len(payload["train_sample_ids"]),
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "normalization_path": str(output_path),
        "normalization_sha256": training.sha256_file(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=training.REPO_ROOT / "config.json",
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
