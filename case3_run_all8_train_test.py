#!/usr/bin/env python3
"""
Run the full 8-scene Case 3 training and online testing pipeline.

This is the post-collection entrypoint:
1. train the R2 hazard GRU on the collected dataset
2. calibrate the R3 branches on the same dataset
3. run R2 online evaluation
4. run R3 paired batch evaluation
5. run the unified Full R123 evaluation

The script is intentionally sequential so the output directory becomes a
single, reproducible run record.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--r2-epochs", type=int, default=4)
    parser.add_argument("--r2-threshold", type=float, default=0.5)
    parser.add_argument("--lead-window", type=int, default=25)
    parser.add_argument("--safe-grace-window", type=int, default=0)
    parser.add_argument("--persistence", type=int, default=1)
    parser.add_argument(
        "--r3-branches",
        nargs="+",
        choices=["ood", "gradient", "phi", "routing"],
        default=["ood", "gradient", "phi", "routing"],
    )
    parser.add_argument("--train-splits", nargs="+", default=["train"])
    parser.add_argument("--val-splits", nargs="+", default=["val"])
    parser.add_argument("--test-splits", nargs="+", default=["test"])
    parser.add_argument("--history-json", default=None)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    r2_dir = out_root / "r2"
    r3_dir = out_root / "r3"
    full_dir = out_root / "full"
    r2_dir.mkdir(parents=True, exist_ok=True)
    r3_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable,
        "-m",
        "py_compile",
        str(root / "case3_paired_degradation.py"),
        str(root / "case3_train_r2_hazard_gru.py"),
        str(root / "case3_eval_r2_online.py"),
        str(root / "case3_calibrate_r3_branches.py"),
        str(root / "case3_batch_eval_r3.py"),
        str(root / "case3_eval_full_r123.py"),
    ])

    run([
        sys.executable,
        str(root / "case3_train_r2_hazard_gru.py"),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(r2_dir),
        "--splits",
        *args.train_splits,
        "--epochs",
        str(args.r2_epochs),
        "--device",
        args.device,
    ])

    run([
        sys.executable,
        str(root / "case3_eval_r2_online.py"),
        "--data-dir",
        str(data_dir),
        "--checkpoint",
        str(r2_dir / "r2_hazard_gru.pt"),
        "--output-json",
        str(r2_dir / "r2_online_eval.json"),
        "--splits",
        *args.test_splits,
        "--device",
        args.device,
    ])

    run([
        sys.executable,
        str(root / "case3_calibrate_r3_branches.py"),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(r3_dir),
        "--splits",
        *args.train_splits,
    ])

    run([
        sys.executable,
        str(root / "case3_batch_eval_r3.py"),
        "--data-dir",
        str(data_dir),
        "--calibration-dir",
        str(r3_dir),
        "--output-json",
        str(r3_dir / "r3_batch_eval_paired.json"),
        "--splits",
        *args.test_splits,
    ])

    run([
        sys.executable,
        str(root / "case3_eval_full_r123.py"),
        "--data-dir",
        str(data_dir),
        "--r2-checkpoint",
        str(r2_dir / "r2_hazard_gru.pt"),
        "--r3-calibration-dir",
        str(r3_dir),
        "--output-json",
        str(full_dir / "full_r123_online_eval.json"),
        "--splits",
        *args.test_splits,
        "--device",
        args.device,
        "--r2-threshold",
        str(args.r2_threshold),
        "--lead-window",
        str(args.lead_window),
        "--safe-grace-window",
        str(args.safe_grace_window),
        "--persistence",
        str(args.persistence),
        "--r3-branches",
        *args.r3_branches,
    ])

    history_json = Path(args.history_json) if args.history_json else (out_root / "metrics_history.json")
    full_eval_path = full_dir / "full_r123_online_eval.json"
    if full_eval_path.exists():
        history = []
        if history_json.exists():
            history = json.loads(history_json.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                raise RuntimeError(f"History file must contain a JSON list: {history_json}")
        full_eval = json.loads(full_eval_path.read_text(encoding="utf-8"))
        history.append(
            {
                "source": str(full_eval_path),
                "data_dir": str(data_dir),
                "overall": full_eval["overall"],
                "by_scene": full_eval["by_scene"],
                "by_scene_fault": full_eval["by_scene_fault"],
                "config": full_eval["config"],
            }
        )
        history_json.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(
        str(out_root / "done.json"),
        flush=True,
    )


if __name__ == "__main__":
    main()
