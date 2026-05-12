#!/usr/bin/env python3
"""
Wait for the split-aware perception block to finish, then continue the
remaining all-8 collection blocks and launch the full train/test pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from case3_run_all8_collection import scene_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-test-output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-id", default="lerobot/act_aloha_sim_transfer_cube_human")
    parser.add_argument("--legacy-source-root", default="/home/vipuser/lerobot-3c0a209f9fac4d2a57617e686a7f2a2309144ba2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--fault-timings", nargs="+", default=["early", "mid", "late"])
    parser.add_argument("--intensity-bands", nargs="+", default=["mild", "medium", "severe"])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gradient-every", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--remaining-blocks", nargs="+", default=["remaining_all"])
    parser.add_argument("--schedule", choices=["sequential", "round_robin"], default="round_robin")
    parser.add_argument("--group-by", choices=["scene", "axis"], default="axis")
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


def read_manifest_rows(data_dir: Path) -> list[dict]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected manifest JSON list: {manifest_path}")
    return data


def count_by_scene(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        scene = str(row.get("scene", "missing"))
        counts[scene] = counts.get(scene, 0) + 1
    return counts


def expected_rows_per_scene(*, faults: list[str], seeds: list[int], fault_timings: list[str], intensity_bands: list[str]) -> int:
    normal_rows = len(seeds) if "normal" in faults else 0
    fault_types = [fault for fault in faults if fault != "normal"]
    return normal_rows + len(fault_types) * len(seeds) * len(fault_timings) * len(intensity_bands)


def perception_ready(rows: list[dict], *, seeds: list[int], fault_timings: list[str], intensity_bands: list[str]) -> tuple[bool, dict[str, int], dict[str, int]]:
    blocks = scene_blocks()
    perception = blocks["perception"]
    expected = expected_rows_per_scene(
        faults=perception["faults"],
        seeds=seeds,
        fault_timings=fault_timings,
        intensity_bands=intensity_bands,
    )
    targets = {scene: expected for scene in perception["scenes"]}
    by_scene = count_by_scene(rows)
    ready = all(by_scene.get(scene, 0) >= required for scene, required in targets.items())
    return ready, by_scene, targets


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_collection_cmd(args: argparse.Namespace, *, blocks_to_run: list[str]) -> list[str]:
    root = Path(__file__).resolve().parent
    return [
        sys.executable,
        str(root / "case3_run_all8_collection.py"),
        "--output-dir",
        args.data_dir,
        "--device",
        args.device,
        "--model-id",
        args.model_id,
        "--legacy-source-root",
        args.legacy_source_root,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--fault-timings",
        *args.fault_timings,
        "--intensity-bands",
        *args.intensity_bands,
        "--max-steps",
        str(args.max_steps),
        "--gradient-every",
        str(args.gradient_every),
        "--schedule",
        args.schedule,
        "--group-by",
        args.group_by,
        "--blocks",
        *blocks_to_run,
    ]


def build_train_test_cmd(args: argparse.Namespace) -> list[str]:
    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(root / "case3_run_all8_train_test.py"),
        "--data-dir",
        args.data_dir,
        "--output-root",
        args.train_test_output_root,
        "--device",
        args.device,
        "--r2-epochs",
        str(args.r2_epochs),
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
        "--train-splits",
        *args.train_splits,
        "--val-splits",
        *args.val_splits,
        "--test-splits",
        *args.test_splits,
    ]
    if args.history_json:
        cmd.extend(["--history-json", args.history_json])
    return cmd


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    while True:
        rows = read_manifest_rows(data_dir)
        ready, by_scene, targets = perception_ready(
            rows,
            seeds=args.seeds,
            fault_timings=args.fault_timings,
            intensity_bands=args.intensity_bands,
        )
        status = {
            "phase": "wait_for_perception",
            "n_rows": len(rows),
            "by_scene": by_scene,
            "targets": targets,
            "ready": ready,
        }
        print(json.dumps(status), flush=True)
        if ready:
            break
        time.sleep(args.poll_seconds)

    if args.remaining_blocks:
        run(build_collection_cmd(args, blocks_to_run=args.remaining_blocks))

    run(build_train_test_cmd(args))


if __name__ == "__main__":
    main()
