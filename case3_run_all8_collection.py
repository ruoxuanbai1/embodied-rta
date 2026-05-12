#!/usr/bin/env python3
"""
Run the current 8-scene executable Case 3 collection protocol in logical blocks.

Why this entrypoint exists:
1. The 8 executable scenes do not all share the same relevant fault families.
2. We want one canonical place that expands the benchmark design into concrete
   collector invocations.
3. We want to append into one dataset directory without overwriting finished
   trials, so interrupted runs can resume safely.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from case3_scene_registry import supported_scene_variants_by_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collector-script", default=str(Path(__file__).with_name("case3_collect_r23_dataset.py")))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-id", default="lerobot/act_aloha_sim_transfer_cube_human")
    parser.add_argument("--legacy-source-root", default="/home/vipuser/lerobot-3c0a209f9fac4d2a57617e686a7f2a2309144ba2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--fault-timings", nargs="+", default=["early", "mid", "late"])
    parser.add_argument("--intensity-bands", nargs="+", default=["mild", "medium", "severe"])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gradient-every", type=int, default=20)
    parser.add_argument("--blocks", nargs="+", default=["perception", "dynamics", "mixed"])
    parser.add_argument("--schedule", choices=["sequential", "round_robin"], default="round_robin")
    parser.add_argument("--group-by", choices=["scene", "axis"], default="axis")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def scene_blocks() -> dict[str, dict[str, list[str]]]:
    grouped = supported_scene_variants_by_axis()
    return {
        "perception": {
            "scenes": ["nominal_clear_reference"] + grouped.get("perception", []),
            "faults": ["normal", "F1_lighting", "F2_occlusion", "F3_adversarial"],
        },
        "dynamics": {
            "scenes": grouped.get("dynamics", []) + grouped.get("timing_shift", []),
            "faults": ["normal", "F4_payload", "F5_friction", "F6_dynamic", "F8_compound"],
        },
        "mixed": {
            "scenes": grouped.get("mixed", []),
            "faults": ["normal", "F3_adversarial", "F6_dynamic", "F8_compound"],
        },
    }


def build_collector_cmd(args: argparse.Namespace, *, scenes: list[str], faults: list[str]) -> list[str]:
    return [
        sys.executable,
        args.collector_script,
        "--model-id",
        args.model_id,
        "--legacy-source-root",
        args.legacy_source_root,
        "--scenes",
        *scenes,
        "--faults",
        *faults,
        "--fault-timings",
        *args.fault_timings,
        "--intensity-bands",
        *args.intensity_bands,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--max-steps",
        str(args.max_steps),
        "--gradient-every",
        str(args.gradient_every),
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--schedule",
        args.schedule,
        "--group-by",
        args.group_by,
        "--append",
        "--skip-existing",
    ]


def main() -> None:
    args = parse_args()
    blocks = scene_blocks()

    selected = []
    for block_name in args.blocks:
        if block_name not in blocks:
            raise KeyError(f"Unknown block: {block_name}")
        selected.append((block_name, blocks[block_name]))

    for block_name, spec in selected:
        if not spec["scenes"]:
            print(f"[skip-empty-block] {block_name}")
            continue
        cmd = build_collector_cmd(args, scenes=spec["scenes"], faults=spec["faults"])
        print(f"[run-block] {block_name}")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
