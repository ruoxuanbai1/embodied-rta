#!/usr/bin/env python3
"""
Calibrate compact Region-3 branch statistics from canonical real-env rollouts.

This script does not claim to restore the final manuscript-grade R3. It
rebuilds the essential calibration assets needed to continue development:
1. OOD state reference statistics
2. Gradient / phi magnitude reference statistics
3. Routing pooled-feature reference statistics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from case3_paired_degradation import filter_runs_by_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train"])
    return parser.parse_args()


def _safe_quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "q95": 0.0, "q99": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
    }


def load_runs(data_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(data_dir.rglob("*.npz")):
        data = np.load(path)
        scene = str(data["scene"].item())
        fault_type = str(data["fault_type"].item())
        runs.append(
            {
                "path": str(path),
                "scene": scene,
                "fault_type": fault_type,
                "states": data["states"].astype(np.float32),
                "gradients": data["gradients"].astype(np.float32),
                "phis": data["phis"].astype(np.float32),
                "gradient_mask": data["gradient_mask"].astype(np.uint8),
                "route_src": data["route_src"].astype(np.float32),
                "route_dst": data["route_dst"].astype(np.float32),
            }
        )
    return runs


def fit_scene_stats(runs: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    normals_by_scene: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        if run["fault_type"] != "normal":
            continue
        normals_by_scene.setdefault(run["scene"], []).append(run)

    summary: dict[str, Any] = {"scenes": {}}
    for scene, scene_runs in normals_by_scene.items():
        states = np.concatenate([r["states"] for r in scene_runs], axis=0)
        gradients = np.concatenate([r["gradients"] for r in scene_runs], axis=0)
        phis = np.concatenate([r["phis"] for r in scene_runs], axis=0)
        gradient_mask = np.concatenate([r["gradient_mask"] for r in scene_runs], axis=0) > 0
        route_src = np.concatenate([r["route_src"] for r in scene_runs], axis=0)
        route_dst = np.concatenate([r["route_dst"] for r in scene_runs], axis=0)

        state_mean = states.mean(axis=0)
        state_cov = np.cov(states.T)
        state_cov += np.eye(state_cov.shape[0], dtype=np.float32) * 1e-5
        state_cov_inv = np.linalg.inv(state_cov)

        grad_norm = np.linalg.norm(gradients[gradient_mask], axis=1) if np.any(gradient_mask) else np.zeros((0,), dtype=np.float32)
        phi_norm = np.linalg.norm(phis[gradient_mask], axis=1) if np.any(gradient_mask) else np.zeros((0,), dtype=np.float32)

        route_src_mean = route_src.mean(axis=0) if route_src.size else np.zeros((0,), dtype=np.float32)
        route_dst_mean = route_dst.mean(axis=0) if route_dst.size else np.zeros((0,), dtype=np.float32)
        route_src_dev = np.linalg.norm(route_src - route_src_mean[None, :], axis=1) if route_src.size else np.zeros((0,), dtype=np.float32)
        route_dst_dev = np.linalg.norm(route_dst - route_dst_mean[None, :], axis=1) if route_dst.size else np.zeros((0,), dtype=np.float32)

        np.savez_compressed(
            output_dir / f"r3_scene_stats__{scene}.npz",
            state_mean=state_mean.astype(np.float32),
            state_cov_inv=state_cov_inv.astype(np.float32),
            route_src_mean=route_src_mean.astype(np.float32),
            route_dst_mean=route_dst_mean.astype(np.float32),
        )

        summary["scenes"][scene] = {
            "n_normal_runs": len(scene_runs),
            "n_normal_steps": int(states.shape[0]),
            "ood_state_stats": _safe_quantiles(
                np.sqrt(np.sum(((states - state_mean) @ state_cov_inv) * (states - state_mean), axis=1))
            ),
            "gradient_norm_stats": _safe_quantiles(grad_norm),
            "phi_norm_stats": _safe_quantiles(phi_norm),
            "route_src_dev_stats": _safe_quantiles(route_src_dev),
            "route_dst_dev_stats": _safe_quantiles(route_dst_dev),
        }
    return summary


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = filter_runs_by_split(load_runs(data_dir), args.splits)
    summary = fit_scene_stats(runs, output_dir)
    (output_dir / "r3_branch_calibration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(output_dir / "r3_branch_calibration_summary.json")


if __name__ == "__main__":
    main()
