#!/usr/bin/env python3
"""
Score Case 3 R3 branch signals from collected NPZ rollouts.

Inputs:
- rollout NPZ produced by `case3_collect_r23_dataset.py`
- scene-wise calibration assets produced by `case3_calibrate_r3_branches.py`

Outputs:
- per-step OOD / gradient / routing scores
- simple thresholded alarms for each branch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ood-threshold-q", type=float, default=0.99)
    parser.add_argument("--grad-threshold-q", type=float, default=0.99)
    parser.add_argument("--route-threshold-q", type=float, default=0.99)
    return parser.parse_args()


def _safe_quantile(summary: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = summary.get(key)
    return float(default if value is None else value)


def load_scene_calibration(calibration_dir: Path, scene: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary_path = calibration_dir / "r3_branch_calibration_summary.json"
    scene_npz = calibration_dir / f"r3_scene_stats__{scene}.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing calibration summary: {summary_path}")
    if not scene_npz.exists():
        raise FileNotFoundError(f"Missing scene calibration NPZ: {scene_npz}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if scene not in summary.get("scenes", {}):
        raise KeyError(f"Scene {scene} missing from calibration summary")
    raw = np.load(scene_npz)
    arrays = {key: raw[key].astype(np.float32) for key in raw.files}
    return summary["scenes"][scene], arrays


def mahalanobis_scores(states: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diff = states - mean[None, :]
    return np.sqrt(np.sum(((diff @ cov_inv) * diff), axis=1)).astype(np.float32)


def l2_dev_scores(values: np.ndarray, mean: np.ndarray) -> np.ndarray:
    if values.size == 0 or mean.size == 0:
        return np.zeros((values.shape[0],), dtype=np.float32)
    return np.linalg.norm(values - mean[None, :], axis=1).astype(np.float32)


def build_branch_scores(npz_path: Path, calibration_dir: Path) -> dict[str, Any]:
    data = np.load(npz_path)
    scene = str(data["scene"].item())
    scene_summary, arrays = load_scene_calibration(calibration_dir, scene)

    states = data["states"].astype(np.float32)
    gradients = data["gradients"].astype(np.float32)
    phis = data["phis"].astype(np.float32)
    route_src = data["route_src"].astype(np.float32)
    route_dst = data["route_dst"].astype(np.float32)

    ood_scores = mahalanobis_scores(states, arrays["state_mean"], arrays["state_cov_inv"])
    grad_scores = np.linalg.norm(gradients, axis=1).astype(np.float32)
    phi_scores = np.linalg.norm(phis, axis=1).astype(np.float32)
    route_src_scores = l2_dev_scores(route_src, arrays["route_src_mean"])
    route_dst_scores = l2_dev_scores(route_dst, arrays["route_dst_mean"])
    route_scores = 0.5 * (route_src_scores + route_dst_scores)

    ood_thr = _safe_quantile(scene_summary["ood_state_stats"], "q99")
    grad_thr = _safe_quantile(scene_summary["gradient_norm_stats"], "q99")
    phi_thr = _safe_quantile(scene_summary["phi_norm_stats"], "q99")
    route_thr = 0.5 * (
        _safe_quantile(scene_summary["route_src_dev_stats"], "q99")
        + _safe_quantile(scene_summary["route_dst_dev_stats"], "q99")
    )

    result = {
        "scene": scene,
        "fault_type": str(data["fault_type"].item()),
        "seed": int(data["seed"][0]),
        "steps": int(data["steps"][0]),
        "scores": {
            "ood": ood_scores.tolist(),
            "gradient_norm": grad_scores.tolist(),
            "phi_norm": phi_scores.tolist(),
            "routing": route_scores.tolist(),
            "route_src_dev": route_src_scores.tolist(),
            "route_dst_dev": route_dst_scores.tolist(),
        },
        "thresholds": {
            "ood_q99": ood_thr,
            "gradient_norm_q99": grad_thr,
            "phi_norm_q99": phi_thr,
            "routing_q99_proxy": route_thr,
        },
        "alarms": {
            "ood": (ood_scores >= ood_thr).astype(int).tolist(),
            "gradient": (grad_scores >= grad_thr).astype(int).tolist(),
            "phi": (phi_scores >= phi_thr).astype(int).tolist(),
            "routing": (route_scores >= route_thr).astype(int).tolist(),
        },
    }
    return result


def main() -> None:
    args = parse_args()
    result = build_branch_scores(Path(args.npz), Path(args.calibration_dir))
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
