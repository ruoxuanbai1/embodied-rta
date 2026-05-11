#!/usr/bin/env python3
"""
Batch-evaluate R3 branch warnings on Case 3 rollout NPZ files.

This script ties together:
1. scene-wise R3 calibration assets,
2. per-rollout branch scoring,
3. event-level warning metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from case3_online_eval import apply_alarm_persistence, compute_event_metrics, danger_sequence_from_npz
from case3_paired_degradation import (
    build_normal_map,
    build_paired_degradation_sequence,
    load_rollout_arrays,
)
from case3_score_r3_branches import build_branch_scores


BRANCH_SCORE_KEYS = {
    "ood": "ood",
    "gradient": "gradient_norm",
    "phi": "phi_norm",
    "routing": "routing",
}

BRANCH_THRESHOLD_KEYS = {
    "ood": "ood_q99",
    "gradient": "gradient_norm_q99",
    "phi": "phi_norm_q99",
    "routing": "routing_q99_proxy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=["ood", "gradient", "phi", "routing"],
        default=["ood", "gradient", "phi", "routing"],
    )
    parser.add_argument(
        "--target-mode",
        choices=["absolute_danger", "paired_degradation"],
        default="paired_degradation",
    )
    parser.add_argument(
        "--danger-mode",
        choices=["any", "hard", "task", "persistent", "hard_or_task"],
        default="hard_or_task",
    )
    parser.add_argument("--lead-window", type=int, default=25)
    parser.add_argument("--safe-grace-window", type=int, default=0)
    parser.add_argument("--persistence", type=int, default=1)
    parser.add_argument("--reward-gap-threshold", type=float, default=1.0)
    parser.add_argument("--degradation-sustain", type=int, default=5)
    parser.add_argument("--splits", nargs="+", default=["test"])
    return parser.parse_args()


def _aggregate_event_results(items: list[dict[str, Any]]) -> dict[str, Any]:
    danger_events = sum(item["metrics"]["tp_events"] + item["metrics"]["fn_events"] for item in items)
    warned_events = sum(item["metrics"]["tp_events"] for item in items)
    safe_events = sum(item["metrics"]["safe_events"] for item in items)
    false_alarms = sum(item["metrics"]["fp_events"] for item in items)
    lead_times: list[int] = []
    for item in items:
        lead_times.extend(item["metrics"]["lead_times"])

    precision = warned_events / (warned_events + false_alarms) if (warned_events + false_alarms) else 0.0
    recall = warned_events / danger_events if danger_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_alarm_rate = false_alarms / safe_events if safe_events else 0.0
    mean_lead_time = float(np.mean(lead_times)) if lead_times else 0.0

    return {
        "n_trials": len(items),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_alarm_rate": false_alarm_rate,
        "mean_lead_time": mean_lead_time,
        "tp_events": warned_events,
        "fp_events": false_alarms,
        "fn_events": max(danger_events - warned_events, 0),
        "safe_events": safe_events,
        "lead_times": lead_times,
    }


def _build_target_sequence(
    run_data: dict[str, Any],
    *,
    target_mode: str,
    danger_mode: str,
    normal_map: dict[tuple[str, int], dict[str, Any]],
    reward_gap_threshold: float,
    degradation_sustain: int,
) -> tuple[list[bool] | None, dict[str, Any]]:
    if target_mode == "absolute_danger":
        danger_data = {
            "danger_any": run_data["danger_any"],
            "danger_hard": run_data["danger_hard"],
            "danger_task": run_data["danger_task"],
            "danger_persistent": run_data["danger_persistent"],
        }
        return danger_sequence_from_npz(danger_data, danger_mode=danger_mode), {
            "target_mode": target_mode,
        }

    if run_data["fault_type"] == "normal":
        return [False] * len(run_data["rewards"]), {
            "target_mode": target_mode,
            "paired_normal_path": run_data["path"],
            "paired_gap_steps": 0,
            "paired_positive_steps": 0,
        }

    degradation_seq, meta = build_paired_degradation_sequence(
        run=run_data,
        normal_map=normal_map,
        reward_gap_threshold=reward_gap_threshold,
        degradation_sustain=degradation_sustain,
    )
    if degradation_seq is None:
        return None, {
            "target_mode": target_mode,
            **meta,
        }
    return degradation_seq.tolist(), {
        "target_mode": target_mode,
        **meta,
    }


def evaluate_one_rollout(
    npz_path: Path,
    *,
    calibration_dir: Path,
    branches: list[str],
    target_mode: str,
    danger_mode: str,
    normal_map: dict[tuple[str, int], dict[str, Any]],
    lead_window: int,
    safe_grace_window: int,
    persistence: int,
    reward_gap_threshold: float,
    degradation_sustain: int,
) -> dict[str, Any]:
    branch_payload = build_branch_scores(npz_path, calibration_dir)
    run_data = _load_rollout_arrays(npz_path)
    danger_seq, target_meta = _build_target_sequence(
        run_data,
        target_mode=target_mode,
        danger_mode=danger_mode,
        normal_map=normal_map,
        reward_gap_threshold=reward_gap_threshold,
        degradation_sustain=degradation_sustain,
    )
    if danger_seq is None:
        return {
            "path": str(npz_path),
            "scene": run_data["scene"],
            "fault_type": run_data["fault_type"],
            "seed": run_data["seed"],
            "steps": run_data["steps"],
            "skipped": True,
            **target_meta,
        }

    branch_results: dict[str, Any] = {}
    for branch in branches:
        score_key = BRANCH_SCORE_KEYS[branch]
        threshold_key = BRANCH_THRESHOLD_KEYS[branch]
        scores = np.asarray(branch_payload["scores"][score_key], dtype=np.float32)
        threshold = float(branch_payload["thresholds"][threshold_key])
        raw_alarm_seq = (scores >= threshold).astype(bool).tolist()
        alarm_seq = apply_alarm_persistence(raw_alarm_seq, min_consecutive=persistence)
        metrics = compute_event_metrics(
            danger_seq,
            alarm_seq,
            lead_window=lead_window,
            safe_grace_window=safe_grace_window,
        )
        branch_results[branch] = {
            "threshold": threshold,
            "alarm_count": int(sum(alarm_seq)),
            "metrics": {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "false_alarm_rate": metrics.false_alarm_rate,
                "mean_lead_time": metrics.mean_lead_time,
                "tp_events": metrics.tp_events,
                "fp_events": metrics.fp_events,
                "fn_events": metrics.fn_events,
                "safe_events": metrics.safe_events,
                "lead_times": metrics.lead_times,
            },
        }

    return {
        "path": str(npz_path),
        "scene": run_data["scene"],
        "fault_type": run_data["fault_type"],
        "seed": run_data["seed"],
        "steps": run_data["steps"],
        "skipped": False,
        **target_meta,
        "branch_results": branch_results,
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    calibration_dir = Path(args.calibration_dir)
    output_json = Path(args.output_json)

    npz_paths = sorted(data_dir.rglob("*.npz"))
    run_index = [_load_rollout_arrays(npz_path) for npz_path in npz_paths]
    run_index = [run for run in run_index if run.get("split") in set(args.splits)]
    npz_paths = [Path(run["path"]) for run in run_index]
    normal_map = build_normal_map(run_index)

    per_trial: list[dict[str, Any]] = []
    skipped_trials: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for npz_path in npz_paths:
        result = evaluate_one_rollout(
            npz_path,
            calibration_dir=calibration_dir,
            branches=list(args.branches),
            target_mode=args.target_mode,
            danger_mode=args.danger_mode,
            normal_map=normal_map,
            lead_window=args.lead_window,
            safe_grace_window=args.safe_grace_window,
            persistence=args.persistence,
            reward_gap_threshold=args.reward_gap_threshold,
            degradation_sustain=args.degradation_sustain,
        )
        per_trial.append(result)
        if result.get("skipped"):
            skipped_trials.append(result)
            continue
        scene = result["scene"]
        fault_type = result["fault_type"]
        for branch, branch_result in result["branch_results"].items():
            grouped[f"{scene}::{fault_type}"][branch].append(branch_result)

    by_scene_fault: dict[str, Any] = {}
    for key, branch_map in grouped.items():
        by_scene_fault[key] = {}
        for branch, items in branch_map.items():
            by_scene_fault[key][branch] = _aggregate_event_results(items)

    by_scene_grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    overall_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in per_trial:
        scene = result["scene"]
        for branch, branch_result in result["branch_results"].items():
            by_scene_grouped[scene][branch].append(branch_result)
            overall_grouped[branch].append(branch_result)

    by_scene: dict[str, Any] = {}
    for scene, branch_map in by_scene_grouped.items():
        by_scene[scene] = {}
        for branch, items in branch_map.items():
            by_scene[scene][branch] = _aggregate_event_results(items)

    overall: dict[str, Any] = {}
    for branch, items in overall_grouped.items():
        overall[branch] = _aggregate_event_results(items)

    payload = {
        "config": {
            "data_dir": str(data_dir),
            "calibration_dir": str(calibration_dir),
            "branches": list(args.branches),
            "target_mode": args.target_mode,
            "danger_mode": args.danger_mode,
            "lead_window": args.lead_window,
            "safe_grace_window": args.safe_grace_window,
            "persistence": args.persistence,
                "reward_gap_threshold": args.reward_gap_threshold,
                "degradation_sustain": args.degradation_sustain,
                "splits": list(args.splits),
            },
        "n_total_trials": len(per_trial),
        "n_evaluated_trials": len(per_trial) - len(skipped_trials),
        "n_skipped_trials": len(skipped_trials),
        "overall": overall,
        "by_scene": by_scene,
        "by_scene_fault": by_scene_fault,
        "skipped_trials": skipped_trials,
        "per_trial": per_trial,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    main()
