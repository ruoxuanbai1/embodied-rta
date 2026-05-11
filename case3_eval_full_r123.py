#!/usr/bin/env python3
"""
Unified Case 3 online evaluation for R1 / R2 / R3 / Full.

Current prototype choices:
- R1 uses rollout risk channels as a baseline hard/task alarm proxy.
- R2 uses the trained GRU over state+action history.
- R3 uses scene-wise calibrated OOD / gradient / phi / routing branches.
- Full uses a simple union over selected layer alarms.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from case3_online_eval import apply_alarm_persistence, compute_event_metrics
from case3_paired_degradation import (
    build_normal_map,
    build_paired_degradation_sequence,
    load_rollout_arrays,
)
from case3_score_r3_branches import build_branch_scores
from case3_train_r2_hazard_gru import HazardGRU


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
    parser.add_argument("--r2-checkpoint", required=True)
    parser.add_argument("--r3-calibration-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--r2-threshold", type=float, default=0.5)
    parser.add_argument("--r3-branches", nargs="+", choices=["ood", "gradient", "phi", "routing"], default=["ood", "gradient", "phi", "routing"])
    parser.add_argument("--lead-window", type=int, default=25)
    parser.add_argument("--safe-grace-window", type=int, default=0)
    parser.add_argument("--persistence", type=int, default=1)
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


def _load_r2_model(ckpt_path: Path, device: torch.device) -> tuple[HazardGRU, dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location=device)
    model = HazardGRU(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        num_layers=int(payload["num_layers"]),
        dropout=float(payload["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _predict_r2_scores(run: dict[str, Any], model: HazardGRU, history_len: int, device: torch.device) -> np.ndarray:
    features = np.concatenate([run["states"], run["actions"]], axis=1).astype(np.float32)
    scores = np.zeros((len(features),), dtype=np.float32)
    if len(features) <= history_len:
        return scores
    windows = []
    end_indices = []
    for end_idx in range(history_len, len(features)):
        windows.append(features[end_idx - history_len : end_idx])
        end_indices.append(end_idx)
    x = torch.from_numpy(np.stack(windows, axis=0)).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).detach().cpu().numpy().astype(np.float32)
    for end_idx, prob in zip(end_indices, probs):
        scores[end_idx] = float(prob)
    return scores


def _r1_alarm_sequence(run: dict[str, Any]) -> list[bool]:
    # Prototype R1: trigger on hard safety or task-critical failure, but not on persistent instability alone.
    return np.logical_or(run["danger_hard"] > 0.5, run["danger_task"] > 0.5).astype(bool).tolist()


def _r3_alarm_sequences(npz_path: Path, calibration_dir: Path, branches: list[str]) -> dict[str, list[bool]]:
    payload = build_branch_scores(npz_path, calibration_dir)
    alarms: dict[str, list[bool]] = {}
    for branch in branches:
        score_key = BRANCH_SCORE_KEYS[branch]
        threshold_key = BRANCH_THRESHOLD_KEYS[branch]
        scores = np.asarray(payload["scores"][score_key], dtype=np.float32)
        threshold = float(payload["thresholds"][threshold_key])
        alarms[branch] = (scores >= threshold).astype(bool).tolist()
    return alarms


def _evaluate_alarm(danger_seq: list[bool], alarm_seq: list[bool], lead_window: int, safe_grace_window: int) -> dict[str, Any]:
    metrics = compute_event_metrics(
        danger_seq,
        alarm_seq,
        lead_window=lead_window,
        safe_grace_window=safe_grace_window,
    )
    return {
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
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    r2_ckpt = Path(args.r2_checkpoint)
    r3_cal = Path(args.r3_calibration_dir)
    output_json = Path(args.output_json)
    device = torch.device(args.device)

    model, r2_payload = _load_r2_model(r2_ckpt, device)
    npz_paths = sorted(data_dir.rglob("*.npz"))
    runs = [load_rollout_arrays(path) for path in npz_paths]
    runs = [run for run in runs if run.get("split") in set(args.splits)]
    npz_paths = [Path(run["path"]) for run in runs]
    path_to_run = {run["path"]: run for run in runs}
    normal_map = build_normal_map(runs)

    per_trial: list[dict[str, Any]] = []
    skipped_trials: list[dict[str, Any]] = []
    grouped_overall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_by_scene: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    grouped_by_scene_fault: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for npz_path in npz_paths:
        run = path_to_run[str(npz_path)]
        degradation_seq, meta = build_paired_degradation_sequence(
            run=run,
            normal_map=normal_map,
            reward_gap_threshold=float(r2_payload.get("reward_gap_threshold", 1.0)),
            degradation_sustain=int(r2_payload.get("degradation_sustain", 5)),
        )
        if degradation_seq is None:
            skipped_trials.append({
                "path": str(npz_path),
                "scene": run["scene"],
                "fault_type": run["fault_type"],
                "seed": run["seed"],
                "steps": run["steps"],
                "skipped": True,
                **meta,
            })
            continue

        danger_seq = degradation_seq.astype(bool).tolist()
        r1_alarm = apply_alarm_persistence(_r1_alarm_sequence(run), min_consecutive=args.persistence)

        r2_scores = _predict_r2_scores(run, model, int(r2_payload["history_len"]), device)
        r2_alarm = apply_alarm_persistence((r2_scores >= args.r2_threshold).astype(bool).tolist(), min_consecutive=args.persistence)

        try:
            r3_branch_raw = _r3_alarm_sequences(npz_path, r3_cal, list(args.r3_branches))
        except FileNotFoundError as exc:
            skipped_trials.append({
                "path": str(npz_path),
                "scene": run["scene"],
                "fault_type": run["fault_type"],
                "seed": run["seed"],
                "steps": run["steps"],
                "skipped": True,
                "skip_reason": "missing_r3_calibration",
                "skip_detail": str(exc),
                **meta,
            })
            continue
        r3_branch_alarm = {
            branch: apply_alarm_persistence(seq, min_consecutive=args.persistence)
            for branch, seq in r3_branch_raw.items()
        }
        r3_union = [any(r3_branch_alarm[branch][idx] for branch in args.r3_branches) for idx in range(run["steps"])]
        full_union = [
            r1_alarm[idx] or r2_alarm[idx] or r3_union[idx]
            for idx in range(run["steps"])
        ]

        layer_metrics = {
            "R1": _evaluate_alarm(danger_seq, r1_alarm, args.lead_window, args.safe_grace_window),
            "R2": _evaluate_alarm(danger_seq, r2_alarm, args.lead_window, args.safe_grace_window),
            "R3": _evaluate_alarm(danger_seq, r3_union, args.lead_window, args.safe_grace_window),
            "Full": _evaluate_alarm(danger_seq, full_union, args.lead_window, args.safe_grace_window),
        }

        trial_result = {
            "path": str(npz_path),
            "scene": run["scene"],
            "fault_type": run["fault_type"],
            "seed": run["seed"],
            "steps": run["steps"],
            "skipped": False,
            **meta,
            "layer_metrics": layer_metrics,
        }
        per_trial.append(trial_result)
        scene = run["scene"]
        scene_fault = f"{scene}::{run['fault_type']}"
        for layer_name, metrics in layer_metrics.items():
            item = {"metrics": metrics}
            grouped_overall[layer_name].append(item)
            grouped_by_scene[scene][layer_name].append(item)
            grouped_by_scene_fault[scene_fault][layer_name].append(item)

    overall = {layer: _aggregate_event_results(items) for layer, items in grouped_overall.items()}
    by_scene = {
        scene: {layer: _aggregate_event_results(items) for layer, items in layer_map.items()}
        for scene, layer_map in grouped_by_scene.items()
    }
    by_scene_fault = {
        key: {layer: _aggregate_event_results(items) for layer, items in layer_map.items()}
        for key, layer_map in grouped_by_scene_fault.items()
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "config": {
                    "data_dir": str(data_dir),
                    "r2_checkpoint": str(r2_ckpt),
                    "r3_calibration_dir": str(r3_cal),
                    "r2_threshold": args.r2_threshold,
                    "r3_branches": list(args.r3_branches),
                    "lead_window": args.lead_window,
                "safe_grace_window": args.safe_grace_window,
                "persistence": args.persistence,
                "splits": list(args.splits),
            },
                "n_total_trials": len(npz_paths),
                "n_evaluated_trials": len(per_trial),
                "n_skipped_trials": len(skipped_trials),
                "overall": overall,
                "by_scene": by_scene,
                "by_scene_fault": by_scene_fault,
                "skipped_trials": skipped_trials,
                "per_trial": per_trial,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_json)


if __name__ == "__main__":
    main()
