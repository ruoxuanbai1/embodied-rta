#!/usr/bin/env python3
"""
Batch-evaluate a trained Case 3 R2 GRU on rollout NPZ files.

This uses the same paired-degradation target family as:
- case3_train_r2_hazard_gru.py
- case3_batch_eval_r3.py
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
    filter_runs_by_split,
    load_rollout_arrays,
)
from case3_train_r2_hazard_gru import HazardGRU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alarm-threshold", type=float, default=0.5)
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


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[HazardGRU, dict[str, Any]]:
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


def _predict_scores(
    *,
    run: dict[str, Any],
    model: HazardGRU,
    history_len: int,
    device: torch.device,
) -> np.ndarray:
    features = np.concatenate([run["states"], run["actions"]], axis=1).astype(np.float32)
    scores = np.zeros((len(features),), dtype=np.float32)
    if len(features) <= history_len:
        return scores

    windows = []
    end_indices = []
    for end_idx in range(history_len, len(features)):
        start_idx = end_idx - history_len
        windows.append(features[start_idx:end_idx])
        end_indices.append(end_idx)
    x = torch.from_numpy(np.stack(windows, axis=0)).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    for end_idx, prob in zip(end_indices, probs):
        scores[end_idx] = float(prob)
    return scores


def _evaluate_one_run(
    *,
    run: dict[str, Any],
    normal_map: dict[tuple[str, int], dict[str, Any]],
    model: HazardGRU,
    payload: dict[str, Any],
    device: torch.device,
    alarm_threshold: float,
    lead_window: int,
    safe_grace_window: int,
    persistence: int,
) -> dict[str, Any]:
    degradation_seq, meta = build_paired_degradation_sequence(
        run=run,
        normal_map=normal_map,
        reward_gap_threshold=float(payload.get("reward_gap_threshold", 1.0)),
        degradation_sustain=int(payload.get("degradation_sustain", 5)),
    )
    if degradation_seq is None:
        return {
            "path": run["path"],
            "scene": run["scene"],
            "fault_type": run["fault_type"],
            "seed": run["seed"],
            "steps": run["steps"],
            "skipped": True,
            **meta,
        }

    scores = _predict_scores(
        run=run,
        model=model,
        history_len=int(payload["history_len"]),
        device=device,
    )
    raw_alarm_seq = (scores >= alarm_threshold).astype(bool).tolist()
    alarm_seq = apply_alarm_persistence(raw_alarm_seq, min_consecutive=persistence)
    metrics = compute_event_metrics(
        degradation_seq.astype(bool).tolist(),
        alarm_seq,
        lead_window=lead_window,
        safe_grace_window=safe_grace_window,
    )
    return {
        "path": run["path"],
        "scene": run["scene"],
        "fault_type": run["fault_type"],
        "seed": run["seed"],
        "steps": run["steps"],
        "skipped": False,
        "alarm_threshold": alarm_threshold,
        "score_nonzero_steps": int(np.sum(scores > 0)),
        **meta,
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


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.checkpoint)
    output_json = Path(args.output_json)
    device = torch.device(args.device)

    model, payload = _load_model(ckpt_path, device)
    runs = filter_runs_by_split([load_rollout_arrays(path) for path in sorted(data_dir.rglob("*.npz"))], args.splits)
    normal_map = build_normal_map(runs)

    per_trial: list[dict[str, Any]] = []
    skipped_trials: list[dict[str, Any]] = []
    grouped_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_by_scene_fault: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for run in runs:
        result = _evaluate_one_run(
            run=run,
            normal_map=normal_map,
            model=model,
            payload=payload,
            device=device,
            alarm_threshold=args.alarm_threshold,
            lead_window=args.lead_window,
            safe_grace_window=args.safe_grace_window,
            persistence=args.persistence,
        )
        per_trial.append(result)
        if result.get("skipped"):
            skipped_trials.append(result)
            continue
        grouped_by_scene[result["scene"]].append(result)
        grouped_by_scene_fault[f"{result['scene']}::{result['fault_type']}"].append(result)

    evaluated = [item for item in per_trial if not item.get("skipped")]
    overall = _aggregate_event_results(evaluated) if evaluated else {}
    by_scene = {scene: _aggregate_event_results(items) for scene, items in grouped_by_scene.items()}
    by_scene_fault = {
        key: _aggregate_event_results(items) for key, items in grouped_by_scene_fault.items()
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "config": {
                    "data_dir": str(data_dir),
                    "checkpoint": str(ckpt_path),
                    "alarm_threshold": args.alarm_threshold,
                    "lead_window": args.lead_window,
                    "safe_grace_window": args.safe_grace_window,
                    "persistence": args.persistence,
                    "history_len": int(payload["history_len"]),
                    "future_horizon": int(payload["future_horizon"]),
                    "label_mode": payload.get("label_mode", "paired_degradation_onset"),
                    "reward_gap_threshold": float(payload.get("reward_gap_threshold", 1.0)),
                    "degradation_sustain": int(payload.get("degradation_sustain", 5)),
                    "splits": list(args.splits),
                },
                "n_total_trials": len(per_trial),
                "n_evaluated_trials": len(evaluated),
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
