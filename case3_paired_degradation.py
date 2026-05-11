#!/usr/bin/env python3
"""
Shared helpers for Case 3 paired-degradation targets.

These helpers keep the "fault rollout relative to its matched normal rollout"
target definition consistent across:
- R2 training labels
- R2 online evaluation
- R3 branch evaluation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from case3_scene_registry import default_split_for_seed


def load_rollout_arrays(npz_path: Path) -> dict[str, Any]:
    data = np.load(npz_path)
    split = str(data["split"].item()) if "split" in data else default_split_for_seed(int(data["seed"][0]))
    payload: dict[str, Any] = {
        "path": str(npz_path),
        "scene": str(data["scene"].item()),
        "fault_type": str(data["fault_type"].item()),
        "seed": int(data["seed"][0]),
        "split": split,
        "steps": int(data["steps"][0]),
        "states": data["states"].astype(np.float32),
        "actions": data["actions"].astype(np.float32),
        "rewards": data["rewards"].astype(np.float32),
        "danger_any": data["danger_any"].astype(np.float32),
        "danger_hard": data["danger_hard"].astype(np.float32),
        "danger_task": data["danger_task"].astype(np.float32),
        "danger_persistent": data["danger_persistent"].astype(np.float32),
    }
    for key in ("gradients", "phis", "gradient_mask", "route_src", "route_dst"):
        if key in data:
            value = data[key]
            if key == "gradient_mask":
                payload[key] = value.astype(np.uint8)
            else:
                payload[key] = value.astype(np.float32)
    return payload


def build_normal_map(runs: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (run["scene"], run["seed"]): run
        for run in runs
        if run["fault_type"] == "normal"
    }


def combined_risk(run_data: dict[str, Any]) -> np.ndarray:
    return (
        (run_data["danger_hard"] > 0.5)
        | (run_data["danger_task"] > 0.5)
        | (run_data["danger_persistent"] > 0.5)
    )


def keep_only_sustained_runs(raw_signal: np.ndarray, min_run_length: int) -> np.ndarray:
    if min_run_length <= 1:
        return raw_signal.astype(bool)

    raw_bool = raw_signal.astype(bool)
    kept = np.zeros_like(raw_bool, dtype=bool)
    run_start: int | None = None
    for idx, value in enumerate(raw_bool):
        if value and run_start is None:
            run_start = idx
        elif (not value) and run_start is not None:
            if idx - run_start >= min_run_length:
                kept[run_start:idx] = True
            run_start = None
    if run_start is not None and len(raw_bool) - run_start >= min_run_length:
        kept[run_start:] = True
    return kept


def onset_from_binary_signal(signal: np.ndarray, onset_lookback: int) -> np.ndarray:
    onset = np.zeros_like(signal, dtype=np.float32)
    last_positive_end = -10**9
    for idx, current in enumerate(signal):
        if current <= 0:
            continue
        prev_start = max(0, idx - onset_lookback)
        prev_active = np.any(signal[prev_start:idx] > 0)
        if not prev_active and idx > last_positive_end:
            onset[idx] = 1.0
        if idx + 1 >= len(signal) or signal[idx + 1] <= 0:
            last_positive_end = idx
    return onset


def build_paired_degradation_sequence(
    *,
    run: dict[str, Any],
    normal_map: dict[tuple[str, int], dict[str, Any]],
    reward_gap_threshold: float,
    degradation_sustain: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if run["fault_type"] == "normal":
        return np.zeros_like(run["rewards"], dtype=bool), {
            "paired_normal_path": run["path"],
            "paired_gap_steps": 0,
            "paired_positive_steps": 0,
        }

    baseline = normal_map.get((run["scene"], run["seed"]))
    if baseline is None:
        return None, {
            "skip_reason": "missing_paired_normal",
        }

    steps = min(len(run["rewards"]), len(baseline["rewards"]))
    reward_gap = baseline["rewards"][:steps] - run["rewards"][:steps]
    run_risk = combined_risk(run)[:steps]
    baseline_risk = combined_risk(baseline)[:steps]
    raw_degradation = (reward_gap >= reward_gap_threshold) | (run_risk & ~baseline_risk)
    degradation = keep_only_sustained_runs(raw_degradation, degradation_sustain)

    target = np.zeros((len(run["rewards"]),), dtype=bool)
    target[:steps] = degradation
    return target, {
        "paired_normal_path": baseline["path"],
        "paired_gap_steps": int(np.sum(reward_gap >= reward_gap_threshold)),
        "paired_positive_steps": int(np.sum(degradation)),
    }


def filter_runs_by_split(runs: list[dict[str, Any]], splits: list[str]) -> list[dict[str, Any]]:
    allowed = {s.strip() for s in splits if s.strip()}
    if not allowed:
        return runs
    return [run for run in runs if run.get("split", default_split_for_seed(int(run["seed"]))) in allowed]
