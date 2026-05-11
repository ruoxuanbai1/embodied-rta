#!/usr/bin/env python3
"""
Case 3 online warning evaluation helpers.

Purpose:
1. Provide a single event-level evaluator for R1 / R2 / R3 / fused methods.
2. Keep metric definitions explicit before full online testing is wired in.
3. Reuse the same outputs for offline replay and future real online rollouts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EventMetrics:
    precision: float
    recall: float
    f1_score: float
    false_alarm_rate: float
    mean_lead_time: float
    tp_events: int
    fp_events: int
    fn_events: int
    safe_events: int
    lead_times: list[int]


def apply_alarm_persistence(alarm_seq: list[bool], min_consecutive: int = 1) -> list[bool]:
    if min_consecutive <= 1 or not alarm_seq:
        return list(alarm_seq)

    persisted = [False] * len(alarm_seq)
    run_start: int | None = None
    for idx, alarm in enumerate(alarm_seq):
        if alarm and run_start is None:
            run_start = idx
        elif (not alarm) and run_start is not None:
            run_len = idx - run_start
            if run_len >= min_consecutive:
                onset = run_start + min_consecutive - 1
                for pos in range(onset, idx):
                    persisted[pos] = True
            run_start = None
    if run_start is not None:
        run_len = len(alarm_seq) - run_start
        if run_len >= min_consecutive:
            onset = run_start + min_consecutive - 1
            for pos in range(onset, len(alarm_seq)):
                persisted[pos] = True
    return persisted


def compute_event_metrics(
    danger_seq: list[bool],
    alarm_seq: list[bool],
    lead_window: int = 25,
    safe_grace_window: int = 0,
) -> EventMetrics:
    danger_events = 0
    warned_events = 0
    lead_times: list[int] = []
    safe_events = 0
    false_alarms = 0

    in_danger = False
    in_safe = False
    safe_start = 0

    for t, danger in enumerate(danger_seq):
        if danger and not in_danger:
            danger_events += 1
            in_danger = True
            window_start = max(0, t - lead_window)
            warning_steps = [i for i in range(window_start, t + 1) if alarm_seq[i]]
            if warning_steps:
                warned_events += 1
                lead_times.append(t - warning_steps[0])

        if not danger:
            in_danger = False

        if not danger and not in_safe:
            in_safe = True
            safe_start = t
        elif danger and in_safe:
            safe_events += 1
            safe_end = max(safe_start, t - safe_grace_window)
            if any(alarm_seq[safe_start:safe_end]):
                false_alarms += 1
            in_safe = False

    if in_safe:
        safe_events += 1
        if any(alarm_seq[safe_start:]):
            false_alarms += 1

    precision = warned_events / (warned_events + false_alarms) if (warned_events + false_alarms) else 0.0
    recall = warned_events / danger_events if danger_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_alarm_rate = false_alarms / safe_events if safe_events else 0.0
    mean_lead_time = float(np.mean(lead_times)) if lead_times else 0.0
    fn_events = max(danger_events - warned_events, 0)

    return EventMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        false_alarm_rate=false_alarm_rate,
        mean_lead_time=mean_lead_time,
        tp_events=warned_events,
        fp_events=false_alarms,
        fn_events=fn_events,
        safe_events=safe_events,
        lead_times=lead_times,
    )


def danger_sequence_from_npz(
    data: Any,
    *,
    danger_mode: str,
) -> list[bool]:
    if danger_mode == "any":
        return data["danger_any"].astype(bool).tolist()
    if danger_mode == "hard":
        return data["danger_hard"].astype(bool).tolist()
    if danger_mode == "task":
        return data["danger_task"].astype(bool).tolist()
    if danger_mode == "persistent":
        return data["danger_persistent"].astype(bool).tolist()
    if danger_mode == "hard_or_task":
        return np.logical_or(data["danger_hard"] > 0, data["danger_task"] > 0).astype(bool).tolist()
    raise ValueError(f"Unsupported danger mode: {danger_mode}")


def alarm_sequence_from_scores(
    scores: np.ndarray,
    *,
    threshold: float,
) -> list[bool]:
    return (scores >= threshold).astype(bool).tolist()


def evaluate_npz_file(
    npz_path: Path,
    *,
    alarm_scores_key: str,
    alarm_threshold: float,
    danger_mode: str,
    lead_window: int,
    safe_grace_window: int,
    persistence: int,
) -> dict[str, Any]:
    data = np.load(npz_path)
    if alarm_scores_key not in data:
        raise KeyError(f"{npz_path} does not contain score key: {alarm_scores_key}")

    danger_seq = danger_sequence_from_npz(data, danger_mode=danger_mode)
    raw_alarm_seq = alarm_sequence_from_scores(
        data[alarm_scores_key].astype(np.float32),
        threshold=alarm_threshold,
    )
    alarm_seq = apply_alarm_persistence(raw_alarm_seq, min_consecutive=persistence)
    metrics = compute_event_metrics(
        danger_seq,
        alarm_seq,
        lead_window=lead_window,
        safe_grace_window=safe_grace_window,
    )
    return {
        "path": str(npz_path),
        "scene": str(data["scene"].item()),
        "fault_type": str(data["fault_type"].item()),
        "seed": int(data["seed"][0]),
        "alarm_scores_key": alarm_scores_key,
        "alarm_threshold": alarm_threshold,
        "danger_mode": danger_mode,
        "persistence": persistence,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--alarm-scores-key", required=True)
    parser.add_argument("--alarm-threshold", type=float, required=True)
    parser.add_argument(
        "--danger-mode",
        choices=["any", "hard", "task", "persistent", "hard_or_task"],
        default="hard_or_task",
    )
    parser.add_argument("--lead-window", type=int, default=25)
    parser.add_argument("--safe-grace-window", type=int, default=0)
    parser.add_argument("--persistence", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_npz_file(
        Path(args.npz),
        alarm_scores_key=args.alarm_scores_key,
        alarm_threshold=args.alarm_threshold,
        danger_mode=args.danger_mode,
        lead_window=args.lead_window,
        safe_grace_window=args.safe_grace_window,
        persistence=args.persistence,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
