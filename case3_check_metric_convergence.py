#!/usr/bin/env python3
"""
Check whether online metrics have stabilized as more repeats are added.

This utility is intentionally simple:
- input is a JSON list of evaluation snapshots ordered by collection stage
- it estimates uncertainty on the most recent summary metrics
- it reports whether the recent metric means are precise enough to stop adding repeats
- it also reports variance, confidence-interval half-width, and relative half-width
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-json", required=True)
    parser.add_argument("--scope", choices=["overall", "by_scene"], default="overall")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--metrics", nargs="+", default=["precision", "recall", "f1_score", "false_alarm_rate", "mean_lead_time"])
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--rel-half-width-max", type=float, default=0.10)
    return parser.parse_args()


def _extract_layer_metrics(snapshot: dict[str, Any], scope: str, scene: str | None, layer: str) -> dict[str, float]:
    if scope == "overall":
        return snapshot["overall"][layer]
    if scene is None:
        raise ValueError("--scene is required when scope=by_scene")
    return snapshot["by_scene"][scene][layer]


def _z_value_for_ci(ci_level: float) -> float:
    # Keep the utility lightweight: support the usual confidence levels directly.
    if abs(ci_level - 0.90) < 1e-9:
        return 1.645
    if abs(ci_level - 0.95) < 1e-9:
        return 1.96
    if abs(ci_level - 0.99) < 1e-9:
        return 2.576
    # Normal approximation fallback.
    return 1.96


def _summary_stats(values: list[float], ci_level: float) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    variance = float(std * std)
    sem = std / math.sqrt(len(arr)) if len(arr) > 0 else 0.0
    z = _z_value_for_ci(ci_level)
    half_width = float(z * sem)
    rel_half_width = float(half_width / abs(mean)) if abs(mean) > 1e-12 else float("inf")
    return {
        "mean": mean,
        "std": std,
        "variance": variance,
        "sem": sem,
        "half_width": half_width,
        "rel_half_width": rel_half_width,
    }


def main() -> None:
    args = parse_args()
    history = json.loads(Path(args.history_json).read_text(encoding="utf-8"))
    if len(history) < args.window:
        raise RuntimeError(
            f"Need at least {args.window} snapshots to estimate the current uncertainty window, got {len(history)}"
        )

    series = [
        _extract_layer_metrics(snapshot, args.scope, args.scene, args.layer)
        for snapshot in history
    ]
    curr = series[-args.window :]
    prev = series[-2 * args.window : -args.window] if len(series) >= args.window * 2 else None

    metric_report: dict[str, Any] = {}
    stable = True
    for metric in args.metrics:
        curr_stats = _summary_stats([row[metric] for row in curr], args.ci_level)
        prev_stats = _summary_stats([row[metric] for row in prev], args.ci_level) if prev is not None else None
        delta = abs(curr_stats["mean"] - prev_stats["mean"]) if prev_stats is not None else None
        metric_stable = curr_stats["rel_half_width"] <= args.rel_half_width_max
        stable = stable and metric_stable
        metric_report[metric] = {
            "prev_mean": prev_stats["mean"] if prev_stats is not None else None,
            "prev_std": prev_stats["std"] if prev_stats is not None else None,
            "prev_variance": prev_stats["variance"] if prev_stats is not None else None,
            "prev_sem": prev_stats["sem"] if prev_stats is not None else None,
            "prev_half_width": prev_stats["half_width"] if prev_stats is not None else None,
            "prev_rel_half_width": prev_stats["rel_half_width"] if prev_stats is not None else None,
            "curr_mean": curr_stats["mean"],
            "curr_std": curr_stats["std"],
            "curr_variance": curr_stats["variance"],
            "curr_sem": curr_stats["sem"],
            "curr_half_width": curr_stats["half_width"],
            "curr_rel_half_width": curr_stats["rel_half_width"],
            "abs_delta": delta,
            "rel_half_width_ok": curr_stats["rel_half_width"] <= args.rel_half_width_max,
            "stable": metric_stable,
        }

    payload = {
        "criterion": "relative_half_width_only",
        "scope": args.scope,
        "scene": args.scene,
        "layer": args.layer,
        "window": args.window,
        "ci_level": args.ci_level,
        "rel_half_width_max": args.rel_half_width_max,
        "stable": stable,
        "metrics": metric_report,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
