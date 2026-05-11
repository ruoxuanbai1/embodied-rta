#!/usr/bin/env python3
"""
Collect compact real-env trajectories for Case 3 R2 / R3 training.

Design choices:
1. Use the restored real ACT + gym-aloha path, not the toy smoke env.
2. Collect compact NPZ trajectories that are small enough to keep and sync.
3. Keep one data format that supports both:
   - R2 short-horizon hazard prediction
   - R3 OOD / gradient / routing calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from case3_act_legacy_r3_signals import (
    LegacyRoutingCache,
    compute_first_action_state_attribution_with_legacy_policy,
    pool_route_feature,
    register_legacy_canonical_routing_hooks,
)
from case3_act_legacy_source_loader import (
    load_legacy_act_policy_from_source,
    select_action_with_legacy_policy,
)
from case3_canonical_spec import (
    canonical_fault_names,
    canonical_intensity_band_names,
    resolved_fault_profile_with_band_dict,
)
from case3_gym_aloha_adapter import GymAlohaTransferCubeAdapter
from case3_scene_registry import (
    DEFAULT_SCENE_TRIAL_DESIGN,
    default_split_for_seed,
    get_scene_variant,
    scene_variant_names,
)


DEFAULT_MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"
DEFAULT_LEGACY_SOURCE_ROOT = "/home/vipuser/lerobot-3c0a209f9fac4d2a57617e686a7f2a2309144ba2"
DEFAULT_OUTPUT_DIR = "./outputs/case3_r23_dataset"

PHASE_TO_ID = {
    "approach": 0,
    "grasp": 1,
    "lift": 2,
    "transfer": 3,
    "place": 4,
    "recovery": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--legacy-source-root", default=DEFAULT_LEGACY_SOURCE_ROOT)
    parser.add_argument("--scenes", nargs="+", default=scene_variant_names(supported_only=True))
    parser.add_argument("--faults", nargs="+", default=["normal"])
    parser.add_argument("--fault-timings", nargs="+", choices=["early", "mid", "late"], default=list(DEFAULT_SCENE_TRIAL_DESIGN.fault_timings))
    parser.add_argument("--intensity-bands", nargs="+", choices=canonical_intensity_band_names(), default=list(DEFAULT_SCENE_TRIAL_DESIGN.intensity_bands))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gradient-every", type=int, default=1)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def collect_single_rollout(
    *,
    adapter: GymAlohaTransferCubeAdapter,
    policy: Any,
    scene: str,
    fault_type: str | None,
    fault_profile: dict[str, Any] | None,
    seed: int,
    max_steps: int,
    device: torch.device,
    gradient_every: int,
) -> dict[str, Any]:
    obs = adapter.reset(
        scene=scene,
        fault_type=fault_type,
        fault_inject_step=None if fault_profile is None else int(fault_profile["onset_step"]),
        fault_profile=fault_profile,
        seed=seed,
    )
    if hasattr(policy, "reset"):
        policy.reset()

    cache = LegacyRoutingCache()
    handles = register_legacy_canonical_routing_hooks(policy, cache)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    danger_any: list[int] = []
    danger_hard: list[int] = []
    danger_task: list[int] = []
    danger_persistent: list[int] = []
    gradients: list[np.ndarray] = []
    phis: list[np.ndarray] = []
    gradient_mask: list[int] = []
    route_src: list[np.ndarray] = []
    route_dst: list[np.ndarray] = []
    phases: list[int] = []

    done = False
    steps = 0
    for step_idx in range(max_steps):
        attribution = None
        if gradient_every <= 1 or step_idx % gradient_every == 0:
            attribution = compute_first_action_state_attribution_with_legacy_policy(
                policy=policy,
                state=obs.state,
                top_image=obs.top_image,
                device=device,
                scalar_mode="l1",
            )

        action = select_action_with_legacy_policy(
            policy=policy,
            state=obs.state,
            top_image=obs.top_image,
            device=device,
        )
        step_result = adapter.step(action)

        states.append(obs.state.astype(np.float32))
        actions.append(action.astype(np.float32))
        rewards.append(float(step_result.reward))
        danger_any.append(int(step_result.danger.any_danger))
        danger_hard.append(int(step_result.danger.hard_safety))
        danger_task.append(int(step_result.danger.task_critical_failure))
        danger_persistent.append(int(step_result.danger.persistent_instability))
        phases.append(PHASE_TO_ID.get(obs.phase, -1))

        if attribution is None:
            gradients.append(np.zeros_like(obs.state, dtype=np.float32))
            phis.append(np.zeros_like(obs.state, dtype=np.float32))
            gradient_mask.append(0)
        else:
            gradients.append(np.asarray(attribution["gradient"], dtype=np.float32))
            phis.append(np.asarray(attribution["phi"], dtype=np.float32))
            gradient_mask.append(1)

        src = pool_route_feature(cache.decoder_last_ffn)
        dst = pool_route_feature(cache.action_head_input)
        route_src.append(
            np.zeros((0,), dtype=np.float32) if src is None else src.astype(np.float32)
        )
        route_dst.append(
            np.zeros((0,), dtype=np.float32) if dst is None else dst.astype(np.float32)
        )

        obs = step_result.observation
        done = bool(step_result.done)
        steps = step_idx + 1
        if done:
            break

    for handle in handles:
        handle.remove()

    route_src_dim = max((arr.shape[0] for arr in route_src), default=0)
    route_dst_dim = max((arr.shape[0] for arr in route_dst), default=0)
    route_src_arr = np.zeros((steps, route_src_dim), dtype=np.float32)
    route_dst_arr = np.zeros((steps, route_dst_dim), dtype=np.float32)
    for i, arr in enumerate(route_src[:steps]):
        if arr.size:
            route_src_arr[i, : arr.shape[0]] = arr
    for i, arr in enumerate(route_dst[:steps]):
        if arr.size:
            route_dst_arr[i, : arr.shape[0]] = arr

    return {
        "states": np.stack(states[:steps], axis=0).astype(np.float32),
        "actions": np.stack(actions[:steps], axis=0).astype(np.float32),
        "rewards": np.asarray(rewards[:steps], dtype=np.float32),
        "danger_any": np.asarray(danger_any[:steps], dtype=np.uint8),
        "danger_hard": np.asarray(danger_hard[:steps], dtype=np.uint8),
        "danger_task": np.asarray(danger_task[:steps], dtype=np.uint8),
        "danger_persistent": np.asarray(danger_persistent[:steps], dtype=np.uint8),
        "gradients": np.stack(gradients[:steps], axis=0).astype(np.float32),
        "phis": np.stack(phis[:steps], axis=0).astype(np.float32),
        "gradient_mask": np.asarray(gradient_mask[:steps], dtype=np.uint8),
        "route_src": route_src_arr,
        "route_dst": route_dst_arr,
        "phase_ids": np.asarray(phases[:steps], dtype=np.int16),
        "done": np.asarray([done], dtype=np.uint8),
        "scene": np.asarray(scene),
        "fault_type": np.asarray(fault_type or "normal"),
        "seed": np.asarray([seed], dtype=np.int32),
        "steps": np.asarray([steps], dtype=np.int32),
        "fault_onset_step": np.asarray(
            [-1 if fault_profile is None else int(fault_profile["onset_step"])], dtype=np.int32
        ),
        "fault_end_step": np.asarray(
            [-1 if fault_profile is None else int(fault_profile["end_step"])], dtype=np.int32
        ),
        "fault_peak_intensity": np.asarray(
            [0.0 if fault_profile is None else float(fault_profile["peak_intensity"])], dtype=np.float32
        ),
        "fault_intensity_band": np.asarray(
            ["none" if fault_profile is None else str(fault_profile.get("intensity_band", "unknown"))]
        ),
        "fault_timing": np.asarray(
            ["none" if fault_profile is None else str(fault_profile.get("fault_timing", "unknown"))]
        ),
        "split": np.asarray([default_split_for_seed(seed)], dtype=np.str_),
    }


def manifest_entry_from_rollout(
    *,
    out_path: Path,
    scene_spec: Any,
    fault_type: str,
    seed: int,
    rollout: dict[str, Any],
    fault_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "path": str(out_path),
        "scene": str(rollout["scene"]),
        "scene_family": scene_spec.family,
        "stress_axis": scene_spec.stress_axis,
        "severity_level": scene_spec.severity_level,
        "paired_reference": scene_spec.paired_reference,
        "fault_type": fault_type,
        "seed": seed,
        "steps": int(rollout["steps"][0]),
        "fault_timing": "none" if fault_profile is None else str(fault_profile["fault_timing"]),
        "intensity_band": "none" if fault_profile is None else str(fault_profile["intensity_band"]),
        "fault_onset_step": -1 if fault_profile is None else int(fault_profile["onset_step"]),
        "fault_end_step": -1 if fault_profile is None else int(fault_profile["end_step"]),
        "fault_peak_intensity": 0.0 if fault_profile is None else float(fault_profile["peak_intensity"]),
        "split": default_split_for_seed(seed),
    }


def write_manifest(manifest_path: Path, manifest: list[dict[str, Any]]) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_rollout(
    *,
    out_path: Path,
    rollout: dict[str, Any],
    manifest: list[dict[str, Any]],
    manifest_entry: dict[str, Any],
    manifest_path: Path,
) -> None:
    np.savez_compressed(out_path, **rollout)
    manifest.append(manifest_entry)
    write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "saved": str(out_path),
                "fault_type": manifest_entry["fault_type"],
                "seed": manifest_entry["seed"],
                "steps": manifest_entry["steps"],
                "fault_timing": manifest_entry["fault_timing"],
                "intensity_band": manifest_entry["intensity_band"],
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = load_legacy_act_policy_from_source(
        model_id=args.model_id,
        legacy_source_root=args.legacy_source_root,
        device=device,
    )
    adapter = GymAlohaTransferCubeAdapter()

    fault_names = [f for f in args.faults]
    invalid_faults = [f for f in fault_names if f != "normal" and f not in canonical_fault_names()]
    if invalid_faults:
        raise ValueError(f"Unknown fault names: {invalid_faults}")

    manifest_path = output_dir / "manifest.json"
    manifest: list[dict[str, Any]] = load_manifest(manifest_path) if args.append else []
    write_manifest(manifest_path, manifest)
    for scene in args.scenes:
        scene_spec = get_scene_variant(scene)
        scene_dir = output_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            normal_name = f"{scene}__normal__seed{seed}.npz"
            normal_path = scene_dir / normal_name
            if not (args.skip_existing and normal_path.exists()):
                normal_rollout = collect_single_rollout(
                    adapter=adapter,
                    policy=policy,
                    scene=scene,
                    fault_type=None,
                    fault_profile=None,
                    seed=seed,
                    max_steps=args.max_steps,
                    device=device,
                    gradient_every=args.gradient_every,
                )
                save_rollout(
                    out_path=normal_path,
                    rollout=normal_rollout,
                    manifest=manifest,
                    manifest_entry=manifest_entry_from_rollout(
                        out_path=normal_path,
                        scene_spec=scene_spec,
                        fault_type="normal",
                        seed=seed,
                        rollout=normal_rollout,
                        fault_profile=None,
                    ),
                    manifest_path=manifest_path,
                )
            else:
                print(json.dumps({"skipped_existing": str(normal_path)}), flush=True)

            for fault_name in fault_names:
                if fault_name == "normal":
                    continue
                for fault_timing in args.fault_timings:
                    for intensity_band in args.intensity_bands:
                        fault_profile = resolved_fault_profile_with_band_dict(
                            fault_timing,
                            seed=seed,
                            fault_name=fault_name,
                            intensity_band=intensity_band,
                        )
                        fault_profile["fault_timing"] = fault_timing
                        fault_profile["intensity_band"] = intensity_band
                        rollout = collect_single_rollout(
                            adapter=adapter,
                            policy=policy,
                            scene=scene,
                            fault_type=fault_name,
                            fault_profile=fault_profile,
                            seed=seed,
                            max_steps=args.max_steps,
                            device=device,
                            gradient_every=args.gradient_every,
                        )
                        out_name = (
                            f"{scene}__{fault_name}__{fault_timing}__{intensity_band}__seed{seed}.npz"
                        )
                        out_path = scene_dir / out_name
                        if args.skip_existing and out_path.exists():
                            print(json.dumps({"skipped_existing": str(out_path)}), flush=True)
                            continue
                        rollout = collect_single_rollout(
                            adapter=adapter,
                            policy=policy,
                            scene=scene,
                            fault_type=fault_name,
                            fault_profile=fault_profile,
                            seed=seed,
                            max_steps=args.max_steps,
                            device=device,
                            gradient_every=args.gradient_every,
                        )
                        save_rollout(
                            out_path=out_path,
                            rollout=rollout,
                            manifest=manifest,
                            manifest_entry=manifest_entry_from_rollout(
                                out_path=out_path,
                                scene_spec=scene_spec,
                                fault_type=fault_name,
                                seed=seed,
                                rollout=rollout,
                                fault_profile=fault_profile,
                            ),
                            manifest_path=manifest_path,
                        )

    print(manifest_path)


if __name__ == "__main__":
    main()
