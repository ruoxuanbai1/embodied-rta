#!/usr/bin/env python3
"""
Canonical Case 3 scene-variant registry.

Why this file exists:
1. The current real ALOHA path only restored a few upstream scene families.
2. The experiment plan needs 8-10 concrete benchmark scenes, not only 3 coarse
   family labels.
3. We therefore make the benchmark explicit as scene *variants* built on top of
   the real supported families, while honestly marking unsupported variants.

The registry is the stable source of truth for data collection and later R2/R3
training. It separates:
- scene family: the upstream env family actually used by the adapter
- scene variant: the benchmark condition we want to collect / train on
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SceneVariantSpec:
    name: str
    family: str
    description: str
    supported: bool
    stress_axis: str
    severity_level: int
    paired_reference: str
    collection_priority: int
    adapter_overrides: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneTrialDesign:
    normal_repeats: int
    seeds_per_fault: int
    fault_timings: tuple[str, ...]
    intensity_bands: tuple[str, ...]
    min_fault_trials_per_scene: int
    min_normal_trials_per_scene: int
    convergence_primary_metrics: tuple[str, ...]
    convergence_window: int
    convergence_min_scene_seeds: int
    convergence_ci_level: float
    convergence_rel_half_width_max: float
    convergence_rule: str
    notes: tuple[str, ...] = ()


SCENE_VARIANTS: tuple[SceneVariantSpec, ...] = (
    SceneVariantSpec(
        name="nominal_clear_reference",
        family="nominal_clear",
        description="Reference clean transfer scene used for nominal calibration and paired controls.",
        supported=True,
        stress_axis="reference",
        severity_level=0,
        paired_reference="nominal_clear_reference",
        collection_priority=0,
        notes=(
            "Keep as the main normal-data pool for R2 and R3 calibration.",
        ),
    ),
    SceneVariantSpec(
        name="perception_stressed_mild",
        family="perception_stressed",
        description="Mild but persistent camera-path stress with moderate dimming and vignette.",
        supported=True,
        stress_axis="perception",
        severity_level=1,
        paired_reference="nominal_clear_reference",
        collection_priority=1,
        adapter_overrides={
            "perception_dim_scale": 0.86,
            "perception_vignette_strength": 0.15,
            "perception_vignette_floor": 0.78,
        },
    ),
    SceneVariantSpec(
        name="perception_stressed_strong",
        family="perception_stressed",
        description="Stronger perception stress with darker render and heavier peripheral loss.",
        supported=True,
        stress_axis="perception",
        severity_level=2,
        paired_reference="nominal_clear_reference",
        collection_priority=2,
        adapter_overrides={
            "perception_dim_scale": 0.68,
            "perception_vignette_strength": 0.35,
            "perception_vignette_floor": 0.55,
        },
    ),
    SceneVariantSpec(
        name="dynamic_disturbance_sparse",
        family="dynamic_disturbance",
        description="Sparse dynamic pushes after approach, used as the gentlest dynamic baseline.",
        supported=True,
        stress_axis="dynamics",
        severity_level=1,
        paired_reference="nominal_clear_reference",
        collection_priority=3,
        adapter_overrides={
            "dynamic_start_step": 120,
            "dynamic_period": 9,
            "dynamic_active_mods": (0, 1),
            "dynamic_force": (1.0, 0.5, 0.0, 0.0, 0.0, 0.0),
        },
    ),
    SceneVariantSpec(
        name="dynamic_disturbance_dense",
        family="dynamic_disturbance",
        description="More frequent dynamic pushes with stronger force during the manipulation horizon.",
        supported=True,
        stress_axis="dynamics",
        severity_level=2,
        paired_reference="dynamic_disturbance_sparse",
        collection_priority=4,
        adapter_overrides={
            "dynamic_start_step": 100,
            "dynamic_period": 5,
            "dynamic_active_mods": (0, 1, 2),
            "dynamic_force": (1.5, 0.8, 0.0, 0.0, 0.0, 0.0),
        },
    ),
    SceneVariantSpec(
        name="dynamic_disturbance_late",
        family="dynamic_disturbance",
        description="Late dynamic disturbance concentrated near lift / transfer rather than approach.",
        supported=True,
        stress_axis="timing_shift",
        severity_level=2,
        paired_reference="dynamic_disturbance_sparse",
        collection_priority=5,
        adapter_overrides={
            "dynamic_start_step": 180,
            "dynamic_period": 6,
            "dynamic_active_mods": (0, 1),
            "dynamic_force": (1.2, 0.6, 0.0, 0.0, 0.0, 0.0),
        },
    ),
    SceneVariantSpec(
        name="mixed_perception_dynamics_mild",
        family="dynamic_disturbance",
        description="Mild mixed scene with simultaneous visual stress and sparse dynamic disturbance.",
        supported=True,
        stress_axis="mixed",
        severity_level=1,
        paired_reference="nominal_clear_reference",
        collection_priority=6,
        adapter_overrides={
            "dynamic_start_step": 125,
            "dynamic_period": 8,
            "dynamic_active_mods": (0, 1),
            "dynamic_force": (1.0, 0.5, 0.0, 0.0, 0.0, 0.0),
            "perception_dim_scale": 0.84,
            "perception_vignette_strength": 0.18,
            "perception_vignette_floor": 0.76,
            "mixed_enable_perception": True,
        },
    ),
    SceneVariantSpec(
        name="mixed_perception_dynamics_strong",
        family="dynamic_disturbance",
        description="Strong mixed scene with heavy visual degradation and dense dynamic disturbance.",
        supported=True,
        stress_axis="mixed",
        severity_level=2,
        paired_reference="mixed_perception_dynamics_mild",
        collection_priority=7,
        adapter_overrides={
            "dynamic_start_step": 105,
            "dynamic_period": 5,
            "dynamic_active_mods": (0, 1, 2),
            "dynamic_force": (1.5, 0.8, 0.0, 0.0, 0.0, 0.0),
            "perception_dim_scale": 0.70,
            "perception_vignette_strength": 0.34,
            "perception_vignette_floor": 0.56,
            "mixed_enable_perception": True,
        },
    ),
    SceneVariantSpec(
        name="static_clutter_light",
        family="static_clutter",
        description="Planned light clutter benchmark once a real clutter-capable ALOHA backend is restored.",
        supported=False,
        stress_axis="geometry",
        severity_level=1,
        paired_reference="nominal_clear_reference",
        collection_priority=8,
        notes=(
            "Blocked because upstream gym-aloha transfer cube currently has no real static clutter variant.",
        ),
    ),
    SceneVariantSpec(
        name="static_clutter_heavy",
        family="static_clutter",
        description="Planned heavy clutter benchmark once a real clutter-capable ALOHA backend is restored.",
        supported=False,
        stress_axis="geometry",
        severity_level=2,
        paired_reference="static_clutter_light",
        collection_priority=9,
        notes=(
            "Blocked because upstream gym-aloha transfer cube currently has no real static clutter variant.",
        ),
    ),
)


SCENE_VARIANT_MAP = {spec.name: spec for spec in SCENE_VARIANTS}


DEFAULT_SCENE_TRIAL_DESIGN = SceneTrialDesign(
    normal_repeats=5,
    seeds_per_fault=5,
    fault_timings=("early", "mid", "late"),
    intensity_bands=("mild", "medium", "severe"),
    min_fault_trials_per_scene=45,
    min_normal_trials_per_scene=5,
    convergence_primary_metrics=("precision", "recall", "f1_score", "false_alarm_rate", "mean_lead_time"),
    convergence_window=2,
    convergence_min_scene_seeds=5,
    convergence_ci_level=0.95,
    convergence_rel_half_width_max=0.10,
    convergence_rule="relative_half_width_only",
    notes=(
        "One scene is not one rollout. Each scene variant should be repeated with multiple seeds.",
        "Fault runs should sweep onset timing and intensity band separately.",
        "Use paired normal-vs-fault comparisons within the same seed and base fault profile.",
        "Do not treat 1-3 repeats as enough for training. Use at least 5 normal seeds and 45 fault trials per scene-fault block.",
        "Repeat count is not fixed forever. The final stopping rule should come from metric-mean convergence across added seeds.",
        "For the final stop rule, use the confidence-interval relative half-width as the criterion; keep window-to-window drift only as a diagnostic.",
    ),
)


def default_split_for_seed(seed: int) -> str:
    if seed <= 44:
        return "train"
    if seed == 45:
        return "val"
    return "test"


def scene_variant_names(*, supported_only: bool = False) -> list[str]:
    filtered = [spec for spec in SCENE_VARIANTS if (spec.supported or not supported_only)]
    filtered.sort(key=lambda spec: (spec.collection_priority, spec.name))
    return [spec.name for spec in filtered]


def get_scene_variant(name: str) -> SceneVariantSpec:
    if name not in SCENE_VARIANT_MAP:
        raise KeyError(f"Unknown scene variant: {name}")
    return SCENE_VARIANT_MAP[name]


def is_scene_variant(name: str) -> bool:
    return name in SCENE_VARIANT_MAP


def recommended_trial_count_per_fault(scene_name: str) -> int:
    _ = get_scene_variant(scene_name)
    return (
        DEFAULT_SCENE_TRIAL_DESIGN.seeds_per_fault
        * len(DEFAULT_SCENE_TRIAL_DESIGN.fault_timings)
        * len(DEFAULT_SCENE_TRIAL_DESIGN.intensity_bands)
    )


def supported_scene_variants_by_axis() -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[int, str]]] = {}
    for spec in SCENE_VARIANTS:
        if not spec.supported:
            continue
        grouped.setdefault(spec.stress_axis, []).append((spec.severity_level, spec.name))
    return {
        axis: [name for _, name in sorted(rows)]
        for axis, rows in grouped.items()
    }
