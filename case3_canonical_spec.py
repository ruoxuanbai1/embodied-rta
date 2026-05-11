#!/usr/bin/env python3
"""
Canonical Case 3 experimental specification for the ACT-based line.

This module does not run experiments by itself. It freezes the design choices
that later data collection, evaluation, and R123 improvements should follow.

Key intent:
1. Keep the tested object faithful to ACT.
2. Avoid collapsing Case 3 into a toy envelope-only benchmark.
3. Preserve all three Region-3 branches:
   - OOD
   - feature logic / gradient
   - routing
4. Treat legacy smoke collectors and simplified simulators as transitional
   tools rather than the final benchmark definition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import random


PhaseName = Literal["approach", "grasp", "lift", "transfer", "place", "recovery"]
SceneFamilyName = Literal[
    "nominal_clear",
    "static_clutter",
    "dynamic_disturbance",
    "perception_stressed",
]
FaultTimingName = Literal["early", "mid", "late"]
IntensityBandName = Literal["mild", "medium", "severe"]


@dataclass(frozen=True)
class SceneSpec:
    name: SceneFamilyName
    description: str
    legacy_aliases: tuple[str, ...] = ()
    requires_real_camera_path: bool = True
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FaultSpec:
    name: str
    family: str
    description: str
    expected_sensitive_branches: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FaultScheduleSpec:
    rollout_steps: int
    onset_range: tuple[int, int]
    duration_range: tuple[int, int]
    peak_intensity_range: tuple[float, float]
    intensity_profile: Literal["step", "ramp", "pulse"]
    description: str


@dataclass(frozen=True)
class ResolvedFaultProfile:
    onset_step: int
    end_step: int
    peak_intensity: float
    intensity_profile: Literal["step", "ramp", "pulse"]
    description: str


@dataclass(frozen=True)
class IntensityBandSpec:
    name: IntensityBandName
    peak_scale_range: tuple[float, float]
    duration_scale_range: tuple[float, float]
    description: str


@dataclass(frozen=True)
class DangerEventChannelSpec:
    name: str
    description: str
    examples: tuple[str, ...]


@dataclass(frozen=True)
class RoutingTapSpec:
    name: str
    source_module: str
    target_module: str
    purpose: str
    is_primary: bool = False


@dataclass(frozen=True)
class GradientObjectiveSpec:
    objective_name: str
    target_action_step: int
    scalarization: str
    attribution_formula: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CanonicalCase3Spec:
    rollout_steps: int = 400
    lead_window: int = 40
    cooldown_steps: int = 40
    merge_gap: int = 0
    scenes: tuple[SceneSpec, ...] = field(default_factory=tuple)
    faults: tuple[FaultSpec, ...] = field(default_factory=tuple)
    danger_channels: tuple[DangerEventChannelSpec, ...] = field(default_factory=tuple)
    routing_taps: tuple[RoutingTapSpec, ...] = field(default_factory=tuple)
    gradient_objective: GradientObjectiveSpec | None = None


SCENES: tuple[SceneSpec, ...] = (
    SceneSpec(
        name="nominal_clear",
        description="Clean tabletop transfer with minimal clutter and nominal sensing.",
        legacy_aliases=("empty",),
        notes=(
            "Use as the nominal reference pool for calibration.",
            "Must still use the real ACT camera-input path rather than synthetic images.",
        ),
    ),
    SceneSpec(
        name="static_clutter",
        description="Static clutter around the transfer workspace to stress geometric margin.",
        legacy_aliases=("static",),
        notes=(
            "Keep obstacle layout fixed per seed for paired normal-vs-fault comparison.",
            "Use to stress R1 collision margin and R2 short-horizon reachability.",
        ),
    ),
    SceneSpec(
        name="dynamic_disturbance",
        description="A dynamic scene with late obstacle entry or contact-side disturbance.",
        legacy_aliases=("dense",),
        notes=(
            "Used to test whether warning signals remain informative under nonstationary contact geometry.",
            "Prefer explicit disturbance schedules rather than random obstacle churn.",
        ),
    ),
    SceneSpec(
        name="perception_stressed",
        description="A real camera-observation scene with lighting, occlusion, and background stress.",
        notes=(
            "Perception faults should enter through the observation path, not by post-hoc score hacking.",
            "This scene family is the main validation bed for OOD and part of feature logic.",
        ),
    ),
)


FAULTS: tuple[FaultSpec, ...] = (
    FaultSpec(
        name="F1_lighting",
        family="perception",
        description="Illumination degradation along the real camera path.",
        expected_sensitive_branches=("ood", "gradient"),
    ),
    FaultSpec(
        name="F2_occlusion",
        family="perception",
        description="Camera occlusion or partial field-of-view corruption.",
        expected_sensitive_branches=("ood", "routing"),
    ),
    FaultSpec(
        name="F3_adversarial",
        family="perception",
        description="Adversarial or structured observation corruption.",
        expected_sensitive_branches=("ood", "gradient", "routing"),
    ),
    FaultSpec(
        name="F4_payload",
        family="dynamics",
        description="Payload shift or inertial mismatch during manipulation.",
        expected_sensitive_branches=("gradient", "routing", "r2"),
    ),
    FaultSpec(
        name="F5_friction",
        family="actuation",
        description="Actuation/friction degradation causing action-to-state mismatch.",
        expected_sensitive_branches=("r2", "gradient", "routing"),
    ),
    FaultSpec(
        name="F6_dynamic",
        family="environment",
        description="Late dynamic obstacle or dynamic contact disturbance.",
        expected_sensitive_branches=("r1", "r2", "routing"),
    ),
    FaultSpec(
        name="F7_sensor",
        family="state_estimation",
        description="State-sensor corruption entering the structured observation stream.",
        expected_sensitive_branches=("ood", "gradient"),
    ),
    FaultSpec(
        name="F8_compound",
        family="compound",
        description="Coupled perception and dynamics disturbance.",
        expected_sensitive_branches=("ood", "gradient", "routing", "r2"),
    ),
)


FAULT_SCHEDULES: dict[FaultTimingName, FaultScheduleSpec] = {
    "early": FaultScheduleSpec(
        rollout_steps=400,
        onset_range=(40, 90),
        duration_range=(60, 120),
        peak_intensity_range=(0.35, 0.70),
        intensity_profile="ramp",
        description="Early corruption during the long approach stage, with variable duration and ramped strength.",
    ),
    "mid": FaultScheduleSpec(
        rollout_steps=400,
        onset_range=(130, 190),
        duration_range=(50, 110),
        peak_intensity_range=(0.45, 0.85),
        intensity_profile="ramp",
        description="Mid-rollout corruption around first contact / lift, with variable onset and strength; this is the default paired warning benchmark.",
    ),
    "late": FaultScheduleSpec(
        rollout_steps=400,
        onset_range=(240, 320),
        duration_range=(30, 80),
        peak_intensity_range=(0.55, 1.00),
        intensity_profile="pulse",
        description="Late corruption during lift / transfer with shorter, sharper bursts to stress short lead time and fast-response warning quality.",
    ),
}


INTENSITY_BANDS: dict[IntensityBandName, IntensityBandSpec] = {
    "mild": IntensityBandSpec(
        name="mild",
        peak_scale_range=(0.70, 0.90),
        duration_scale_range=(0.85, 1.00),
        description="Visible but recoverable corruption used for weak-signal sensitivity checks.",
    ),
    "medium": IntensityBandSpec(
        name="medium",
        peak_scale_range=(0.95, 1.05),
        duration_scale_range=(0.95, 1.05),
        description="Default benchmark intensity centered on the canonical profile.",
    ),
    "severe": IntensityBandSpec(
        name="severe",
        peak_scale_range=(1.15, 1.40),
        duration_scale_range=(1.00, 1.20),
        description="Strong corruption used to test whether warning quality scales with fault severity.",
    ),
}


DANGER_EVENT_CHANNELS: tuple[DangerEventChannelSpec, ...] = (
    DangerEventChannelSpec(
        name="hard_safety",
        description="Physical safety breach or near-breach.",
        examples=(
            "collision",
            "self_collision",
            "joint_limit_violation",
            "velocity_or_torque_violation",
        ),
    ),
    DangerEventChannelSpec(
        name="task_critical_failure",
        description="Manipulation failure that can still occur before hard envelope breach.",
        examples=(
            "grasp_loss",
            "object_slip",
            "object_drop",
            "gross_place_misalignment",
        ),
    ),
    DangerEventChannelSpec(
        name="persistent_instability",
        description="Sustained precursor behavior indicating likely near-future failure.",
        examples=(
            "no_progress_for_n_steps",
            "oscillatory_action_burst",
            "repeated_corrective_dithering",
            "monotonic_target_error_worsening",
        ),
    ),
)


ROUTING_TAPS: tuple[RoutingTapSpec, ...] = (
    RoutingTapSpec(
        name="decoder_last_to_action_head",
        source_module="decoder.layers[-1].linear2",
        target_module="action_head",
        purpose="Primary hidden-to-action route used for the canonical R3 routing branch.",
        is_primary=True,
    ),
    RoutingTapSpec(
        name="decoder_cross_attention_to_action_head",
        source_module="decoder.layers[-1].multihead_attn",
        target_module="action_head",
        purpose="Secondary route capturing whether visual/context fusion reaches the action head in a nominal way.",
        is_primary=False,
    ),
)


GRADIENT_OBJECTIVE = GradientObjectiveSpec(
    objective_name="first_action_state_attribution",
    target_action_step=0,
    scalarization="sum(abs(a_t0)) over the first executable action vector",
    attribution_formula="phi_i = x_i * d objective / d x_i",
    notes=(
        "Use true autograd rather than finite difference whenever the ACT path is available.",
        "Legal feature subsets should be phase-conditioned rather than fixed globally.",
        "The final logic score should measure attribution mass inside the condition-specific legal subset.",
    ),
)


CANONICAL_CASE3_SPEC = CanonicalCase3Spec(
    scenes=SCENES,
    faults=FAULTS,
    danger_channels=DANGER_EVENT_CHANNELS,
    routing_taps=ROUTING_TAPS,
    gradient_objective=GRADIENT_OBJECTIVE,
)


LEGACY_SCENE_ALIAS_MAP = {
    alias: scene.name
    for scene in SCENES
    for alias in scene.legacy_aliases
}


def normalize_scene_name(scene_name: str) -> str:
    """Map legacy scene aliases to canonical scene-family names."""
    return LEGACY_SCENE_ALIAS_MAP.get(scene_name, scene_name)


def canonical_fault_names() -> list[str]:
    return [fault.name for fault in FAULTS]


def canonical_scene_names() -> list[str]:
    return [scene.name for scene in SCENES]


def canonical_intensity_band_names() -> list[str]:
    return [band.name for band in INTENSITY_BANDS.values()]


def resolve_fault_profile(
    timing: FaultTimingName,
    *,
    seed: int,
    fault_name: str | None = None,
) -> ResolvedFaultProfile:
    schedule = FAULT_SCHEDULES[timing]
    salt = sum(ord(ch) for ch in (fault_name or "normal"))
    rng = random.Random(seed * 9973 + salt)
    onset_step = rng.randint(*schedule.onset_range)
    duration = rng.randint(*schedule.duration_range)
    peak_intensity = rng.uniform(*schedule.peak_intensity_range)
    end_step = min(schedule.rollout_steps - 1, onset_step + duration - 1)
    return ResolvedFaultProfile(
        onset_step=onset_step,
        end_step=end_step,
        peak_intensity=peak_intensity,
        intensity_profile=schedule.intensity_profile,
        description=schedule.description,
    )


def resolved_fault_profile_dict(
    timing: FaultTimingName,
    *,
    seed: int,
    fault_name: str | None = None,
) -> dict[str, object]:
    return asdict(resolve_fault_profile(timing, seed=seed, fault_name=fault_name))


def apply_intensity_band(
    profile: ResolvedFaultProfile,
    *,
    band: IntensityBandName,
    seed: int,
    fault_name: str | None = None,
) -> ResolvedFaultProfile:
    band_spec = INTENSITY_BANDS[band]
    salt = sum(ord(ch) for ch in f"{fault_name or 'normal'}::{band}")
    rng = random.Random(seed * 12391 + salt)
    peak_scale = rng.uniform(*band_spec.peak_scale_range)
    duration_scale = rng.uniform(*band_spec.duration_scale_range)

    onset_step = int(profile.onset_step)
    duration = max(1, profile.end_step - profile.onset_step + 1)
    scaled_duration = max(1, int(round(duration * duration_scale)))
    end_step = onset_step + scaled_duration - 1

    return ResolvedFaultProfile(
        onset_step=onset_step,
        end_step=end_step,
        peak_intensity=float(profile.peak_intensity * peak_scale),
        intensity_profile=profile.intensity_profile,
        description=f"{profile.description} Intensity band: {band_spec.description}",
    )


def resolved_fault_profile_with_band_dict(
    timing: FaultTimingName,
    *,
    seed: int,
    fault_name: str | None = None,
    intensity_band: IntensityBandName = "medium",
) -> dict[str, object]:
    base = resolve_fault_profile(timing, seed=seed, fault_name=fault_name)
    return asdict(apply_intensity_band(base, band=intensity_band, seed=seed, fault_name=fault_name))
