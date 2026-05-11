#!/usr/bin/env python3
"""
Real `gym-aloha` adapter for the canonical ACT Case 3 pipeline.

Design intent:
1. Use the upstream `gym-aloha` MuJoCo environment instead of local synthetic
   image collectors.
2. Expose a canonical observation with real top-camera pixels and real
   14-dimensional agent state.
3. Keep fault injection on the real observation / physics path whenever
   possible.

Current scope:
- task: `gym_aloha/AlohaTransferCube-v0`
- observation mode: `pixels_agent_pos`
- supported canonical scenes:
  - `nominal_clear`
  - `perception_stressed`
  - `dynamic_disturbance`
- `static_clutter` is not yet supported because the upstream task does not ship
  a cluttered transfer-cube variant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from case3_canonical_env import Case3EnvAdapter, Case3Observation, DangerChannels, StepResult
from case3_scene_registry import get_scene_variant, is_scene_variant


@dataclass
class _FaultRuntimeState:
    payload_applied: bool = False
    friction_applied: bool = False


class GymAlohaTransferCubeAdapter(Case3EnvAdapter):
    """
    Canonical adapter around the upstream `gym-aloha` transfer-cube task.
    """

    SUPPORTED_SCENES = {"nominal_clear", "perception_stressed", "dynamic_disturbance"}
    SUPPORTED_FAULTS = {
        None,
        "F1_lighting",
        "F2_occlusion",
        "F3_adversarial",
        "F4_payload",
        "F5_friction",
        "F6_dynamic",
        "F7_sensor",
        "F8_compound",
    }

    def __init__(
        self,
        *,
        task_id: str = "gym_aloha/AlohaTransferCube-v0",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        action_clip: tuple[float, float] | None = None,
        no_progress_window: int = 12,
    ) -> None:
        self.task_id = task_id
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.action_clip = action_clip
        self.no_progress_window = no_progress_window

        self._env: Any | None = None
        self._scene: str = "nominal_clear"
        self._scene_variant_name: str = "nominal_clear"
        self._scene_config: dict[str, Any] = {}
        self._fault_type: str | None = None
        self._fault_inject_step: int | None = None
        self._fault_profile: dict[str, Any] | None = None
        self._current_step: int = 0
        self._last_obs: dict[str, Any] | None = None
        self._last_info: dict[str, Any] = {}
        self._last_reward: float = 0.0
        self._runtime = _FaultRuntimeState()

        self._reward_history: list[float] = []
        self._action_history: list[np.ndarray] = []
        self._box_target_distance_history: list[float] = []
        self._max_reward_seen: float = 0.0
        self._lift_seen: bool = False

        self._base_box_mass: float | None = None
        self._base_box_friction: np.ndarray | None = None
        self._base_table_friction: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public adapter API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        scene: str,
        fault_type: str | None,
        fault_inject_step: int | None,
        fault_profile: dict[str, Any] | None,
        seed: int,
    ) -> Case3Observation:
        scene = self._resolve_scene_variant(scene)
        self._validate_scene(scene)
        self._validate_fault(fault_type)
        self._scene = scene
        self._scene_variant_name = scene
        self._fault_type = fault_type
        self._fault_inject_step = fault_inject_step
        self._fault_profile = fault_profile
        self._current_step = 0
        self._last_reward = 0.0
        self._last_info = {}
        self._reward_history = []
        self._action_history = []
        self._box_target_distance_history = []
        self._max_reward_seen = 0.0
        self._lift_seen = False
        self._runtime = _FaultRuntimeState()

        self._recreate_env()
        obs, info = self._env.reset(seed=seed)
        self._last_obs = obs
        self._last_info = dict(info)
        self._capture_baseline_physics()
        canonical_obs = self._build_observation(obs, info)
        self._record_progress(canonical_obs, reward=0.0, action=None)
        return canonical_obs

    def get_observation(self) -> Case3Observation:
        if self._last_obs is None:
            raise RuntimeError("Environment has not been reset.")
        return self._build_observation(self._last_obs, self._last_info)

    def step(self, action: np.ndarray) -> StepResult:
        if self._env is None or self._last_obs is None:
            raise RuntimeError("Environment has not been reset.")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim():
            raise ValueError(
                f"Expected action_dim={self.action_dim()}, got shape {action.shape}"
            )

        action_to_env = action.astype(np.float32)
        if self.action_clip is not None:
            action_to_env = np.clip(action_to_env, self.action_clip[0], self.action_clip[1]).astype(np.float32)
        self._current_step += 1
        self._clear_external_forces()
        self._apply_scene_pre_step_effects()
        self._apply_pre_step_faults()

        if self._fault_is_active() and self._fault_type in {"F6_dynamic", "F8_compound"}:
            self._apply_dynamic_box_force()

        obs, reward, terminated, truncated, info = self._env.step(action_to_env)
        done = bool(terminated or truncated)
        self._last_obs = obs
        self._last_info = dict(info)
        self._last_reward = float(reward)

        canonical_obs = self._build_observation(obs, info)
        self._record_progress(canonical_obs, reward=float(reward), action=action_to_env)
        danger = self._compute_danger_channels(canonical_obs, reward=float(reward), done=done)
        return StepResult(
            observation=canonical_obs,
            reward=float(reward),
            done=done,
            info=dict(info),
            danger=danger,
        )

    def canonical_state_dim(self) -> int:
        return 14

    def action_dim(self) -> int:
        if self._env is not None and getattr(self._env.action_space, "shape", None):
            return int(self._env.action_space.shape[0])
        return 14

    def current_step(self) -> int:
        return self._current_step

    # ------------------------------------------------------------------
    # Environment / observation helpers
    # ------------------------------------------------------------------
    def _recreate_env(self) -> None:
        self.close()
        os.environ.setdefault("MUJOCO_GL", "egl")
        import gymnasium as gym  # local import so py_compile does not require the package
        import gym_aloha  # noqa: F401

        self._env = gym.make(
            self.task_id,
            obs_type=self.obs_type,
            render_mode=self.render_mode,
        )

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None

    def _build_observation(self, obs: dict[str, Any], info: dict[str, Any]) -> Case3Observation:
        state = np.asarray(obs["agent_pos"], dtype=np.float32).copy()
        top_image_hwc = np.asarray(obs["pixels"]["top"], dtype=np.uint8).copy()
        top_image_hwc = self._apply_scene_observation_effects(top_image_hwc)

        if self._fault_is_active():
            state, top_image_hwc = self._apply_observation_faults(state, top_image_hwc)

        top_image = np.transpose(top_image_hwc.astype(np.float32) / 255.0, (2, 0, 1))

        physics = self._physics()
        ee_pose = self._extract_ee_pose(physics)
        object_pose = self._extract_object_pose(physics)
        target_pose = self._extract_target_pose(physics)
        gripper_state = state[[6, 13]].copy()
        phase = self._infer_phase(reward=self._last_reward, object_pose=object_pose, info=info)

        return Case3Observation(
            state=state,
            top_image=top_image,
            phase=phase,
            ee_pose=ee_pose,
            object_pose=object_pose,
            target_pose=target_pose,
            gripper_state=gripper_state,
            raw={
                "gym_obs": obs,
                "info": dict(info),
                "scene": self._scene,
                "scene_variant": self._scene_variant_name,
                "fault_type": self._fault_type,
                "fault_profile": self._fault_profile,
                "step": self._current_step,
                "top_image_hwc": top_image_hwc,
            },
        )

    def _physics(self) -> Any:
        if self._env is None:
            raise RuntimeError("Environment is not initialized.")
        return self._env.unwrapped._env.physics

    def _capture_baseline_physics(self) -> None:
        physics = self._physics()
        model = physics.model
        box_body_id = model.body("box").id
        box_geom_id = model.geom("red_box").id
        table_geom_id = model.geom("table").id
        self._base_box_mass = float(model.body_mass[box_body_id])
        self._base_box_friction = np.array(model.geom_friction[box_geom_id], dtype=np.float64)
        self._base_table_friction = np.array(model.geom_friction[table_geom_id], dtype=np.float64)

    def _extract_ee_pose(self, physics: Any) -> np.ndarray:
        xpos = physics.named.data.xpos
        left = np.asarray(xpos["vx300s_left/gripper_link"], dtype=np.float32)
        right = np.asarray(xpos["vx300s_right/gripper_link"], dtype=np.float32)
        return np.concatenate([left, right]).astype(np.float32)

    def _extract_object_pose(self, physics: Any) -> np.ndarray:
        qpos = np.asarray(physics.data.qpos.copy(), dtype=np.float32)
        if qpos.shape[0] < 23:
            return qpos.copy()
        return qpos[16:23].copy()

    def _extract_target_pose(self, physics: Any) -> np.ndarray:
        target = np.asarray(physics.named.data.site_xpos["midair"], dtype=np.float32)
        return target.copy()

    # ------------------------------------------------------------------
    # Fault injection
    # ------------------------------------------------------------------
    def _fault_is_active(self) -> bool:
        if self._fault_type is None:
            return False
        onset = self._fault_onset_step()
        if onset is None:
            return False
        end_step = self._fault_end_step()
        if end_step is None:
            return self._current_step >= onset
        return onset <= self._current_step <= end_step

    def _fault_onset_step(self) -> int | None:
        if self._fault_profile is not None and "onset_step" in self._fault_profile:
            return int(self._fault_profile["onset_step"])
        return self._fault_inject_step

    def _fault_end_step(self) -> int | None:
        if self._fault_profile is not None and "end_step" in self._fault_profile:
            return int(self._fault_profile["end_step"])
        return None

    def _fault_peak_intensity(self) -> float:
        if self._fault_profile is not None and "peak_intensity" in self._fault_profile:
            return float(self._fault_profile["peak_intensity"])
        return 1.0

    def _fault_intensity_profile(self) -> str:
        if self._fault_profile is not None and "intensity_profile" in self._fault_profile:
            return str(self._fault_profile["intensity_profile"])
        return "step"

    def _fault_level(self) -> float:
        if not self._fault_is_active():
            return 0.0
        onset = self._fault_onset_step()
        if onset is None:
            return 0.0
        peak = self._fault_peak_intensity()
        end_step = self._fault_end_step()
        if end_step is None or end_step <= onset:
            return peak
        progress = float(self._current_step - onset) / float(max(1, end_step - onset))
        profile = self._fault_intensity_profile()
        if profile == "ramp":
            return peak * np.clip(progress, 0.0, 1.0)
        if profile == "pulse":
            phase = np.clip(progress, 0.0, 1.0)
            if phase <= 0.4:
                return peak * (phase / 0.4)
            if phase <= 0.75:
                return peak
            return peak * max(0.0, 1.0 - (phase - 0.75) / 0.25)
        return peak

    def _apply_pre_step_faults(self) -> None:
        if not self._fault_is_active():
            return

        physics = self._physics()
        model = physics.model
        box_body_id = model.body("box").id
        box_geom_id = model.geom("red_box").id
        table_geom_id = model.geom("table").id
        level = self._fault_level()

        if self._fault_type in {"F4_payload"}:
            if self._base_box_mass is None:
                raise RuntimeError("Box mass baseline not captured.")
            model.body_mass[box_body_id] = self._base_box_mass * (1.0 + 0.8 * level)
            self._runtime.payload_applied = True

        if self._fault_type in {"F5_friction", "F8_compound"}:
            if self._base_box_friction is None or self._base_table_friction is None:
                raise RuntimeError("Friction baselines not captured.")
            model.geom_friction[box_geom_id] = np.array(
                [
                    max(1e-4, self._base_box_friction[0] * (1.0 - 0.85 * level)),
                    self._base_box_friction[1],
                    self._base_box_friction[2],
                ],
                dtype=np.float64,
            )
            model.geom_friction[table_geom_id] = np.array(
                [
                    max(1e-4, self._base_table_friction[0] * (1.0 - 0.85 * level)),
                    self._base_table_friction[1],
                    self._base_table_friction[2],
                ],
                dtype=np.float64,
            )
            self._runtime.friction_applied = True

    def _clear_external_forces(self) -> None:
        if self._env is None:
            return
        physics = self._physics()
        physics.data.xfrc_applied[:] = 0.0

    def _apply_dynamic_box_force(self) -> None:
        physics = self._physics()
        body_id = physics.model.body("box").id
        level = self._fault_level()
        # Apply a lateral push and a slight upward/downward disturbance.
        force = np.array([2.5, -1.0, 0.4, 0.0, 0.0, 0.0], dtype=np.float64) * max(level, 0.1)
        physics.data.xfrc_applied[body_id] = force

    def _apply_scene_pre_step_effects(self) -> None:
        if self._scene != "dynamic_disturbance":
            return
        start_step = int(self._scene_config.get("dynamic_start_step", 15))
        if self._current_step < start_step:
            return
        period = max(1, int(self._scene_config.get("dynamic_period", 6)))
        active_mods = tuple(self._scene_config.get("dynamic_active_mods", (0, 1)))
        if self._current_step % period not in set(active_mods):
            return
        physics = self._physics()
        body_id = physics.model.body("box").id
        force = np.asarray(
            self._scene_config.get("dynamic_force", (1.2, 0.6, 0.0, 0.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        physics.data.xfrc_applied[body_id] = force

    def _apply_scene_observation_effects(self, top_image: np.ndarray) -> np.ndarray:
        perception_enabled = self._scene == "perception_stressed" or bool(
            self._scene_config.get("mixed_enable_perception", False)
        )
        if not perception_enabled:
            return top_image
        image = top_image.astype(np.float32)
        image = image * float(self._scene_config.get("perception_dim_scale", 0.78))
        h, w = image.shape[:2]
        yy, xx = np.indices((h, w))
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        radius = np.sqrt(((yy - cy) / max(cy, 1.0)) ** 2 + ((xx - cx) / max(cx, 1.0)) ** 2)
        vignette_strength = float(self._scene_config.get("perception_vignette_strength", 0.25))
        vignette_floor = float(self._scene_config.get("perception_vignette_floor", 0.65))
        vignette = np.clip(1.0 - vignette_strength * radius, vignette_floor, 1.0)[..., None]
        image = image * vignette
        return np.clip(image, 0, 255).astype(np.uint8)

    def _apply_observation_faults(
        self,
        state: np.ndarray,
        top_image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = state.copy()
        image = top_image.copy()
        level = self._fault_level()

        if self._fault_type == "F1_lighting":
            scale = max(0.15, 1.0 - 0.65 * level)
            image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        elif self._fault_type == "F2_occlusion":
            h, w = image.shape[:2]
            occ_h = int(h * (0.18 + 0.35 * level))
            occ_w = int(w * (0.20 + 0.28 * level))
            y0 = max(0, h // 2 - occ_h // 2)
            y1 = min(h, y0 + occ_h)
            x0 = max(0, w // 2 - occ_w // 2)
            x1 = min(w, x0 + occ_w)
            image[y0:y1, x0:x1] = 0
        elif self._fault_type == "F3_adversarial":
            image = self._apply_checkerboard_overlay(image, alpha=0.15 + 0.40 * level)
        elif self._fault_type == "F7_sensor":
            state = state + np.random.normal(0.0, 0.02 + 0.08 * level, size=state.shape).astype(np.float32)
        elif self._fault_type == "F8_compound":
            image = np.clip(image.astype(np.float32) * max(0.2, 1.0 - 0.55 * level), 0, 255).astype(np.uint8)
            state = state + np.random.normal(0.0, 0.015 + 0.06 * level, size=state.shape).astype(np.float32)

        return state.astype(np.float32), image

    @staticmethod
    def _apply_checkerboard_overlay(image: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        h, w = image.shape[:2]
        yy, xx = np.indices((h, w))
        pattern = ((yy // 16 + xx // 16) % 2).astype(np.float32)
        pattern = (pattern * 255.0)[..., None]
        mixed = (1.0 - alpha) * image.astype(np.float32) + alpha * pattern
        return np.clip(mixed, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Phase / danger logic
    # ------------------------------------------------------------------
    def _infer_phase(self, *, reward: float, object_pose: np.ndarray, info: dict[str, Any]) -> str:
        success = bool(info.get("is_success", False))
        object_z = float(object_pose[2]) if object_pose.shape[0] >= 3 else 0.0
        target_z = float(self._extract_target_pose(self._physics())[2])

        if success:
            return "transfer"
        if reward >= 3:
            return "transfer"
        if reward >= 2 or object_z > target_z * 0.6:
            return "lift"
        if reward >= 1:
            return "grasp"
        if self._max_reward_seen >= 2 and reward <= 0:
            return "recovery"
        return "approach"

    def _record_progress(
        self,
        obs: Case3Observation,
        *,
        reward: float,
        action: np.ndarray | None,
    ) -> None:
        self._reward_history.append(float(reward))
        self._max_reward_seen = max(self._max_reward_seen, float(reward))
        if float(reward) >= 2.0:
            self._lift_seen = True
        if action is not None:
            self._action_history.append(np.asarray(action, dtype=np.float32))
        if obs.object_pose is not None and obs.target_pose is not None:
            distance = float(np.linalg.norm(obs.object_pose[:3] - obs.target_pose[:3]))
            self._box_target_distance_history.append(distance)

    def _compute_danger_channels(
        self,
        obs: Case3Observation,
        *,
        reward: float,
        done: bool,
    ) -> DangerChannels:
        contact_pairs = self._contact_pairs()
        hard_details = self._hard_safety_details(contact_pairs)
        task_details = self._task_failure_details(obs, reward, done, contact_pairs)
        persistent_details = self._persistent_instability_details()

        details = {
            **hard_details,
            **task_details,
            **persistent_details,
            "reward": float(reward),
            "max_reward_seen": float(self._max_reward_seen),
            "step": self._current_step,
            "fault_type": self._fault_type,
            "scene": self._scene,
            "scene_variant": self._scene_variant_name,
        }
        return DangerChannels(
            hard_safety=hard_details["hard_safety"],
            task_critical_failure=task_details["task_critical_failure"],
            persistent_instability=persistent_details["persistent_instability"],
            details=details,
        )

    def _contact_pairs(self) -> list[tuple[str, str]]:
        physics = self._physics()
        pairs: list[tuple[str, str]] = []
        for i_contact in range(int(physics.data.ncon)):
            c = physics.data.contact[i_contact]
            g1 = physics.model.id2name(int(c.geom1), "geom")
            g2 = physics.model.id2name(int(c.geom2), "geom")
            pairs.append((g1, g2))
        return pairs

    def _hard_safety_details(self, contact_pairs: list[tuple[str, str]]) -> dict[str, Any]:
        robot_table_contacts: list[tuple[str, str]] = []
        robot_robot_contacts: list[tuple[str, str]] = []
        unexpected_box_contacts: list[tuple[str, str]] = []

        for g1, g2 in contact_pairs:
            pair = {g1, g2}
            if "table" in pair and any(name.startswith("vx300s_") for name in pair):
                robot_table_contacts.append((g1, g2))
            if g1.startswith("vx300s_left") and g2.startswith("vx300s_right"):
                robot_robot_contacts.append((g1, g2))
            if "red_box" in pair:
                other = g2 if g1 == "red_box" else g1
                allowed = {
                    "table",
                    "vx300s_left/10_left_gripper_finger",
                    "vx300s_left/10_right_gripper_finger",
                    "vx300s_right/10_left_gripper_finger",
                    "vx300s_right/10_right_gripper_finger",
                }
                if other not in allowed:
                    unexpected_box_contacts.append((g1, g2))

        hard_safety = bool(robot_table_contacts or robot_robot_contacts or unexpected_box_contacts)
        return {
            "hard_safety": hard_safety,
            "robot_table_contacts": robot_table_contacts,
            "robot_robot_contacts": robot_robot_contacts,
            "unexpected_box_contacts": unexpected_box_contacts,
        }

    def _task_failure_details(
        self,
        obs: Case3Observation,
        reward: float,
        done: bool,
        contact_pairs: list[tuple[str, str]],
    ) -> dict[str, Any]:
        object_pose = obs.object_pose if obs.object_pose is not None else np.zeros(7, dtype=np.float32)
        target_pose = obs.target_pose if obs.target_pose is not None else np.zeros(3, dtype=np.float32)
        object_z = float(object_pose[2]) if object_pose.shape[0] >= 3 else 0.0
        target_distance = (
            float(np.linalg.norm(object_pose[:3] - target_pose[:3]))
            if object_pose.shape[0] >= 3 and target_pose.shape[0] >= 3
            else float("nan")
        )

        box_on_table = any({"red_box", "table"} == {g1, g2} for g1, g2 in contact_pairs)
        grasp_loss = self._max_reward_seen >= 1.0 and reward <= 0.0
        object_drop = self._lift_seen and box_on_table and reward <= 1.0
        timeout_failure = done and not bool(self._last_info.get("is_success", False)) and reward < 4.0
        severe_misalignment = (
            not np.isnan(target_distance)
            and self._current_step > 50
            and self._max_reward_seen >= 2.0
            and target_distance > 0.18
        )

        task_critical_failure = bool(grasp_loss or object_drop or timeout_failure or severe_misalignment)
        return {
            "task_critical_failure": task_critical_failure,
            "grasp_loss": grasp_loss,
            "object_drop": object_drop,
            "timeout_failure": timeout_failure,
            "severe_misalignment": severe_misalignment,
            "box_on_table": box_on_table,
            "object_z": object_z,
            "target_distance": target_distance,
        }

    def _persistent_instability_details(self) -> dict[str, Any]:
        no_progress = False
        oscillatory_action = False
        worsening_target_distance = False

        if len(self._reward_history) >= self.no_progress_window:
            recent_rewards = self._reward_history[-self.no_progress_window :]
            no_progress = max(recent_rewards) <= 1.0

        if len(self._action_history) >= 4:
            recent = np.stack(self._action_history[-4:], axis=0)
            deltas = np.diff(recent, axis=0)
            sign_changes = np.sum(np.sign(deltas[1:]) != np.sign(deltas[:-1]))
            oscillatory_action = bool(sign_changes >= max(4, deltas.shape[1] // 2))

        if len(self._box_target_distance_history) >= self.no_progress_window:
            recent_dist = self._box_target_distance_history[-self.no_progress_window :]
            worsening_target_distance = recent_dist[-1] > recent_dist[0] + 0.03

        persistent_instability = bool(no_progress or oscillatory_action or worsening_target_distance)
        return {
            "persistent_instability": persistent_instability,
            "no_progress": no_progress,
            "oscillatory_action": oscillatory_action,
            "worsening_target_distance": worsening_target_distance,
        }

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_scene(self, scene: str) -> None:
        if scene == "static_clutter":
            raise NotImplementedError(
                "gym-aloha transfer-cube does not currently provide a real static-clutter variant."
            )
        if scene not in self.SUPPORTED_SCENES:
            raise ValueError(f"Unsupported scene: {scene}")

    def _resolve_scene_variant(self, scene: str) -> str:
        self._scene_config = {}
        if not is_scene_variant(scene):
            return scene
        spec = get_scene_variant(scene)
        self._scene_config = dict(spec.adapter_overrides)
        if not spec.supported:
            raise NotImplementedError(f"Scene variant is registered but not yet supported: {scene}")
        return spec.family

    def _validate_fault(self, fault_type: str | None) -> None:
        if fault_type not in self.SUPPORTED_FAULTS:
            raise ValueError(f"Unsupported fault type: {fault_type}")
