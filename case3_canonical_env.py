#!/usr/bin/env python3
"""
Canonical environment / observation interfaces for ACT-based Case 3.

Purpose:
1. Prevent the final Case 3 collector from silently falling back to synthetic
   images or toy-only state updates.
2. Make the required information flow explicit for R1/R2/R3 reconstruction.
3. Provide a single adapter contract for future real simulator integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Case3Observation:
    """
    Canonical ACT-facing observation for Case 3.

    Notes:
    - `state` should be the structured robot/task state used by ACT-side
      attribution and by R2.
    - `top_image` must come from a real camera/simulator render path.
    - `top_image` should be ACT-facing channel-first `[C, H, W]`.
    - additional fields support richer GT and per-phase analysis.
    """

    state: np.ndarray
    top_image: np.ndarray
    phase: str
    ee_pose: np.ndarray | None = None
    object_pose: np.ndarray | None = None
    target_pose: np.ndarray | None = None
    gripper_state: np.ndarray | None = None
    raw: dict[str, Any] | None = None


@dataclass
class DangerChannels:
    hard_safety: bool
    task_critical_failure: bool
    persistent_instability: bool
    details: dict[str, Any]

    @property
    def any_danger(self) -> bool:
        return bool(
            self.hard_safety or self.task_critical_failure or self.persistent_instability
        )


@dataclass
class StepResult:
    observation: Case3Observation
    reward: float
    done: bool
    info: dict[str, Any]
    danger: DangerChannels


class Case3EnvAdapter(ABC):
    """
    Required adapter for the canonical Case 3 collector.

    The final collector should depend on this interface instead of directly
    depending on a simplified environment implementation.
    """

    @abstractmethod
    def reset(
        self,
        *,
        scene: str,
        fault_type: str | None,
        fault_inject_step: int | None,
        fault_profile: dict[str, Any] | None,
        seed: int,
    ) -> Case3Observation:
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> Case3Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def canonical_state_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def action_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def current_step(self) -> int:
        raise NotImplementedError
