#!/usr/bin/env python3
"""
R3 signal extraction helpers for the legacy-source ACT compatibility path.

These utilities mirror the current canonical R3 helpers, but avoid importing
the current `lerobot` package implementation directly. They operate on the
older source-loaded policy object used by `case3_act_legacy_source_loader.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


OBS_IMAGES_KEY = "observation.images"


@dataclass
class LegacyRoutingCache:
    decoder_last_ffn: torch.Tensor | None = None
    action_head_input: torch.Tensor | None = None
    action_head_output: torch.Tensor | None = None


def register_legacy_canonical_routing_hooks(policy: Any, cache: LegacyRoutingCache) -> list[Any]:
    handles: list[Any] = []

    def save_decoder(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        cache.decoder_last_ffn = output.detach()

    def save_action_head_input(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        cache.action_head_input = None if not inputs else inputs[0].detach()

    def save_action_head_output(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        cache.action_head_output = output.detach()

    handles.append(policy.model.decoder.layers[-1].linear2.register_forward_hook(save_decoder))
    handles.append(policy.model.action_head.register_forward_pre_hook(save_action_head_input))
    handles.append(policy.model.action_head.register_forward_hook(save_action_head_output))
    return handles


def _legacy_batch(
    *,
    state_t: torch.Tensor,
    image_t: torch.Tensor,
    policy: Any,
) -> dict[str, Any]:
    image_batch = image_t.unsqueeze(1)
    return {
        "observation.state": state_t,
        "observation.images.top": image_t,
        "observation.images": image_batch,
    }


def _first_action_from_legacy_core(policy: Any, batch: dict[str, Any]) -> torch.Tensor:
    try:
        actions = policy.model(batch)[0]
    except Exception:
        image_features = getattr(getattr(policy, "config", None), "image_features", None)
        if not image_features:
            raise
        fallback_batch = dict(batch)
        fallback_batch[OBS_IMAGES_KEY] = [fallback_batch[key] for key in image_features]
        actions = policy.model(fallback_batch)[0]
    if actions.ndim != 3:
        raise ValueError(f"Unexpected legacy ACT core action shape: {tuple(actions.shape)}")
    return actions[:, 0, :]


def compute_first_action_state_attribution_with_legacy_policy(
    *,
    policy: Any,
    state: np.ndarray,
    top_image: np.ndarray,
    device: torch.device | str = "cpu",
    scalar_mode: str = "l1",
) -> dict[str, np.ndarray | float]:
    device = torch.device(device)
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    state_t.requires_grad_(True)
    image_t = torch.as_tensor(top_image, dtype=torch.float32, device=device).unsqueeze(0)
    batch = _legacy_batch(state_t=state_t, image_t=image_t, policy=policy)

    policy.zero_grad(set_to_none=True)
    first_action = _first_action_from_legacy_core(policy, batch)

    if scalar_mode == "l1":
        objective = first_action.abs().sum()
    elif scalar_mode == "l2":
        objective = first_action.square().sum()
    elif scalar_mode == "signed_sum":
        objective = first_action.sum()
    else:
        raise ValueError(f"Unsupported scalar_mode: {scalar_mode}")

    objective.backward()
    grad = state_t.grad
    if grad is None:
        raise RuntimeError("Legacy ACT gradient is None; autograd path did not propagate.")

    grad_np = grad.detach().cpu().numpy()[0].astype(np.float32)
    state_np = state_t.detach().cpu().numpy()[0].astype(np.float32)
    phi_np = (state_np * grad_np).astype(np.float32)
    action_np = first_action.detach().cpu().numpy()[0].astype(np.float32)
    return {
        "action": action_np,
        "gradient": grad_np,
        "phi": phi_np,
        "objective_value": float(objective.detach().cpu().item()),
    }


def pool_route_feature(tensor: torch.Tensor | None) -> np.ndarray | None:
    if tensor is None:
        return None
    x = tensor[0].detach().to(dtype=torch.float32).cpu()
    while x.ndim > 1:
        x = x.mean(dim=0)
    return x.numpy().astype(np.float32).reshape(-1)
