#!/usr/bin/env python3
"""
Compatibility loader for ACT checkpoints trained with an older LeRobot source tree.

Purpose:
1. Reuse the currently working `act_case3` runtime environment.
2. Load the old checkpoint-trained ACT implementation directly from a source tree.
3. Filter known config-key drift (`device`, `use_amp`, `type`) that breaks
   older `draccus` config parsing.

This avoids maintaining a separate fully-installed legacy environment when the
main incompatibility is isolated to the policy source and config schema.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download


INCOMPATIBLE_CONFIG_KEYS = ("device", "use_amp", "type")


def _prepend_source_root(source_root: str | Path) -> None:
    root = str(Path(source_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_legacy_act_policy_from_source(
    *,
    model_id: str,
    legacy_source_root: str | Path,
    device: torch.device | str = "cpu",
) -> Any:
    """
    Load an ACT checkpoint with the older LeRobot ACT implementation.

    Notes:
    - This uses the current Python environment, but imports `lerobot` from the
      provided source tree by putting it at the front of `sys.path`.
    - The checkpoint config from the Hub is filtered to remove keys that were
      introduced after the older ACTConfig schema.
    """

    _prepend_source_root(legacy_source_root)
    from lerobot.common.policies.act.configuration_act import ACTConfig  # type: ignore
    from lerobot.common.policies.act.modeling_act import ACTPolicy  # type: ignore

    cfg_path = hf_hub_download(model_id, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in INCOMPATIBLE_CONFIG_KEYS:
        cfg.pop(key, None)

    with tempfile.TemporaryDirectory() as td:
        tmp_cfg = Path(td) / "config.json"
        tmp_cfg.write_text(json.dumps(cfg), encoding="utf-8")
        config = ACTConfig.from_pretrained(td)
        policy = ACTPolicy.from_pretrained(model_id, config=config, map_location=str(device))
    policy.eval()
    return policy


def select_action_with_legacy_policy(
    *,
    policy: Any,
    state: np.ndarray,
    top_image: np.ndarray,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """
    Run one ACT action step through a legacy-source policy.

    Expected inputs:
    - `state`: raw structured state, shape `(14,)`
    - `top_image`: ACT-facing channel-first float image, shape `(3,H,W)`
    """

    device = torch.device(device)
    batch = {
        "observation.images.top": torch.as_tensor(top_image, dtype=torch.float32, device=device).unsqueeze(0),
        "observation.state": torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
    }
    with torch.no_grad():
        action = policy.select_action(batch)
    return action[0].detach().cpu().numpy().astype(np.float32)
