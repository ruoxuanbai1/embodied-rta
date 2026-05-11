#!/usr/bin/env python3
"""
Train a compact Case 3 Region-2 predictor from canonical rollout NPZ files.

R2 target in this recovery line:
- input: short history of state + action
- output: probability of entering any danger state within a short future window

This is not the original large-scale reachability model, but it restores the
predictive layer on top of the newly recovered real-env data path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from case3_paired_degradation import (
    build_normal_map,
    build_paired_degradation_sequence,
    filter_runs_by_split,
    load_rollout_arrays,
    onset_from_binary_signal,
)


@dataclass
class TrainStats:
    n_train: int
    n_val: int
    positive_rate_train: float
    positive_rate_val: float
    best_val_loss: float
    best_val_auc_proxy: float


class SequenceHazardDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class HazardGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        logits = self.head(hidden[-1]).squeeze(-1)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--future-horizon", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--label-mode",
        choices=["any_danger", "hard_or_task_onset", "paired_degradation_onset"],
        default="paired_degradation_onset",
    )
    parser.add_argument("--onset-lookback", type=int, default=3)
    parser.add_argument("--reward-gap-threshold", type=float, default=1.0)
    parser.add_argument("--degradation-sustain", type=int, default=5)
    parser.add_argument("--splits", nargs="+", default=["train"])
    return parser.parse_args()


def load_npz_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*.npz"))


def load_runs(npz_files: list[Path]) -> list[dict[str, Any]]:
    return [load_rollout_arrays(path) for path in npz_files]


def _compute_targets(
    *,
    danger_any: np.ndarray,
    danger_hard: np.ndarray,
    danger_task: np.ndarray,
    label_mode: str,
    onset_lookback: int,
) -> np.ndarray:
    if label_mode == "any_danger":
        return danger_any.astype(np.float32)

    if label_mode != "hard_or_task_onset":
        raise ValueError(f"Unsupported label mode: {label_mode}")

    target_signal = np.logical_or(danger_hard > 0.5, danger_task > 0.5).astype(np.uint8)
    onset = np.zeros_like(target_signal, dtype=np.float32)
    last_positive_end = -10**9
    for idx, current in enumerate(target_signal):
        if current <= 0:
            continue
        prev_start = max(0, idx - onset_lookback)
        prev_active = np.any(target_signal[prev_start:idx] > 0)
        if not prev_active and idx > last_positive_end:
            onset[idx] = 1.0
        if idx + 1 >= len(target_signal) or target_signal[idx + 1] <= 0:
            last_positive_end = idx
    return onset


def _onset_from_binary_signal(signal: np.ndarray, onset_lookback: int) -> np.ndarray:
    return onset_from_binary_signal(signal, onset_lookback)


def build_samples(
    runs: list[dict[str, Any]],
    history_len: int,
    future_horizon: int,
    label_mode: str,
    onset_lookback: int,
    reward_gap_threshold: float,
    degradation_sustain: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    normal_map = build_normal_map(runs)

    for run in runs:
        states = run["states"]
        actions = run["actions"]
        danger_any = run["danger_any"]
        danger_hard = run["danger_hard"]
        danger_task = run["danger_task"]
        if len(states) <= history_len:
            continue

        if label_mode == "paired_degradation_onset":
            degradation_seq, _ = build_paired_degradation_sequence(
                run=run,
                normal_map=normal_map,
                reward_gap_threshold=reward_gap_threshold,
                degradation_sustain=degradation_sustain,
            )
            if degradation_seq is None:
                target_steps = np.zeros_like(run["rewards"], dtype=np.float32)
            else:
                target_steps = onset_from_binary_signal(degradation_seq.astype(np.uint8), onset_lookback)
        else:
            target_steps = _compute_targets(
                danger_any=danger_any,
                danger_hard=danger_hard,
                danger_task=danger_task,
                label_mode=label_mode,
                onset_lookback=onset_lookback,
            )
        features = np.concatenate([states, actions], axis=1)
        for end_idx in range(history_len, len(features)):
            start_idx = end_idx - history_len
            seq = features[start_idx:end_idx]
            future_end = min(len(target_steps), end_idx + future_horizon)
            label = float(np.any(target_steps[end_idx:future_end] > 0.5))
            sequences.append(seq)
            labels.append(label)

    if not sequences:
        raise RuntimeError("No training samples could be built from the dataset.")

    x = np.stack(sequences, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return x, y


def split_train_val(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def auc_proxy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    pos = probs[y > 0.5]
    neg = probs[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    comparisons = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = load_npz_files(data_dir)
    runs = filter_runs_by_split(load_runs(npz_files), args.splits)
    x, y = build_samples(
        runs,
        args.history_len,
        args.future_horizon,
        args.label_mode,
        args.onset_lookback,
        args.reward_gap_threshold,
        args.degradation_sustain,
    )
    x_train, y_train, x_val, y_val = split_train_val(x, y, args.val_ratio, args.seed)

    train_ds = SequenceHazardDataset(x_train, y_train)
    val_ds = SequenceHazardDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = HazardGRU(
        input_dim=x.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    pos_weight_value = float(max(1.0, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_val_auc = 0.5
    best_path = output_dir / "r2_hazard_gru.pt"

    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses: list[float] = []
        val_logits_all: list[torch.Tensor] = []
        val_y_all: list[torch.Tensor] = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_losses.append(float(loss.item()))
                val_logits_all.append(logits)
                val_y_all.append(batch_y)

        val_loss = float(np.mean(val_losses))
        val_logits = torch.cat(val_logits_all, dim=0)
        val_targets = torch.cat(val_y_all, dim=0)
        val_auc = auc_proxy(val_logits, val_targets)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history_len": args.history_len,
                    "future_horizon": args.future_horizon,
                    "input_dim": x.shape[-1],
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "label_mode": args.label_mode,
                    "onset_lookback": args.onset_lookback,
                    "reward_gap_threshold": args.reward_gap_threshold,
                    "degradation_sustain": args.degradation_sustain,
                    "splits": list(args.splits),
                    "val_loss": val_loss,
                    "val_auc_proxy": val_auc,
                },
                best_path,
            )

    stats = TrainStats(
        n_train=len(train_ds),
        n_val=len(val_ds),
        positive_rate_train=float(np.mean(y_train)),
        positive_rate_val=float(np.mean(y_val)),
        best_val_loss=best_val_loss,
        best_val_auc_proxy=best_val_auc,
    )
    (output_dir / "r2_hazard_gru_stats.json").write_text(
        json.dumps(asdict(stats), indent=2),
        encoding="utf-8",
    )
    print(best_path)


if __name__ == "__main__":
    train(parse_args())
