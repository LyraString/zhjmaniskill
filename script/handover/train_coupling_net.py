from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_DATA_PATH = "/home/idiot/zhj/RoboTwin/tmp/handover/counterfactual_labels_100ep.npz"
DEFAULT_OUTPUT_PATH = "/home/idiot/zhj/RoboTwin/tmp/handover/coupling_net_v0.pt"
TARGET_NAMES = ("left", "right", "both")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an offline CouplingNet on counterfactual labels.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--device", type=str, default="auto")
    return parser


class CouplingNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(TARGET_NAMES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    features = data["features"].astype(np.float32)
    labels = np.stack(
        [
            data["left_label"].astype(np.float32),
            data["right_label"].astype(np.float32),
            data["both_label"].astype(np.float32),
        ],
        axis=1,
    )
    return {
        "features": features,
        "labels": labels,
        "stage": data["stage"].astype(np.int64),
        "episode": data["episode"].astype(np.int64),
        "feature_names": data["feature_names"].astype(str),
    }


def _episode_split(episodes: np.ndarray, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_episodes = np.unique(episodes)
    if unique_episodes.size < 2:
        raise ValueError("Need at least two unique episodes for episode-level train/val split.")
    rng = np.random.default_rng(seed)
    shuffled = unique_episodes.copy()
    rng.shuffle(shuffled)
    train_count = int(round(float(train_frac) * shuffled.size))
    train_count = min(max(train_count, 1), shuffled.size - 1)
    train_episodes = set(shuffled[:train_count].tolist())
    train_mask = np.asarray([episode in train_episodes for episode in episodes], dtype=bool)
    val_mask = ~train_mask
    return train_mask, val_mask


def _normalize_features(
    features: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[train_mask].mean(axis=0, keepdims=True)
    std = features[train_mask].std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    normalized = (features - mean) / std
    return normalized.astype(np.float32), mean.reshape(-1).astype(np.float32), std.reshape(-1).astype(np.float32)


def _pos_weight(labels: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train_labels = labels[train_mask]
    pos = train_labels.sum(axis=0)
    neg = train_labels.shape[0] - pos
    weight = np.ones_like(pos, dtype=np.float32)
    valid = pos > 0
    weight[valid] = (neg[valid] / np.maximum(pos[valid], 1.0)).astype(np.float32)
    return weight


def _make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x = torch.from_numpy(features[mask]).to(torch.float32)
    y = torch.from_numpy(labels[mask]).to(torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + 1 + end)
        start = end

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    pos_rank_sum = float(original_ranks[pos].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _target_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    preds = scores >= 0.5
    truth = labels == 1

    tp = float(np.logical_and(preds, truth).sum())
    tn = float(np.logical_and(~preds, ~truth).sum())
    fp = float(np.logical_and(preds, ~truth).sum())
    fn = float(np.logical_and(~preds, truth).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": _binary_auc(scores, labels),
        "positive_frac": float(truth.mean()) if truth.size else float("nan"),
        "pred_positive_frac": float(preds.mean()) if preds.size else float("nan"),
    }


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    criterion: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    stage: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    x = torch.from_numpy(features[mask]).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(labels[mask]).to(device=device, dtype=torch.float32)
    logits = model(x)
    loss = float(criterion(logits, y).item())
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_np = y.detach().cpu().numpy().astype(np.int64)
    stage_np = stage[mask]

    targets = {
        name: _target_metrics(probs[:, idx], y_np[:, idx])
        for idx, name in enumerate(TARGET_NAMES)
    }
    macro_f1 = float(np.nanmean([target["f1"] for target in targets.values()]))
    macro_auc = float(np.nanmean([target["auc"] for target in targets.values()]))

    stage_metrics = {}
    for stage_id in sorted(np.unique(stage_np).tolist()):
        stage_mask = stage_np == stage_id
        if not stage_mask.any():
            continue
        stage_targets = {
            name: _target_metrics(probs[stage_mask, idx], y_np[stage_mask, idx])
            for idx, name in enumerate(TARGET_NAMES)
        }
        stage_metrics[str(int(stage_id))] = {
            "n": int(stage_mask.sum()),
            "macro_f1": float(np.nanmean([target["f1"] for target in stage_targets.values()])),
            "macro_auc": float(np.nanmean([target["auc"] for target in stage_targets.values()])),
            "targets": stage_targets,
        }

    return {
        "loss": loss,
        "n": int(mask.sum()),
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "targets": targets,
        "stage_metrics": stage_metrics,
    }


def train(args) -> dict[str, Any]:
    _set_seed(int(args.seed))
    device = _resolve_device(args.device)
    dataset = _load_dataset(Path(args.data_path))
    features = dataset["features"]
    labels = dataset["labels"]
    stage = dataset["stage"]
    episodes = dataset["episode"]
    feature_names = dataset["feature_names"]

    if features.shape[0] == 0:
        raise ValueError("Dataset is empty.")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Feature/label length mismatch.")

    train_mask, val_mask = _episode_split(episodes, float(args.train_frac), int(args.seed))
    normalized_features, feature_mean, feature_std = _normalize_features(features, train_mask)
    pos_weight = _pos_weight(labels, train_mask)

    model = CouplingNet(input_dim=features.shape[1], hidden_dim=int(args.hidden_dim)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight).to(device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_loader = _make_loader(
        normalized_features,
        labels,
        train_mask,
        batch_size=int(args.batch_size),
        shuffle=True,
    )

    history = []
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_count = int(x.shape[0])
            total_loss += float(loss.item()) * batch_count
            total_count += batch_count

        train_eval = _evaluate(model, criterion, normalized_features, labels, stage, train_mask, device)
        val_eval = _evaluate(model, criterion, normalized_features, labels, stage, val_mask, device)
        val_macro_f1 = float(val_eval["macro_f1"])
        val_loss = float(val_eval["loss"])
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            best_epoch = int(epoch)
            best_state_dict = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch == int(args.epochs) or epoch % max(1, int(args.epochs) // 10) == 0:
            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_eval["loss"]),
                    "val_loss": float(val_eval["loss"]),
                    "train_macro_f1": float(train_eval["macro_f1"]),
                    "val_macro_f1": float(val_eval["macro_f1"]),
                    "train_macro_auc": float(train_eval["macro_auc"]),
                    "val_macro_auc": float(val_eval["macro_auc"]),
                }
            )

    model.load_state_dict(best_state_dict)
    train_metrics = _evaluate(model, criterion, normalized_features, labels, stage, train_mask, device)
    val_metrics = _evaluate(model, criterion, normalized_features, labels, stage, val_mask, device)

    summary = {
        "data_path": str(args.data_path),
        "output_path": str(args.output_path),
        "feature_dim": int(features.shape[1]),
        "sample_count": int(features.shape[0]),
        "train_count": int(train_mask.sum()),
        "val_count": int(val_mask.sum()),
        "unique_episode_count": int(np.unique(episodes).size),
        "train_episode_count": int(np.unique(episodes[train_mask]).size),
        "val_episode_count": int(np.unique(episodes[val_mask]).size),
        "target_names": list(TARGET_NAMES),
        "pos_weight": pos_weight.tolist(),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": str(device),
        "selected_epoch": int(best_epoch),
        "selected_val_macro_f1": float(best_val_macro_f1),
        "selected_val_loss": float(best_val_loss),
        "history": history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": feature_names.tolist(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_names": list(TARGET_NAMES),
            "hidden_dim": int(args.hidden_dim),
            "summary": summary,
        },
        output_path,
    )
    return summary


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    summary = train(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
