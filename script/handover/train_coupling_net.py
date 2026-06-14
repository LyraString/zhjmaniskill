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
DEFAULT_OUTPUT_PATH = "/home/idiot/zhj/RoboTwin/tmp/handover/coupling_net_scalar_v0.pt"
TARGET_NAME = "coupling_score"
DEFAULT_DEV_SCALE = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train scalar CouplingNet on counterfactual sensitivity.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--coupling-dev-scale", type=float, default=DEFAULT_DEV_SCALE)
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
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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


def _load_dataset(path: Path, dev_scale: float) -> dict[str, np.ndarray]:
    data = np.load(path)
    features = data["features"].astype(np.float32)
    left_dev = data["left_dev"].astype(np.float32)
    right_dev = data["right_dev"].astype(np.float32)
    both_dev = data["both_dev"].astype(np.float32)
    max_dev = np.maximum.reduce([left_dev, right_dev, both_dev])
    target = np.clip(max_dev / float(dev_scale), 0.0, 1.0).astype(np.float32)
    return {
        "features": features,
        "target": target,
        "max_dev": max_dev.astype(np.float32),
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


def _make_loader(
    features: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x = torch.from_numpy(features[mask]).to(torch.float32)
    y = torch.from_numpy(target[mask]).to(torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def _corr(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.size < 2:
        return float("nan")
    pred_std = float(pred.std())
    target_std = float(target.std())
    if pred_std < 1e-12 or target_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])


def _regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson_corr": float("nan"),
            "target_mean": float("nan"),
            "pred_mean": float("nan"),
            "target_std": float("nan"),
            "pred_std": float("nan"),
        }
    err = pred - target
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "pearson_corr": _corr(pred, target),
        "target_mean": float(target.mean()),
        "pred_mean": float(pred.mean()),
        "target_std": float(target.std()),
        "pred_std": float(pred.std()),
    }


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    criterion: nn.Module,
    features: np.ndarray,
    target: np.ndarray,
    stage: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    x = torch.from_numpy(features[mask]).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(target[mask]).to(device=device, dtype=torch.float32)
    pred = torch.sigmoid(model(x))
    loss = float(criterion(pred, y).item())
    pred_np = pred.detach().cpu().numpy()
    target_np = y.detach().cpu().numpy()
    stage_np = stage[mask]

    metrics = _regression_metrics(pred_np, target_np)
    stage_metrics = {}
    for stage_id in sorted(np.unique(stage_np).tolist()):
        stage_mask = stage_np == stage_id
        if not stage_mask.any():
            continue
        stage_metrics[str(int(stage_id))] = {
            "n": int(stage_mask.sum()),
            **_regression_metrics(pred_np[stage_mask], target_np[stage_mask]),
        }

    return {
        "loss": loss,
        "n": int(mask.sum()),
        **metrics,
        "stage_metrics": stage_metrics,
    }


def train(args) -> dict[str, Any]:
    _set_seed(int(args.seed))
    device = _resolve_device(args.device)
    dataset = _load_dataset(Path(args.data_path), float(args.coupling_dev_scale))
    features = dataset["features"]
    target = dataset["target"]
    max_dev = dataset["max_dev"]
    stage = dataset["stage"]
    episodes = dataset["episode"]
    feature_names = dataset["feature_names"]

    if features.shape[0] == 0:
        raise ValueError("Dataset is empty.")
    if features.shape[0] != target.shape[0]:
        raise ValueError("Feature/target length mismatch.")

    train_mask, val_mask = _episode_split(episodes, float(args.train_frac), int(args.seed))
    normalized_features, feature_mean, feature_std = _normalize_features(features, train_mask)

    model = CouplingNet(input_dim=features.shape[1], hidden_dim=int(args.hidden_dim)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_loader = _make_loader(
        normalized_features,
        target,
        train_mask,
        batch_size=int(args.batch_size),
        shuffle=True,
    )

    history = []
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_mae = float("inf")
    best_val_loss = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            pred = torch.sigmoid(model(x))
            loss = criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_eval = _evaluate(model, criterion, normalized_features, target, stage, train_mask, device)
        val_eval = _evaluate(model, criterion, normalized_features, target, stage, val_mask, device)
        val_mae = float(val_eval["mae"])
        val_loss = float(val_eval["loss"])
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_loss = val_loss
            best_epoch = int(epoch)
            best_state_dict = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch == int(args.epochs) or epoch % max(1, int(args.epochs) // 10) == 0:
            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_eval["loss"]),
                    "val_loss": float(val_eval["loss"]),
                    "train_mae": float(train_eval["mae"]),
                    "val_mae": float(val_eval["mae"]),
                    "train_rmse": float(train_eval["rmse"]),
                    "val_rmse": float(val_eval["rmse"]),
                    "train_corr": float(train_eval["pearson_corr"]),
                    "val_corr": float(val_eval["pearson_corr"]),
                }
            )

    model.load_state_dict(best_state_dict)
    train_metrics = _evaluate(model, criterion, normalized_features, target, stage, train_mask, device)
    val_metrics = _evaluate(model, criterion, normalized_features, target, stage, val_mask, device)

    train_mean_baseline = float(np.mean(target[train_mask]))
    val_baseline_pred = np.full_like(target[val_mask], fill_value=train_mean_baseline, dtype=np.float32)
    val_baseline_metrics = _regression_metrics(val_baseline_pred, target[val_mask])

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
        "target_name": TARGET_NAME,
        "coupling_dev_scale": float(args.coupling_dev_scale),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_mean": float(target.mean()),
        "max_dev_mean": float(max_dev.mean()),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": str(device),
        "selected_epoch": int(best_epoch),
        "selected_val_mae": float(best_val_mae),
        "selected_val_loss": float(best_val_loss),
        "val_mean_baseline_metrics": val_baseline_metrics,
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
            "target_name": TARGET_NAME,
            "coupling_dev_scale": float(args.coupling_dev_scale),
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
