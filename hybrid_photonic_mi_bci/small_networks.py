"""Small neural networks for FBCSP feature embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import nn


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SmallMLPConfig:
    hidden_dim: int = 64
    embedding_dim: int = 32
    dropout: float = 0.20
    lr: float = 1e-3
    weight_decay: float = 1e-3
    epochs: int = 120
    batch_size: int = 64
    seed: int = 13


class FBCSPSmallMLP(nn.Module):
    """Compact MLP mapping FBCSP vectors to embeddings and class logits."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(features)
        logits = self.classifier(embedding)
        return logits, embedding


@dataclass
class SmallMLPTrainingResult:
    model: FBCSPSmallMLP
    history: dict[str, list[float]]
    train_scores: FloatArray
    replay_scores: FloatArray
    train_embeddings: FloatArray
    replay_embeddings: FloatArray


def train_small_mlp(
    train_features: ArrayLike,
    train_labels: ArrayLike,
    replay_features: ArrayLike,
    n_classes: int,
    config: SmallMLPConfig | None = None,
) -> SmallMLPTrainingResult:
    cfg = config or SmallMLPConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    x_train = np.asarray(train_features, dtype=np.float32)
    y_train = np.asarray(train_labels, dtype=np.int64)
    x_replay = np.asarray(replay_features, dtype=np.float32)
    model = FBCSPSmallMLP(
        input_dim=x_train.shape[1],
        n_classes=n_classes,
        hidden_dim=cfg.hidden_dim,
        embedding_dim=cfg.embedding_dim,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history = {"loss": [], "accuracy": []}
    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train)

    model.train()
    for _epoch in range(cfg.epochs):
        order = rng.permutation(len(x_train))
        epoch_losses = []
        correct = 0
        seen = 0
        for start in range(0, len(order), cfg.batch_size):
            batch_idx = order[start : start + cfg.batch_size]
            xb = x_tensor[batch_idx]
            yb = y_tensor[batch_idx]
            optimizer.zero_grad()
            logits, _embedding = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            seen += len(batch_idx)
        history["loss"].append(float(np.mean(epoch_losses)))
        history["accuracy"].append(float(correct / max(1, seen)))

    train_scores, train_embeddings = _forward_numpy(model, x_train)
    replay_scores, replay_embeddings = _forward_numpy(model, x_replay)
    return SmallMLPTrainingResult(
        model=model,
        history=history,
        train_scores=train_scores,
        replay_scores=replay_scores,
        train_embeddings=train_embeddings,
        replay_embeddings=replay_embeddings,
    )


def _forward_numpy(model: FBCSPSmallMLP, features: np.ndarray) -> tuple[FloatArray, FloatArray]:
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(torch.from_numpy(features.astype(np.float32)))
    return (
        logits.detach().cpu().numpy().astype(np.float64),
        embeddings.detach().cpu().numpy().astype(np.float64),
    )
