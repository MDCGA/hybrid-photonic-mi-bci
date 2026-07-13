"""Small neural networks for FBCSP feature embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import nn
from scipy.special import erf

from .backends import featurewise_affine
from .backends import linear_scores as backend_linear_scores


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
    x = np.asarray(features, dtype=np.float64)
    layer_norm = model.encoder[0]
    linear_1 = model.encoder[1]
    linear_2 = model.encoder[4]
    classifier = model.classifier
    hidden = _layer_norm_numpy(
        x,
        weight=layer_norm.weight.detach().cpu().numpy(),
        bias=layer_norm.bias.detach().cpu().numpy(),
        eps=float(layer_norm.eps),
    )
    hidden = backend_linear_scores(
        hidden,
        linear_1.weight.detach().cpu().numpy(),
        linear_1.bias.detach().cpu().numpy(),
        name="small_mlp_encoder_linear_1",
    )
    hidden = _gelu_numpy(hidden)
    embeddings = backend_linear_scores(
        hidden,
        linear_2.weight.detach().cpu().numpy(),
        linear_2.bias.detach().cpu().numpy(),
        name="small_mlp_encoder_linear_2",
    )
    embeddings = _gelu_numpy(embeddings)
    logits = backend_linear_scores(
        embeddings,
        classifier.weight.detach().cpu().numpy(),
        classifier.bias.detach().cpu().numpy(),
        name="small_mlp_classifier_linear",
    )
    return logits.astype(np.float64), embeddings.astype(np.float64)


def _layer_norm_numpy(
    features: np.ndarray,
    *,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float,
) -> FloatArray:
    x = np.asarray(features, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(variance + eps)
    return featurewise_affine(
        normalized,
        scale=weight.astype(np.float64),
        bias=bias.astype(np.float64),
        name="small_mlp_layernorm_affine",
    )


def _gelu_numpy(values: np.ndarray) -> FloatArray:
    x = np.asarray(values, dtype=np.float64)
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
