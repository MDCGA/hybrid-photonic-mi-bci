"""Digital decision head after candidate MVM scanning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DecisionConfig:
    """Decision and reject settings."""

    class_names: tuple[str, ...] = ("left_hand", "right_hand", "feet")
    temperature: float = 1.0
    reject_threshold: float = 0.45
    margin_threshold: float = 0.05


@dataclass(frozen=True)
class CandidateDecision:
    """Decision details for one candidate projection."""

    candidate_index: int
    probabilities: FloatArray
    predicted_index: int
    confidence: float
    margin: float
    rejected: bool


class PrototypeDecisionHead:
    """Classify projection points by distance to class prototypes."""

    def __init__(self, prototypes: ArrayLike, config: DecisionConfig | None = None):
        self.config = config or DecisionConfig()
        self.prototypes = np.asarray(prototypes, dtype=np.float64)
        class_count = len(self.config.class_names)
        if self.prototypes.ndim == 2:
            if self.prototypes.shape[0] != class_count:
                raise ValueError(
                    "shared prototypes must have shape "
                    f"({class_count}, projection_dim), got {self.prototypes.shape}"
                )
        elif self.prototypes.ndim == 3:
            if self.prototypes.shape[1] != class_count:
                raise ValueError(
                    "candidate-specific prototypes must have shape "
                    f"(N, {class_count}, projection_dim), got {self.prototypes.shape}"
                )
        else:
            raise ValueError(
                "prototypes must have shape "
                f"({class_count}, projection_dim) or "
                f"(N, {class_count}, projection_dim), got {self.prototypes.shape}"
            )
        if self.config.temperature <= 0:
            raise ValueError("temperature must be positive")

    def decide_all(self, projections: ArrayLike) -> list[CandidateDecision]:
        z = np.asarray(projections, dtype=np.float64)
        if z.ndim != 2:
            raise ValueError(f"projections must have shape (N, projection_dim), got {z.shape}")
        if self.prototypes.ndim == 2:
            if self.prototypes.shape[1] != z.shape[1]:
                raise ValueError(
                    "prototype dimension must match projection dimension, "
                    f"got {self.prototypes.shape[1]} and {z.shape[1]}"
                )
            distances = np.linalg.norm(z[:, None, :] - self.prototypes[None, :, :], axis=2)
        else:
            if self.prototypes.shape[0] != z.shape[0]:
                raise ValueError(
                    "candidate-specific prototypes must match projection count, "
                    f"got {self.prototypes.shape[0]} and {z.shape[0]}"
                )
            if self.prototypes.shape[2] != z.shape[1]:
                raise ValueError(
                    "prototype dimension must match projection dimension, "
                    f"got {self.prototypes.shape[2]} and {z.shape[1]}"
                )
            distances = np.linalg.norm(z[:, None, :] - self.prototypes, axis=2)
        logits = -distances / self.config.temperature
        probabilities = _softmax(logits)
        decisions: list[CandidateDecision] = []
        for i, probs in enumerate(probabilities):
            order = np.argsort(probs)[::-1]
            predicted_index = int(order[0])
            confidence = float(probs[predicted_index])
            margin = float(probs[order[0]] - probs[order[1]])
            rejected = (
                confidence < self.config.reject_threshold
                or margin < self.config.margin_threshold
            )
            decisions.append(
                CandidateDecision(
                    candidate_index=i,
                    probabilities=probs,
                    predicted_index=predicted_index,
                    confidence=confidence,
                    margin=margin,
                    rejected=rejected,
                )
            )
        return decisions


def _softmax(logits: FloatArray) -> FloatArray:
    centered = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)
