"""Forward-only photonic candidate-scan runtime.

This module intentionally contains no evaluation metrics, plotting, dataset
loading, or training code. It is a thin deployment-facing wrapper around the
project's experience library and quantized tiled photonic scan backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hybrid_photonic_mi_bci.backends import (
    PhotonicQuantizationConfig,
    QuantizedPhotonicMatrixOpsBackend,
    TiledMVMBackend,
)
from hybrid_photonic_mi_bci.experience import (
    ExperienceEntry,
    retrieve_top_k,
    scan_experience_heads,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RuntimeDecision:
    predicted_index: int | None
    label: str
    confidence: float
    margin: float
    rejected: bool


@dataclass(frozen=True)
class RuntimeOutput:
    probabilities: FloatArray
    fused_scores: FloatArray
    decisions: tuple[RuntimeDecision, ...]
    tile_count_per_window: int


class PurePhotonicScanRuntime:
    """Calibrate experience retrieval and run 4-bit photonic candidate scans."""

    def __init__(
        self,
        entries: tuple[ExperienceEntry, ...],
        class_names: tuple[str, ...],
        *,
        top_k: int = 8,
        tile_shape: tuple[int, int] = (2, 8),
        reject_threshold: float = 0.45,
        margin_threshold: float = 0.0,
        quantization: PhotonicQuantizationConfig | None = None,
    ) -> None:
        if not entries:
            raise ValueError("experience entries must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.entries = tuple(entries)
        self.class_names = tuple(class_names)
        self.top_k = int(top_k)
        self.reject_threshold = float(reject_threshold)
        self.margin_threshold = float(margin_threshold)
        matrix_backend = QuantizedPhotonicMatrixOpsBackend(
            config=quantization or PhotonicQuantizationConfig.for_bit_width(4),
        )
        self.scan_backend = TiledMVMBackend(
            tile_shape=tile_shape,
            matrix_backend=matrix_backend,
        )
        self.selected_entries: tuple[ExperienceEntry, ...] | None = None
        self.retrieval_weights: FloatArray | None = None
        self.retrieval_distances: FloatArray | None = None

    def calibrate(self, calibration_embeddings: ArrayLike) -> None:
        """Select and weight top-K experience entries from calibration windows."""

        selected, weights, distances = retrieve_top_k(
            self.entries,
            calibration_embeddings,
            k=self.top_k,
        )
        self.selected_entries = selected
        self.retrieval_weights = weights
        self.retrieval_distances = distances

    def predict(self, embeddings: ArrayLike) -> RuntimeOutput:
        """Run forward-only candidate scan and return command decisions."""

        if self.selected_entries is None or self.retrieval_weights is None:
            raise RuntimeError("calibrate must be called before predict")
        scan = scan_experience_heads(
            self.selected_entries,
            self.retrieval_weights,
            embeddings,
            backend=self.scan_backend,
        )
        probabilities = _softmax(scan.fused_scores)
        decisions = tuple(
            self._decision_from_probability(probability)
            for probability in probabilities
        )
        return RuntimeOutput(
            probabilities=probabilities,
            fused_scores=scan.fused_scores,
            decisions=decisions,
            tile_count_per_window=scan.tile_count_per_window,
        )

    def _decision_from_probability(self, probability: FloatArray) -> RuntimeDecision:
        order = np.argsort(probability)[::-1]
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else best
        confidence = float(probability[best])
        margin = float(probability[best] - probability[second])
        rejected = (
            confidence < self.reject_threshold
            or margin < self.margin_threshold
        )
        predicted_index = None if rejected else best
        label = "reject" if rejected else self.class_names[best]
        return RuntimeDecision(
            predicted_index=predicted_index,
            label=label,
            confidence=confidence,
            margin=margin,
            rejected=rejected,
        )


def _softmax(scores: ArrayLike) -> FloatArray:
    x = np.asarray(scores, dtype=np.float64)
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
