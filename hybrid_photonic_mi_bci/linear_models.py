"""Small linear models used by FBCSP reference and candidate heads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backends import linear_scores as backend_linear_scores
from .backends import featurewise_affine
from .backends import matrix_multiply
from .evaluation import softmax


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class LinearHead:
    """A linear score head ``scores = X @ weights.T + bias``."""

    weights: FloatArray
    bias: FloatArray
    class_names: tuple[str, ...]

    def scores(self, features: ArrayLike) -> FloatArray:
        return backend_linear_scores(
            features,
            self.weights,
            self.bias,
            name="linear_head_scores",
        )

    def probabilities(self, features: ArrayLike) -> FloatArray:
        return softmax(self.scores(features))


class FeatureStandardizer:
    """Standardize features using training-set statistics."""

    def __init__(self) -> None:
        self.mean_: FloatArray | None = None
        self.scale_: FloatArray | None = None

    def fit(self, features: ArrayLike) -> "FeatureStandardizer":
        x = np.asarray(features, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        self.scale_ = np.where(scale < 1e-8, 1.0, scale)
        return self

    def transform(self, features: ArrayLike) -> FloatArray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureStandardizer must be fitted before transform")
        return featurewise_affine(
            np.asarray(features, dtype=np.float64),
            scale=1.0 / self.scale_,
            bias=-self.mean_ / self.scale_,
            name="feature_standardizer_affine",
        )

    def fit_transform(self, features: ArrayLike) -> FloatArray:
        return self.fit(features).transform(features)


class ShrinkageLDA:
    """Multiclass LDA with a simple diagonal shrinkage covariance estimator."""

    def __init__(self, shrinkage: float = 0.15):
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        self.shrinkage = float(shrinkage)
        self.head_: LinearHead | None = None
        self.covariance_: FloatArray | None = None

    def fit(
        self,
        features: ArrayLike,
        labels: ArrayLike,
        class_names: tuple[str, ...],
    ) -> "ShrinkageLDA":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        if x.ndim != 2:
            raise ValueError(f"features must have shape (N, D), got {x.shape}")
        if y.shape != (x.shape[0],):
            raise ValueError(f"labels must have shape ({x.shape[0]},), got {y.shape}")

        n_classes = len(class_names)
        means = np.zeros((n_classes, x.shape[1]), dtype=np.float64)
        priors = np.zeros(n_classes, dtype=np.float64)
        pooled = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
        dof = 0
        for class_index in range(n_classes):
            class_x = x[y == class_index]
            if len(class_x) == 0:
                raise ValueError(f"class {class_names[class_index]!r} has no samples")
            means[class_index] = class_x.mean(axis=0)
            priors[class_index] = len(class_x) / len(x)
            centered = class_x - means[class_index]
            pooled += matrix_multiply(
                centered.T,
                centered,
                name="lda_pooled_covariance",
            )
            dof += max(0, len(class_x) - 1)
        pooled /= max(1, dof)
        diagonal = np.diag(np.diag(pooled))
        covariance = (1.0 - self.shrinkage) * pooled + self.shrinkage * diagonal
        covariance += np.eye(covariance.shape[0]) * 1e-6
        inv_cov = np.linalg.pinv(covariance)
        weights = matrix_multiply(means, inv_cov, name="lda_weight_projection")
        bias_projection = matrix_multiply(means, inv_cov, name="lda_bias_projection")
        bias = -0.5 * np.sum(bias_projection * means, axis=1) + np.log(priors + 1e-12)
        self.head_ = LinearHead(weights=weights, bias=bias, class_names=class_names)
        self.covariance_ = covariance
        return self

    @property
    def head(self) -> LinearHead:
        if self.head_ is None:
            raise RuntimeError("ShrinkageLDA must be fitted before use")
        return self.head_

    def scores(self, features: ArrayLike) -> FloatArray:
        return self.head.scores(features)

    def predict(self, features: ArrayLike) -> IntArray:
        return np.argmax(self.scores(features), axis=1).astype(int)


def fisher_scores(features: ArrayLike, labels: ArrayLike, n_classes: int) -> FloatArray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    global_mean = x.mean(axis=0)
    between = np.zeros(x.shape[1], dtype=np.float64)
    within = np.zeros(x.shape[1], dtype=np.float64)
    for class_index in range(n_classes):
        class_x = x[y == class_index]
        if len(class_x) == 0:
            continue
        class_mean = class_x.mean(axis=0)
        between += len(class_x) * (class_mean - global_mean) ** 2
        within += ((class_x - class_mean) ** 2).sum(axis=0)
    return between / (within + 1e-12)


def select_fisher_features(
    train_features: ArrayLike,
    train_labels: ArrayLike,
    n_classes: int,
    n_features: int,
) -> NDArray[np.int_]:
    scores = fisher_scores(train_features, train_labels, n_classes=n_classes)
    count = min(int(n_features), scores.shape[0])
    return np.argsort(scores)[-count:][::-1].astype(int)
