"""Feature preprocessing utilities for 8-D MI-BCI vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass
class Standardizer:
    """Mean/std normalization with a scikit-learn-like API."""

    mean_: FloatArray | None = None
    scale_: FloatArray | None = None
    eps: float = 1e-8

    def fit(self, features: ArrayLike) -> "Standardizer":
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 8:
            raise ValueError(f"features must have shape (n_samples, 8), got {x.shape}")
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0) + self.eps
        return self

    def transform(self, features: ArrayLike) -> FloatArray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fitted before transform")
        x = np.asarray(features, dtype=np.float64)
        if x.shape[-1] != 8:
            raise ValueError(f"last feature dimension must be 8, got {x.shape}")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, features: ArrayLike) -> FloatArray:
        return self.fit(features).transform(features)
