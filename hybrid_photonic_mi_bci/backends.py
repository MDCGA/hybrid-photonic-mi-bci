"""Matrix-vector multiplication backends.

The rest of the project should call :class:`MVMBackend` instead of using
``weights @ x`` directly. That makes the software baseline and the future
photonic hardware path share one contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class MVMBackend(ABC):
    """Interface for scanning candidate ``2 x 8`` projection matrices."""

    @abstractmethod
    def scan(self, weights: ArrayLike, features: ArrayLike) -> FloatArray:
        """Return all candidate projections.

        Parameters
        ----------
        weights:
            Candidate matrices with shape ``(n_candidates, 2, 8)``.
        features:
            One 8-D feature vector, shape ``(8,)``.

        Returns
        -------
        np.ndarray
            Projected points with shape ``(n_candidates, 2)``.
        """


class NumpyMVMBackend(MVMBackend):
    """Reference backend using NumPy batched matrix multiplication."""

    def scan(self, weights: ArrayLike, features: ArrayLike) -> FloatArray:
        weights_arr = np.asarray(weights, dtype=np.float64)
        features_arr = np.asarray(features, dtype=np.float64)
        self._validate(weights_arr, features_arr)
        return weights_arr @ features_arr

    @staticmethod
    def _validate(weights: FloatArray, features: FloatArray) -> None:
        if weights.ndim != 3:
            raise ValueError(f"weights must have shape (N, 2, 8), got {weights.shape}")
        if weights.shape[1:] != (2, 8):
            raise ValueError(f"weights must have shape (N, 2, 8), got {weights.shape}")
        if features.shape != (8,):
            raise ValueError(f"features must have shape (8,), got {features.shape}")


class PhotonicMVMBackendStub(MVMBackend):
    """Placeholder for the future photonic chip integration.

    Keep this class intentionally small: the hardware implementation should
    preserve ``scan(weights, features) -> (N, 2)`` and can add calibration,
    quantization, transport, and readout details behind this boundary.
    """

    def scan(self, weights: ArrayLike, features: ArrayLike) -> FloatArray:
        raise NotImplementedError(
            "Photonic backend is not connected yet. Use NumpyMVMBackend for "
            "software simulation, or implement the hardware transport here."
        )
