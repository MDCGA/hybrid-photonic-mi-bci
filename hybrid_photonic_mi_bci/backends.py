"""Matrix-vector multiplication backends.

The rest of the project should call :class:`MVMBackend` instead of using
``weights @ x`` directly. That makes the software baseline and the future
photonic hardware path share one contract.

The photonic core size is treated as a hardware tile, not as a system-level
algorithm limit. A ``2 x 8`` primitive can evaluate larger matrices by scanning
row and column blocks and accumulating partial sums.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class MVMBackend(ABC):
    """Interface for scanning a bank of candidate projection matrices."""

    @abstractmethod
    def scan(self, weights: ArrayLike, features: ArrayLike) -> FloatArray:
        """Return all candidate projections.

        Parameters
        ----------
        weights:
            Candidate matrices with shape ``(n_candidates, out_dim, in_dim)``.
        features:
            One feature vector with shape ``(in_dim,)``.

        Returns
        -------
        np.ndarray
            Projected points with shape ``(n_candidates, out_dim)``.
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
            raise ValueError(f"weights must have shape (N, M, D), got {weights.shape}")
        if features.ndim != 1:
            raise ValueError(f"features must have shape (D,), got {features.shape}")
        if weights.shape[2] != features.shape[0]:
            raise ValueError(
                "weights input dimension must match features length, "
                f"got weights {weights.shape} and features {features.shape}"
            )


class TiledMVMBackend(MVMBackend):
    """Software model of a tiled photonic MVM primitive.

    The default tile shape is ``2 x 8`` because that is the current photonic
    compute unit. Larger matrices are evaluated by scanning output-row blocks
    and input-column blocks, then accumulating partial dot products digitally.
    """

    def __init__(self, tile_shape: tuple[int, int] = (2, 8)):
        tile_rows, tile_cols = tile_shape
        if tile_rows <= 0 or tile_cols <= 0:
            raise ValueError("tile dimensions must be positive")
        self.tile_shape = (int(tile_rows), int(tile_cols))
        self.last_tile_count = 0

    def scan(self, weights: ArrayLike, features: ArrayLike) -> FloatArray:
        weights_arr = np.asarray(weights, dtype=np.float64)
        features_arr = np.asarray(features, dtype=np.float64)
        NumpyMVMBackend._validate(weights_arr, features_arr)

        n_candidates, out_dim, in_dim = weights_arr.shape
        tile_rows, tile_cols = self.tile_shape
        output = np.zeros((n_candidates, out_dim), dtype=np.float64)
        tile_count = 0
        for candidate_index in range(n_candidates):
            for row_start in range(0, out_dim, tile_rows):
                row_stop = min(row_start + tile_rows, out_dim)
                for col_start in range(0, in_dim, tile_cols):
                    col_stop = min(col_start + tile_cols, in_dim)
                    output[candidate_index, row_start:row_stop] += (
                        weights_arr[
                            candidate_index,
                            row_start:row_stop,
                            col_start:col_stop,
                        ]
                        @ features_arr[col_start:col_stop]
                    )
                    tile_count += 1
        self.last_tile_count = tile_count
        return output

    def count_tiles(self, weights: ArrayLike) -> int:
        """Return how many hardware-tile evaluations one scan would require."""

        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.ndim != 3:
            raise ValueError(f"weights must have shape (N, M, D), got {weights_arr.shape}")
        n_candidates, out_dim, in_dim = weights_arr.shape
        tile_rows, tile_cols = self.tile_shape
        row_tiles = int(np.ceil(out_dim / tile_rows))
        col_tiles = int(np.ceil(in_dim / tile_cols))
        return n_candidates * row_tiles * col_tiles


class PhotonicMVMBackendStub(MVMBackend):
    """Placeholder for the future photonic chip integration.

    Keep this class intentionally small: the hardware implementation should
    preserve ``scan(weights, features) -> (N, M)`` and can add calibration,
    quantization, tiling, transport, and readout details behind this boundary.
    """

    def scan(self, weights: ArrayLike, features: ArrayLike) -> FloatArray:
        raise NotImplementedError(
            "Photonic backend is not connected yet. Use NumpyMVMBackend for "
            "software simulation, or implement the hardware transport here."
        )
