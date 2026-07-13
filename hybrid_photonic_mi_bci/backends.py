"""Matrix-vector and matrix-operation backends.

The rest of the project should route algorithmic matrix products through this
module instead of using ``@`` or ``np.einsum`` directly. That makes the software
baseline and the future photonic hardware path share one contract.

The photonic core size is treated as a hardware tile, not as a system-level
algorithm limit. A ``2 x 8`` primitive can evaluate larger matrices by scanning
row and column blocks and accumulating partial sums.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import sosfiltfilt as _scipy_sosfiltfilt


FloatArray = NDArray[np.float64]


class MatrixOpsBackend(ABC):
    """Backend interface for algorithmic matrix products.

    Future photonic integration can implement this class and install it with
    :func:`set_matrix_ops_backend`. Higher-level helpers below keep operation
    names explicit so a hardware driver can route or profile different kernels.
    """

    @abstractmethod
    def matmul(self, left: ArrayLike, right: ArrayLike, *, name: str = "matmul") -> FloatArray:
        """Return ``left @ right``."""

    @abstractmethod
    def einsum(self, subscripts: str, *operands: ArrayLike, name: str = "einsum") -> FloatArray:
        """Return ``np.einsum(subscripts, *operands)``."""


class SignalOpsBackend(ABC):
    """Backend interface for forward linear EEG signal operations."""

    @abstractmethod
    def common_average_reference(
        self,
        samples: ArrayLike,
        *,
        channel_axis: int,
        name: str = "common_average_reference",
    ) -> FloatArray:
        """Return common-average referenced samples."""

    @abstractmethod
    def sosfiltfilt(
        self,
        sos: ArrayLike,
        samples: ArrayLike,
        *,
        axis: int,
        name: str = "sosfiltfilt",
    ) -> FloatArray:
        """Return zero-phase SOS filtered samples."""


class NumpyMatrixOpsBackend(MatrixOpsBackend):
    """CPU backend using NumPy for all matrix products."""

    def matmul(self, left: ArrayLike, right: ArrayLike, *, name: str = "matmul") -> FloatArray:
        del name
        return np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64)

    def einsum(self, subscripts: str, *operands: ArrayLike, name: str = "einsum") -> FloatArray:
        del name
        arrays = [np.asarray(operand, dtype=np.float64) for operand in operands]
        return np.einsum(subscripts, *arrays)


class SimulatedPhotonicMatrixOpsBackend(MatrixOpsBackend):
    """Software stand-in for the photonic MatrixOps backend.

    This class is the active handoff point for all algorithmic matrix products.
    It keeps numerical behavior deterministic by evaluating with NumPy today,
    while operation names and the backend boundary match the future photonic
    driver contract. The ``2 x 8`` tile shape is metadata for the hardware
    primitive; larger operations are still accepted as system-level matrices.
    """

    def __init__(self, tile_shape: tuple[int, int] = (2, 8), record_calls: bool = False):
        tile_rows, tile_cols = tile_shape
        if tile_rows <= 0 or tile_cols <= 0:
            raise ValueError("tile dimensions must be positive")
        self.tile_shape = (int(tile_rows), int(tile_cols))
        self.record_calls = bool(record_calls)
        self.calls: list[tuple[str, str]] = []

    def matmul(self, left: ArrayLike, right: ArrayLike, *, name: str = "matmul") -> FloatArray:
        if self.record_calls:
            self.calls.append(("matmul", name))
        return np.matmul(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))

    def einsum(self, subscripts: str, *operands: ArrayLike, name: str = "einsum") -> FloatArray:
        if self.record_calls:
            self.calls.append(("einsum", name))
        arrays = [np.asarray(operand, dtype=np.float64) for operand in operands]
        return np.einsum(subscripts, *arrays)


class ScipySignalOpsBackend(SignalOpsBackend):
    """CPU signal backend using NumPy/SciPy."""

    def common_average_reference(
        self,
        samples: ArrayLike,
        *,
        channel_axis: int,
        name: str = "common_average_reference",
    ) -> FloatArray:
        del name
        x = np.asarray(samples, dtype=np.float64)
        return x - x.mean(axis=channel_axis, keepdims=True)

    def sosfiltfilt(
        self,
        sos: ArrayLike,
        samples: ArrayLike,
        *,
        axis: int,
        name: str = "sosfiltfilt",
    ) -> FloatArray:
        del name
        return _scipy_sosfiltfilt(
            np.asarray(sos, dtype=np.float64),
            np.asarray(samples, dtype=np.float64),
            axis=axis,
        )


class SimulatedPhotonicSignalOpsBackend(SignalOpsBackend):
    """Software stand-in for photonic forward signal-processing operations."""

    def __init__(self, record_calls: bool = False):
        self.record_calls = bool(record_calls)
        self.calls: list[tuple[str, str]] = []

    def common_average_reference(
        self,
        samples: ArrayLike,
        *,
        channel_axis: int,
        name: str = "common_average_reference",
    ) -> FloatArray:
        if self.record_calls:
            self.calls.append(("common_average_reference", name))
        x = np.asarray(samples, dtype=np.float64)
        return x - x.mean(axis=channel_axis, keepdims=True)

    def sosfiltfilt(
        self,
        sos: ArrayLike,
        samples: ArrayLike,
        *,
        axis: int,
        name: str = "sosfiltfilt",
    ) -> FloatArray:
        if self.record_calls:
            self.calls.append(("sosfiltfilt", name))
        return _scipy_sosfiltfilt(
            np.asarray(sos, dtype=np.float64),
            np.asarray(samples, dtype=np.float64),
            axis=axis,
        )


class PhotonicSignalOpsBackendStub(SignalOpsBackend):
    """Placeholder for a future photonic signal-processing backend."""

    def common_average_reference(
        self,
        samples: ArrayLike,
        *,
        channel_axis: int,
        name: str = "common_average_reference",
    ) -> FloatArray:
        del samples, channel_axis
        raise NotImplementedError(f"Photonic signal op {name!r} is not connected yet")

    def sosfiltfilt(
        self,
        sos: ArrayLike,
        samples: ArrayLike,
        *,
        axis: int,
        name: str = "sosfiltfilt",
    ) -> FloatArray:
        del sos, samples, axis
        raise NotImplementedError(f"Photonic signal op {name!r} is not connected yet")


class PhotonicMatrixOpsBackendStub(MatrixOpsBackend):
    """Placeholder for a future photonic matrix-operation backend."""

    def matmul(self, left: ArrayLike, right: ArrayLike, *, name: str = "matmul") -> FloatArray:
        del left, right
        raise NotImplementedError(f"Photonic matrix op {name!r} is not connected yet")

    def einsum(self, subscripts: str, *operands: ArrayLike, name: str = "einsum") -> FloatArray:
        del subscripts, operands
        raise NotImplementedError(f"Photonic einsum op {name!r} is not connected yet")


_MATRIX_OPS_BACKEND: MatrixOpsBackend = SimulatedPhotonicMatrixOpsBackend()
_SIGNAL_OPS_BACKEND: SignalOpsBackend = SimulatedPhotonicSignalOpsBackend()


def set_matrix_ops_backend(backend: MatrixOpsBackend) -> None:
    """Install the process-wide matrix-operation backend."""

    global _MATRIX_OPS_BACKEND
    _MATRIX_OPS_BACKEND = backend


def set_signal_ops_backend(backend: SignalOpsBackend) -> None:
    """Install the process-wide forward signal-operation backend."""

    global _SIGNAL_OPS_BACKEND
    _SIGNAL_OPS_BACKEND = backend


def get_matrix_ops_backend() -> MatrixOpsBackend:
    """Return the currently installed matrix-operation backend."""

    return _MATRIX_OPS_BACKEND


def get_signal_ops_backend() -> SignalOpsBackend:
    """Return the currently installed signal-operation backend."""

    return _SIGNAL_OPS_BACKEND


@contextmanager
def use_matrix_ops_backend(backend: MatrixOpsBackend) -> Iterator[None]:
    """Temporarily install a matrix-operation backend."""

    previous = get_matrix_ops_backend()
    set_matrix_ops_backend(backend)
    try:
        yield
    finally:
        set_matrix_ops_backend(previous)


@contextmanager
def use_signal_ops_backend(backend: SignalOpsBackend) -> Iterator[None]:
    """Temporarily install a signal-operation backend."""

    previous = get_signal_ops_backend()
    set_signal_ops_backend(backend)
    try:
        yield
    finally:
        set_signal_ops_backend(previous)


def matrix_multiply(left: ArrayLike, right: ArrayLike, *, name: str = "matmul") -> FloatArray:
    """Unified ``left @ right`` entry point for algorithm code."""

    return get_matrix_ops_backend().matmul(left, right, name=name)


def matrix_einsum(
    subscripts: str,
    *operands: ArrayLike,
    name: str = "einsum",
) -> FloatArray:
    """Unified ``np.einsum`` entry point for algorithm code."""

    return get_matrix_ops_backend().einsum(subscripts, *operands, name=name)


def common_average_reference(
    samples: ArrayLike,
    *,
    channel_axis: int,
    name: str = "common_average_reference",
) -> FloatArray:
    """Unified CAR entry point for forward signal-processing code."""

    return get_signal_ops_backend().common_average_reference(
        samples,
        channel_axis=channel_axis,
        name=name,
    )


def signal_sosfiltfilt(
    sos: ArrayLike,
    samples: ArrayLike,
    *,
    axis: int,
    name: str = "sosfiltfilt",
) -> FloatArray:
    """Unified zero-phase SOS filtering entry point."""

    return get_signal_ops_backend().sosfiltfilt(sos, samples, axis=axis, name=name)


def affine_transform(
    features: ArrayLike,
    weights: ArrayLike,
    bias: ArrayLike,
    *,
    name: str = "affine_transform",
) -> FloatArray:
    """Return ``features @ weights.T + bias`` as one augmented matmul.

    Bias is represented by appending a constant-one input channel, matching the
    usual photonic MVM implementation pattern for affine heads.
    """

    x = np.asarray(features, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    was_vector = x.ndim == 1
    if was_vector:
        x_2d = x[None, :]
    elif x.ndim == 2:
        x_2d = x
    else:
        raise ValueError(f"features must have shape (N, D) or (D,), got {x.shape}")
    if w.ndim != 2:
        raise ValueError(f"weights must have shape (M, D), got {w.shape}")
    if b.shape != (w.shape[0],):
        raise ValueError(f"bias must have shape ({w.shape[0]},), got {b.shape}")
    if x_2d.shape[1] != w.shape[1]:
        raise ValueError(
            "feature dimension must match weights input dimension, "
            f"got {x_2d.shape[1]} and {w.shape[1]}"
        )
    ones = np.ones((x_2d.shape[0], 1), dtype=np.float64)
    augmented_features = np.concatenate([x_2d, ones], axis=1)
    augmented_weights = np.concatenate([w, b[:, None]], axis=1)
    out = matrix_multiply(augmented_features, augmented_weights.T, name=name)
    return out[0] if was_vector else out


def featurewise_affine(
    features: ArrayLike,
    scale: ArrayLike,
    bias: ArrayLike,
    *,
    name: str = "featurewise_affine",
) -> FloatArray:
    """Apply a per-feature affine map through the matrix backend.

    ``scale`` and ``bias`` are converted to a diagonal augmented matrix, so
    standardization can be handed to the same photonic MVM contract.
    """

    x = np.asarray(features, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"scale must have shape (D,), got {s.shape}")
    if b.shape != s.shape:
        raise ValueError(f"bias must have shape {s.shape}, got {b.shape}")
    if x.shape[-1] != s.shape[0]:
        raise ValueError(
            "last feature dimension must match scale length, "
            f"got {x.shape[-1]} and {s.shape[0]}"
        )
    flat = x.reshape(-1, s.shape[0])
    weights = np.diag(s)
    transformed = affine_transform(flat, weights, b, name=name)
    return transformed.reshape(x.shape)


def linear_scores(
    features: ArrayLike,
    weights: ArrayLike,
    bias: ArrayLike,
    *,
    name: str = "linear_scores",
) -> FloatArray:
    """Return ``features @ weights.T + bias`` through one augmented matmul."""

    return affine_transform(features, weights, bias, name=name)


def covariance_gram(centered: ArrayLike, *, name: str = "covariance_gram") -> FloatArray:
    """Return ``centered @ centered.T`` through the matrix backend."""

    x = np.asarray(centered, dtype=np.float64)
    return matrix_multiply(x, x.T, name=name)


def csp_spatial_project(
    filters: ArrayLike,
    band_trials: ArrayLike,
    *,
    name: str = "csp_spatial_project",
) -> FloatArray:
    """Return FBCSP spatial projection ``filters @ trials``.

    Shapes are ``filters=(F, C)`` and ``band_trials=(N, C, T)``, with output
    ``(N, F, T)``.
    """

    return matrix_einsum("fc,nct->nft", filters, band_trials, name=name)


def candidate_probability_fusion(
    retrieval_weights: ArrayLike,
    candidate_probabilities: ArrayLike,
    *,
    name: str = "candidate_probability_fusion",
) -> FloatArray:
    """Fuse candidate probabilities with retrieval weights."""

    return matrix_einsum("k,nkc->nc", retrieval_weights, candidate_probabilities, name=name)


def pairwise_squared_distances(
    vectors: ArrayLike,
    prototypes: ArrayLike,
    *,
    name: str = "pairwise_squared_distances",
) -> FloatArray:
    """Return squared distances while routing the cross term through backend.

    ``prototypes`` may be shared across vectors with shape ``(C, D)`` or
    candidate-specific with shape ``(N, C, D)``. The squared-distance expansion
    exposes the dot-product core to the matrix backend:
    ``||x - p||^2 = ||x||^2 + ||p||^2 - 2 x p.T``.
    """

    x = np.asarray(vectors, dtype=np.float64)
    p = np.asarray(prototypes, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"vectors must have shape (N, D), got {x.shape}")
    if p.ndim == 2:
        if p.shape[1] != x.shape[1]:
            raise ValueError(
                "prototype dimension must match vector dimension, "
                f"got {p.shape[1]} and {x.shape[1]}"
            )
        cross = matrix_multiply(x, p.T, name=f"{name}_shared_cross")
        squared = (
            np.sum(x * x, axis=1, keepdims=True)
            + np.sum(p * p, axis=1, keepdims=True).T
            - 2.0 * cross
        )
    elif p.ndim == 3:
        if p.shape[0] != x.shape[0]:
            raise ValueError(
                "candidate-specific prototypes must match vector count, "
                f"got {p.shape[0]} and {x.shape[0]}"
            )
        if p.shape[2] != x.shape[1]:
            raise ValueError(
                "prototype dimension must match vector dimension, "
                f"got {p.shape[2]} and {x.shape[1]}"
            )
        cross = matrix_einsum("nd,ncd->nc", x, p, name=f"{name}_candidate_cross")
        squared = (
            np.sum(x * x, axis=1, keepdims=True)
            + np.sum(p * p, axis=2)
            - 2.0 * cross
        )
    else:
        raise ValueError(
            "prototypes must have shape (C, D) or (N, C, D), "
            f"got {p.shape}"
        )
    return np.maximum(squared, 0.0)


def prototype_distances(
    vectors: ArrayLike,
    prototypes: ArrayLike,
    *,
    name: str = "prototype_distances",
) -> FloatArray:
    """Return Euclidean prototype distances via backend-routed cross terms."""

    return np.sqrt(pairwise_squared_distances(vectors, prototypes, name=name))


def batched_matrix_vector(
    weights: ArrayLike,
    features: ArrayLike,
    *,
    name: str = "batched_matrix_vector",
) -> FloatArray:
    """Return a bank of matrix-vector products ``weights @ features``."""

    return matrix_multiply(weights, features, name=name)


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
        return batched_matrix_vector(weights_arr, features_arr, name="candidate_bank_scan")

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
                        matrix_multiply(
                            weights_arr[
                                candidate_index,
                                row_start:row_stop,
                                col_start:col_stop,
                            ],
                            features_arr[col_start:col_stop],
                            name="tiled_candidate_scan_tile",
                        )
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
