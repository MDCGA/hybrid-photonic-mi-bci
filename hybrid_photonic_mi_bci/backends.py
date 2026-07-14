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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PhotonicQuantizationConfig:
    """Quantization limits for the photonic integer MVM path."""

    bit_width: int = 4
    qinmin: int = 0
    qinmax: int = 15
    qwtmin: int = -8
    qwtmax: int = 7
    eps: float = 1e-8

    @classmethod
    def for_bit_width(cls, bit_width: int) -> "PhotonicQuantizationConfig":
        if bit_width == 4:
            return cls(bit_width=4, qinmin=0, qinmax=15, qwtmin=-8, qwtmax=7)
        if bit_width == 8:
            return cls(bit_width=8, qinmin=0, qinmax=255, qwtmin=-128, qwtmax=127)
        raise ValueError("photonic quantization supports 4-bit or 8-bit inputs")

    @property
    def input_type(self) -> str:
        return f"uint{self.bit_width}"


class QuantizedPhotonicMatrixOpsBackend(MatrixOpsBackend):
    """4/8-bit software bridge to the photonic integer matmul contract.

    The quantization follows the LT-Simulator ``custom_matmul.py`` pattern:
    activations use affine unsigned quantization, weights use symmetric signed
    quantization, integer matmul runs through the optional Gazelle simulator
    when available, then the zero-point offset is removed before dequantizing.
    """

    def __init__(
        self,
        config: PhotonicQuantizationConfig | None = None,
        *,
        record_calls: bool = False,
        use_gazelle_model: bool = True,
        announce: bool = True,
    ) -> None:
        self.config = config or PhotonicQuantizationConfig.for_bit_width(4)
        self.record_calls = bool(record_calls)
        self.calls: list[tuple[str, str]] = []
        self._gazelle_model = _load_gazelle_model() if use_gazelle_model else None
        self.execution_backend = (
            "gazelle_simulator"
            if self._gazelle_model is not None
            else "numpy_integer_matmul"
        )
        if announce:
            print(
                "[photonic-matmul] "
                f"backend={self.execution_backend}, "
                f"bit_width={self.config.bit_width}, "
                f"qin=[{self.config.qinmin},{self.config.qinmax}], "
                f"qwt=[{self.config.qwtmin},{self.config.qwtmax}]"
            )

    def matmul(self, left: ArrayLike, right: ArrayLike, *, name: str = "matmul") -> FloatArray:
        if self.record_calls:
            self.calls.append(("matmul", name))
        return _quantized_photonic_matmul(
            left,
            right,
            config=self.config,
            gazelle_model=self._gazelle_model,
        )

    def einsum(self, subscripts: str, *operands: ArrayLike, name: str = "einsum") -> FloatArray:
        if self.record_calls:
            self.calls.append(("einsum", name))
        arrays = [np.asarray(operand, dtype=np.float64) for operand in operands]
        if subscripts == "fc,nct->nft" and len(arrays) == 2:
            filters, trials = arrays
            if filters.ndim != 2 or trials.ndim != 3:
                return np.einsum(subscripts, *arrays)
            n_trials, n_channels, n_samples = trials.shape
            if filters.shape[1] != n_channels:
                return np.einsum(subscripts, *arrays)
            flat = trials.transpose(0, 2, 1).reshape(n_trials * n_samples, n_channels)
            projected = self.matmul(flat, filters.T, name=name)
            return projected.reshape(n_trials, n_samples, filters.shape[0]).transpose(0, 2, 1)
        if subscripts == "k,nkc->nc" and len(arrays) == 2:
            weights, probabilities = arrays
            if weights.ndim != 1 or probabilities.ndim != 3:
                return np.einsum(subscripts, *arrays)
            return self.matmul(probabilities.transpose(0, 2, 1), weights, name=name)
        if subscripts == "nd,ncd->nc" and len(arrays) == 2:
            vectors, prototypes = arrays
            if vectors.ndim != 2 or prototypes.ndim != 3:
                return np.einsum(subscripts, *arrays)
            return self.matmul(vectors[:, None, :], prototypes.transpose(0, 2, 1), name=name).squeeze(1)
        if subscripts == "nij,sj->nsi" and len(arrays) == 2:
            weights, samples = arrays
            if weights.ndim != 3 or samples.ndim != 2:
                return np.einsum(subscripts, *arrays)
            expanded_samples = np.broadcast_to(samples, (weights.shape[0], *samples.shape))
            return self.matmul(expanded_samples, weights.transpose(0, 2, 1), name=name)
        return np.einsum(subscripts, *arrays)


def _load_gazelle_model():
    try:
        from osimulator.api import load_gazelle_model

        return load_gazelle_model()
    except Exception:
        return None


def _quantized_photonic_matmul(
    left: ArrayLike,
    right: ArrayLike,
    *,
    config: PhotonicQuantizationConfig,
    gazelle_model,
) -> FloatArray:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape[-1] != (right_arr.shape[0] if right_arr.ndim == 1 else right_arr.shape[-2]):
        return np.matmul(left_arr, right_arr).astype(np.float64)

    left_min = float(np.min(left_arr))
    left_max = float(np.max(left_arr))
    input_range = max(left_max - left_min, config.eps)
    input_scale = input_range / float(config.qinmax - config.qinmin)
    input_zp = int(np.clip(np.round(-left_min / input_scale), config.qinmin, config.qinmax))

    weight_absmax = max(float(np.max(np.abs(right_arr))), config.eps)
    weight_scale = weight_absmax / float(config.qwtmax)

    q_left = np.rint(left_arr / input_scale + input_zp)
    q_left = np.clip(q_left, config.qinmin, config.qinmax).astype(np.int32)
    q_right = np.rint(right_arr / weight_scale)
    q_right = np.clip(q_right, config.qwtmin, config.qwtmax).astype(np.int32)

    raw = _integer_photonic_matmul(
        q_left,
        q_right,
        input_type=config.input_type,
        gazelle_model=gazelle_model,
    ).astype(np.float64)
    right_sum = q_right.sum(axis=-1 if q_right.ndim == 1 else -2, dtype=np.int64).astype(np.float64)
    if right_sum.ndim > 0 and raw.ndim > right_sum.ndim and right_sum.shape[0] == raw.shape[0]:
        right_sum = np.expand_dims(right_sum, axis=-2)
    corrected = raw - float(input_zp) * right_sum
    return (corrected * (input_scale * weight_scale)).astype(np.float64)


def _integer_photonic_matmul(
    q_left: NDArray[np.int32],
    q_right: NDArray[np.int32],
    *,
    input_type: str,
    gazelle_model,
) -> NDArray[np.int64]:
    if gazelle_model is not None:
        model_call = _to_gazelle_batches(q_left, q_right)
        if model_call is not None:
            left_batch, right_batch, restore = model_call
            try:
                result = gazelle_model(left_batch, right_batch, input_type)
                if hasattr(result, "detach"):
                    result = result.detach().cpu().numpy()
                elif hasattr(result, "numpy"):
                    result = result.numpy()
                return restore(np.asarray(result, dtype=np.int64))
            except Exception:
                pass
    return np.matmul(q_left.astype(np.int64), q_right.astype(np.int64))


def _to_gazelle_batches(
    q_left: NDArray[np.int32],
    q_right: NDArray[np.int32],
):
    left = np.asarray(q_left, dtype=np.int32)
    right = np.asarray(q_right, dtype=np.int32)
    squeeze_last = False
    if right.ndim == 1:
        right = right[:, None]
        squeeze_last = True
    if left.ndim == 1:
        left = left[None, None, :]
        squeeze_left_batch = True
        squeeze_left_row = True
    elif left.ndim == 2:
        left = left[None, :, :]
        squeeze_left_batch = True
        squeeze_left_row = False
    elif left.ndim == 3:
        squeeze_left_batch = False
        squeeze_left_row = False
    else:
        return None
    if right.ndim == 2:
        right = right[None, :, :]
        if left.shape[0] != 1:
            right = np.repeat(right, left.shape[0], axis=0)
    elif right.ndim == 3:
        if left.shape[0] == 1 and right.shape[0] != 1:
            left = np.repeat(left, right.shape[0], axis=0)
            squeeze_left_batch = False
        elif right.shape[0] == 1 and left.shape[0] != 1:
            right = np.repeat(right, left.shape[0], axis=0)
        elif right.shape[0] != left.shape[0]:
            return None
    else:
        return None
    if left.shape[2] != right.shape[1]:
        return None

    def restore(result: NDArray[np.int64]) -> NDArray[np.int64]:
        out = result
        if squeeze_left_batch:
            out = out[0]
        if squeeze_left_row:
            out = out[0]
        if squeeze_last:
            out = np.squeeze(out, axis=-1)
        return out

    return left, right, restore


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

    def __init__(
        self,
        tile_shape: tuple[int, int] = (2, 8),
        matrix_backend: MatrixOpsBackend | None = None,
    ):
        tile_rows, tile_cols = tile_shape
        if tile_rows <= 0 or tile_cols <= 0:
            raise ValueError("tile dimensions must be positive")
        self.tile_shape = (int(tile_rows), int(tile_cols))
        self.matrix_backend = matrix_backend or QuantizedPhotonicMatrixOpsBackend()
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
                        self.matrix_backend.matmul(
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
