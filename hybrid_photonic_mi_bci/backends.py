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
from scipy.signal import sosfilt_zi as _scipy_sosfilt_zi
from scipy.signal import sosfiltfilt as _scipy_sosfiltfilt


FloatArray = NDArray[np.float64]


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += ndim
    if not 0 <= normalized < ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return normalized


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
    driver contract. Every matrix product is decomposed into ``2 x 8`` hardware
    tiles and reconstructed from partial sums, while callers continue to pass
    system-level matrices of arbitrary compatible shape.
    """

    def __init__(self, tile_shape: tuple[int, int] = (2, 8), record_calls: bool = False):
        tile_rows, tile_cols = tile_shape
        if tile_rows <= 0 or tile_cols <= 0:
            raise ValueError("tile dimensions must be positive")
        self.tile_shape = (int(tile_rows), int(tile_cols))
        self.record_calls = bool(record_calls)
        self.calls: list[tuple[str, str]] = []
        self.last_tile_count = 0
        self.total_tile_count = 0
        self.execution_backend = "numpy_tiled_photonic_simulation"

    def matmul(self, left: ArrayLike, right: ArrayLike, *, name: str = "matmul") -> FloatArray:
        if self.record_calls:
            self.calls.append(("matmul", name))
        result, tile_count = _tiled_matmul(
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            tile_shape=self.tile_shape,
            tile_executor=np.matmul,
        )
        self.last_tile_count = tile_count
        self.total_tile_count += tile_count
        return result.astype(np.float64)

    def einsum(self, subscripts: str, *operands: ArrayLike, name: str = "einsum") -> FloatArray:
        if self.record_calls:
            self.calls.append(("einsum", name))
        return _einsum_via_matmul(self, subscripts, operands, name=name)


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


@dataclass(frozen=True)
class BitSlicedPhotonicConfig:
    """Logical fixed-point precision reconstructed from physical 4-bit calls."""

    logical_input_bits: int = 8
    logical_weight_bits: int = 8
    slice_bits: int = 4
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.logical_input_bits <= 0 or self.logical_weight_bits <= 1:
            raise ValueError("logical bit widths must be positive")
        if self.slice_bits != 4:
            raise ValueError("the current photonic slice contract is fixed at 4 bits")

    @property
    def input_qmax(self) -> int:
        return (1 << self.logical_input_bits) - 1

    @property
    def weight_qmin(self) -> int:
        return -(1 << (self.logical_weight_bits - 1))

    @property
    def weight_qmax(self) -> int:
        return (1 << (self.logical_weight_bits - 1)) - 1


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
        tile_shape: tuple[int, int] = (2, 8),
    ) -> None:
        self.config = config or PhotonicQuantizationConfig.for_bit_width(4)
        tile_rows, tile_cols = tile_shape
        if tile_rows <= 0 or tile_cols <= 0:
            raise ValueError("tile dimensions must be positive")
        self.tile_shape = (int(tile_rows), int(tile_cols))
        self.record_calls = bool(record_calls)
        self.calls: list[tuple[str, str]] = []
        self.last_tile_count = 0
        self.total_tile_count = 0
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
        result, tile_count = _quantized_photonic_matmul(
            left,
            right,
            config=self.config,
            gazelle_model=self._gazelle_model,
            tile_shape=self.tile_shape,
        )
        self.last_tile_count = tile_count
        self.total_tile_count += tile_count
        return result

    def einsum(self, subscripts: str, *operands: ArrayLike, name: str = "einsum") -> FloatArray:
        if self.record_calls:
            self.calls.append(("einsum", name))
        return _einsum_via_matmul(self, subscripts, operands, name=name)


class BitSlicedPhotonicMatrixOpsBackend(MatrixOpsBackend):
    """Reconstruct higher logical precision from 4-bit photonic tile calls.

    Activations are affine-quantized to an unsigned logical integer and split
    into radix-16 digits in ``[0, 15]``. Signed logical weights use balanced
    radix-16 digits in ``[-8, 7]``. Every digit pair is evaluated by the same
    physical ``2 x 8`` uint4/int4 contract and accumulated with its radix
    weight. This is positional decomposition, not a one-shot 4-bit truncation.
    """

    def __init__(
        self,
        config: BitSlicedPhotonicConfig | None = None,
        *,
        tile_shape: tuple[int, int] = (2, 8),
        record_calls: bool = False,
        use_gazelle_model: bool = True,
        announce: bool = True,
    ) -> None:
        self.config = config or BitSlicedPhotonicConfig()
        tile_rows, tile_cols = tile_shape
        if tile_rows <= 0 or tile_cols <= 0:
            raise ValueError("tile dimensions must be positive")
        self.tile_shape = (int(tile_rows), int(tile_cols))
        self.record_calls = bool(record_calls)
        self.calls: list[tuple[str, str]] = []
        self.last_tile_count = 0
        self.total_tile_count = 0
        self.last_slice_pair_count = 0
        self._gazelle_model = _load_gazelle_model() if use_gazelle_model else None
        self.execution_backend = (
            "gazelle_simulator_bit_sliced"
            if self._gazelle_model is not None
            else "numpy_integer_matmul_bit_sliced"
        )
        if announce:
            print(
                "[photonic-matmul] "
                f"backend={self.execution_backend}, "
                f"logical_input_bits={self.config.logical_input_bits}, "
                f"logical_weight_bits={self.config.logical_weight_bits}, "
                "physical_slice=uint4/int4, "
                f"tile={self.tile_shape}"
            )

    def matmul(self, left: ArrayLike, right: ArrayLike, *, name: str = "matmul") -> FloatArray:
        if self.record_calls:
            self.calls.append(("matmul", name))
        result, tile_count, slice_pair_count = _bit_sliced_photonic_matmul(
            left,
            right,
            config=self.config,
            gazelle_model=self._gazelle_model,
            tile_shape=self.tile_shape,
        )
        self.last_tile_count = tile_count
        self.total_tile_count += tile_count
        self.last_slice_pair_count = slice_pair_count
        return result

    def einsum(self, subscripts: str, *operands: ArrayLike, name: str = "einsum") -> FloatArray:
        if self.record_calls:
            self.calls.append(("einsum", name))
        return _einsum_via_matmul(self, subscripts, operands, name=name)


def _load_gazelle_model():
    try:
        from osimulator.api import load_gazelle_model

        return load_gazelle_model()
    except Exception:
        return None


def _normalize_matmul_operands(
    left: ArrayLike,
    right: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.ndim == 0 or right_arr.ndim == 0:
        raise ValueError("photonic matmul operands must be at least one-dimensional")

    left_was_vector = left_arr.ndim == 1
    right_was_vector = right_arr.ndim == 1
    left_matrix = left_arr[None, :] if left_was_vector else left_arr
    right_matrix = right_arr[:, None] if right_was_vector else right_arr
    if left_matrix.shape[-1] != right_matrix.shape[-2]:
        raise ValueError(
            "photonic matmul inner dimensions must match, "
            f"got {left_arr.shape} and {right_arr.shape}"
        )

    batch_shape = np.broadcast_shapes(left_matrix.shape[:-2], right_matrix.shape[:-2])
    left_matrix = np.broadcast_to(left_matrix, (*batch_shape, *left_matrix.shape[-2:]))
    right_matrix = np.broadcast_to(right_matrix, (*batch_shape, *right_matrix.shape[-2:]))
    return left_matrix, right_matrix, left_was_vector, right_was_vector


def _restore_matmul_shape(
    result: np.ndarray,
    *,
    left_was_vector: bool,
    right_was_vector: bool,
) -> np.ndarray:
    restored = result
    if left_was_vector:
        restored = np.squeeze(restored, axis=-2)
    if right_was_vector:
        restored = np.squeeze(restored, axis=-1)
    return restored


def _tiled_matrix_product(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    *,
    tile_shape: tuple[int, int],
    tile_executor,
) -> tuple[np.ndarray, int]:
    tile_rows, tile_cols = tile_shape
    batch_shape = left_matrix.shape[:-2]
    n_vectors = left_matrix.shape[-2]
    inner_dim = left_matrix.shape[-1]
    out_dim = right_matrix.shape[-1]
    output_dtype = np.result_type(left_matrix.dtype, right_matrix.dtype)
    output = np.zeros((*batch_shape, n_vectors, out_dim), dtype=output_dtype)

    for out_start in range(0, out_dim, tile_rows):
        out_stop = min(out_start + tile_rows, out_dim)
        for inner_start in range(0, inner_dim, tile_cols):
            inner_stop = min(inner_start + tile_cols, inner_dim)
            output[..., out_start:out_stop] += tile_executor(
                left_matrix[..., inner_start:inner_stop],
                right_matrix[..., inner_start:inner_stop, out_start:out_stop],
            )

    batch_count = int(np.prod(batch_shape, dtype=np.int64)) if batch_shape else 1
    tile_count = (
        batch_count
        * n_vectors
        * int(np.ceil(out_dim / tile_rows))
        * int(np.ceil(inner_dim / tile_cols))
    )
    return output, tile_count


def _tiled_matmul(
    left: ArrayLike,
    right: ArrayLike,
    *,
    tile_shape: tuple[int, int],
    tile_executor,
) -> tuple[np.ndarray, int]:
    left_matrix, right_matrix, left_was_vector, right_was_vector = (
        _normalize_matmul_operands(left, right)
    )
    result, tile_count = _tiled_matrix_product(
        left_matrix,
        right_matrix,
        tile_shape=tile_shape,
        tile_executor=tile_executor,
    )
    return (
        _restore_matmul_shape(
            result,
            left_was_vector=left_was_vector,
            right_was_vector=right_was_vector,
        ),
        tile_count,
    )


def _einsum_via_matmul(
    backend: MatrixOpsBackend,
    subscripts: str,
    operands: tuple[ArrayLike, ...],
    *,
    name: str,
) -> FloatArray:
    arrays = [np.asarray(operand, dtype=np.float64) for operand in operands]
    if subscripts == "fc,nct->nft" and len(arrays) == 2:
        filters, trials = arrays
        if filters.ndim != 2 or trials.ndim != 3:
            raise ValueError("fc,nct->nft expects filters=(F,C), trials=(N,C,T)")
        n_trials, n_channels, n_samples = trials.shape
        if filters.shape[1] != n_channels:
            raise ValueError("CSP filter and trial channel dimensions must match")
        flat = trials.transpose(0, 2, 1).reshape(n_trials * n_samples, n_channels)
        projected = backend.matmul(flat, filters.T, name=name)
        return projected.reshape(n_trials, n_samples, filters.shape[0]).transpose(0, 2, 1)
    if subscripts == "k,nkc->nc" and len(arrays) == 2:
        weights, probabilities = arrays
        if weights.ndim != 1 or probabilities.ndim != 3:
            raise ValueError("k,nkc->nc expects weights=(K,), probabilities=(N,K,C)")
        return backend.matmul(probabilities.transpose(0, 2, 1), weights, name=name)
    if subscripts == "nd,ncd->nc" and len(arrays) == 2:
        vectors, prototypes = arrays
        if vectors.ndim != 2 or prototypes.ndim != 3:
            raise ValueError("nd,ncd->nc expects vectors=(N,D), prototypes=(N,C,D)")
        return backend.matmul(
            vectors[:, None, :],
            prototypes.transpose(0, 2, 1),
            name=name,
        ).squeeze(1)
    if subscripts == "nij,sj->nsi" and len(arrays) == 2:
        weights, samples = arrays
        if weights.ndim != 3 or samples.ndim != 2:
            raise ValueError("nij,sj->nsi expects weights=(N,I,J), samples=(S,J)")
        expanded_samples = np.broadcast_to(samples, (weights.shape[0], *samples.shape))
        return backend.matmul(expanded_samples, weights.transpose(0, 2, 1), name=name)
    raise NotImplementedError(
        f"photonic matrix backend does not expose einsum pattern {subscripts!r}"
    )


def _quantized_photonic_matmul(
    left: ArrayLike,
    right: ArrayLike,
    *,
    config: PhotonicQuantizationConfig,
    gazelle_model,
    tile_shape: tuple[int, int],
) -> tuple[FloatArray, int]:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    _normalize_matmul_operands(left_arr, right_arr)

    left_min = min(float(np.min(left_arr)), 0.0)
    left_max = max(float(np.max(left_arr)), 0.0)
    input_range = max(left_max - left_min, config.eps)
    input_scale = input_range / float(config.qinmax - config.qinmin)
    input_zp = int(np.clip(np.round(-left_min / input_scale), config.qinmin, config.qinmax))

    weight_absmax = max(float(np.max(np.abs(right_arr))), config.eps)
    weight_scale = weight_absmax / float(config.qwtmax)

    q_left = np.rint(left_arr / input_scale + input_zp)
    q_left = np.clip(q_left, config.qinmin, config.qinmax).astype(np.int32)
    q_right = np.rint(right_arr / weight_scale)
    q_right = np.clip(q_right, config.qwtmin, config.qwtmax).astype(np.int32)

    q_left_matrix, q_right_matrix, left_was_vector, right_was_vector = (
        _normalize_matmul_operands(q_left, q_right)
    )
    raw, tile_count = _tiled_matrix_product(
        q_left_matrix.astype(np.int64),
        q_right_matrix.astype(np.int64),
        tile_shape=tile_shape,
        tile_executor=lambda tile_left, tile_right: _integer_photonic_matmul(
            tile_left.astype(np.int32),
            tile_right.astype(np.int32),
            input_type=config.input_type,
            gazelle_model=gazelle_model,
        ),
    )
    right_sum = q_right_matrix.sum(axis=-2, dtype=np.int64).astype(np.float64)
    corrected = raw.astype(np.float64) - float(input_zp) * right_sum[..., None, :]
    restored = _restore_matmul_shape(
        corrected,
        left_was_vector=left_was_vector,
        right_was_vector=right_was_vector,
    )
    return (restored * (input_scale * weight_scale)).astype(np.float64), tile_count


def _unsigned_radix_slices(values: NDArray[np.int64], *, radix: int) -> list[NDArray[np.int32]]:
    if np.any(values < 0):
        raise ValueError("unsigned photonic slices cannot contain negative values")
    remaining = values.copy()
    slices: list[NDArray[np.int32]] = []
    while np.any(remaining != 0) or not slices:
        slices.append(np.remainder(remaining, radix).astype(np.int32))
        remaining = np.floor_divide(remaining, radix)
    return slices


def _balanced_radix_slices(values: NDArray[np.int64], *, radix: int) -> list[NDArray[np.int32]]:
    half_radix = radix // 2
    remaining = values.copy()
    slices: list[NDArray[np.int32]] = []
    while np.any(remaining != 0) or not slices:
        digit = np.remainder(remaining + half_radix, radix) - half_radix
        slices.append(digit.astype(np.int32))
        remaining = np.floor_divide(remaining - digit, radix)
    return slices


def _bit_sliced_photonic_matmul(
    left: ArrayLike,
    right: ArrayLike,
    *,
    config: BitSlicedPhotonicConfig,
    gazelle_model,
    tile_shape: tuple[int, int],
) -> tuple[FloatArray, int, int]:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    _normalize_matmul_operands(left_arr, right_arr)

    left_min = min(float(np.min(left_arr)), 0.0)
    left_max = max(float(np.max(left_arr)), 0.0)
    input_range = max(left_max - left_min, config.eps)
    input_scale = input_range / float(config.input_qmax)
    input_zp = int(np.clip(np.round(-left_min / input_scale), 0, config.input_qmax))
    q_left = np.rint(left_arr / input_scale + input_zp)
    q_left = np.clip(q_left, 0, config.input_qmax).astype(np.int64)

    weight_absmax = max(float(np.max(np.abs(right_arr))), config.eps)
    weight_scale = weight_absmax / float(config.weight_qmax)
    q_right = np.rint(right_arr / weight_scale)
    q_right = np.clip(
        q_right,
        config.weight_qmin,
        config.weight_qmax,
    ).astype(np.int64)

    q_left_matrix, q_right_matrix, left_was_vector, right_was_vector = (
        _normalize_matmul_operands(q_left, q_right)
    )
    radix = 1 << config.slice_bits
    left_slices = _unsigned_radix_slices(q_left_matrix, radix=radix)
    right_slices = _balanced_radix_slices(q_right_matrix, radix=radix)
    accumulated = np.zeros(
        (*left_slices[0].shape[:-1], right_slices[0].shape[-1]),
        dtype=np.int64,
    )
    total_tile_count = 0
    slice_pair_count = 0
    for left_index, left_slice in enumerate(left_slices):
        for right_index, right_slice in enumerate(right_slices):
            if not np.any(left_slice) or not np.any(right_slice):
                continue
            partial, tile_count = _tiled_matrix_product(
                left_slice.astype(np.int64),
                right_slice.astype(np.int64),
                tile_shape=tile_shape,
                tile_executor=lambda tile_left, tile_right: _integer_photonic_matmul(
                    tile_left.astype(np.int32),
                    tile_right.astype(np.int32),
                    input_type="uint4",
                    gazelle_model=gazelle_model,
                ),
            )
            accumulated += partial.astype(np.int64) * (radix ** (left_index + right_index))
            total_tile_count += tile_count
            slice_pair_count += 1

    right_sum = q_right_matrix.sum(axis=-2, dtype=np.int64)
    corrected = accumulated - int(input_zp) * right_sum[..., None, :]
    restored = _restore_matmul_shape(
        corrected,
        left_was_vector=left_was_vector,
        right_was_vector=right_was_vector,
    )
    return (
        (restored.astype(np.float64) * (input_scale * weight_scale)).astype(np.float64),
        total_tile_count,
        slice_pair_count,
    )


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
    """Software stand-in for photonic forward signal-processing operations.

    CAR is expanded into an explicit channel-mixing matrix and therefore uses
    the installed tiled MatrixOps backend. SOS filtering remains a named signal
    operator because its recursive state and boundary handling belong to the
    future streaming hardware driver rather than to one dense matrix product.
    """

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
        normalized_axis = _normalize_axis(channel_axis, x.ndim)
        moved = np.moveaxis(x, normalized_axis, -1)
        n_channels = moved.shape[-1]
        car_matrix = np.eye(n_channels, dtype=np.float64) - np.full(
            (n_channels, n_channels),
            1.0 / n_channels,
            dtype=np.float64,
        )
        flat = moved.reshape(-1, n_channels)
        referenced = matrix_multiply(flat, car_matrix.T, name=name).reshape(moved.shape)
        return np.moveaxis(referenced, -1, normalized_axis)

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
        return _photonic_sosfiltfilt(
            np.asarray(sos, dtype=np.float64),
            np.asarray(samples, dtype=np.float64),
            axis=axis,
            name=name,
        )


def _photonic_sosfiltfilt(
    sos: FloatArray,
    samples: FloatArray,
    *,
    axis: int,
    name: str,
) -> FloatArray:
    """Zero-phase SOS filtering with all section MACs routed through MatrixOps."""

    x = np.asarray(samples, dtype=np.float64)
    coefficients = np.asarray(sos, dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[1] != 6:
        raise ValueError(f"sos must have shape (n_sections, 6), got {coefficients.shape}")
    if not np.allclose(coefficients[:, 3], 1.0):
        coefficients = coefficients / coefficients[:, 3:4]

    normalized_axis = _normalize_axis(axis, x.ndim)
    moved = np.moveaxis(x, normalized_axis, -1)
    n_samples = moved.shape[-1]
    n_sections = coefficients.shape[0]
    ntaps = 2 * n_sections + 1
    ntaps -= min(
        int((coefficients[:, 2] == 0).sum()),
        int((coefficients[:, 5] == 0).sum()),
    )
    edge = 3 * ntaps
    if n_samples <= edge:
        raise ValueError(
            f"input length {n_samples} must be greater than photonic filter pad length {edge}"
        )

    flat = moved.reshape(-1, n_samples)
    left_extension = 2.0 * flat[:, :1] - flat[:, edge:0:-1]
    right_extension = 2.0 * flat[:, -1:] - flat[:, -2 : -edge - 2 : -1]
    extended = np.concatenate([left_extension, flat, right_extension], axis=1)

    zi = np.asarray(_scipy_sosfilt_zi(coefficients), dtype=np.float64)
    forward = _photonic_sosfilt(
        coefficients,
        extended,
        zi=zi,
        initial_value=extended[:, 0],
        name=f"{name}_forward",
    )
    backward_input = forward[:, ::-1]
    backward = _photonic_sosfilt(
        coefficients,
        backward_input,
        zi=zi,
        initial_value=forward[:, -1],
        name=f"{name}_backward",
    )[:, ::-1]
    filtered = backward[:, edge:-edge].reshape(moved.shape)
    return np.moveaxis(filtered, -1, normalized_axis)


def _photonic_sosfilt(
    sos: FloatArray,
    signals: FloatArray,
    *,
    zi: FloatArray,
    initial_value: FloatArray,
    name: str,
) -> FloatArray:
    n_signals, n_samples = signals.shape
    n_sections = sos.shape[0]
    states = np.zeros((n_signals, n_sections, 2), dtype=np.float64)
    for section_index in range(n_sections):
        states[:, section_index] = matrix_multiply(
            initial_value[:, None],
            zi[section_index][None, :],
            name=f"{name}_initial_state_s{section_index}",
        )

    transitions = np.zeros((n_sections, 3, 3), dtype=np.float64)
    for section_index, section in enumerate(sos):
        b0, b1, b2, _a0, a1, a2 = section
        transitions[section_index] = np.array(
            [
                [b0, 1.0, 0.0],
                [b1 - a1 * b0, -a1, 1.0],
                [b2 - a2 * b0, -a2, 0.0],
            ],
            dtype=np.float64,
        )

    output = np.zeros_like(signals, dtype=np.float64)
    for sample_index in range(n_samples):
        current = signals[:, sample_index]
        for section_index in range(n_sections):
            section_input = np.column_stack(
                [
                    current,
                    states[:, section_index, 0],
                    states[:, section_index, 1],
                ]
            )
            section_output = matrix_multiply(
                section_input,
                transitions[section_index].T,
                name=f"{name}_section_s{section_index}",
            )
            current = section_output[:, 0]
            states[:, section_index] = section_output[:, 1:]
        output[:, sample_index] = current
    return output


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


_MATRIX_OPS_BACKEND: MatrixOpsBackend = BitSlicedPhotonicMatrixOpsBackend(announce=False)
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

    w = np.asarray(weights, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    if w.ndim != 3 or x.ndim != 1:
        raise ValueError("batched_matrix_vector expects weights=(N,M,D), features=(D,)")
    expanded = np.broadcast_to(x, (w.shape[0], 1, x.shape[0]))
    return matrix_multiply(expanded, w.transpose(0, 2, 1), name=name).squeeze(1)


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
                    feature_tile = features_arr[col_start:col_stop][None, :]
                    weight_tile = weights_arr[
                        candidate_index,
                        row_start:row_stop,
                        col_start:col_stop,
                    ]
                    output[candidate_index, row_start:row_stop] += self.matrix_backend.matmul(
                        feature_tile,
                        weight_tile.T,
                        name="tiled_candidate_scan_tile",
                    )[0]
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
