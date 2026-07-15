"""Filter-bank CSP feature extraction for motor-imagery EEG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh
from scipy.signal import butter

from .backends import covariance_gram, csp_spatial_project
from .backends import signal_sosfiltfilt


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

DEFAULT_FILTER_BANK = (
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
)


@dataclass(frozen=True)
class FBCSPFeatureSet:
    """Vector and tensor views of FBCSP features."""

    vector: FloatArray
    tensor: FloatArray
    feature_names: tuple[str, ...]
    bands: tuple[tuple[float, float], ...]
    class_names: tuple[str, ...]


class FilterBankCSP:
    """One-vs-rest FBCSP with log-variance features."""

    def __init__(
        self,
        bands: Iterable[tuple[float, float]] = DEFAULT_FILTER_BANK,
        n_components: int = 2,
        filter_order: int = 3,
        covariance_shrinkage: float = 0.10,
    ):
        self.bands = tuple((float(low), float(high)) for low, high in bands)
        if not self.bands:
            raise ValueError("at least one filter-bank band is required")
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if not 0.0 <= covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1]")
        if filter_order <= 0:
            raise ValueError("filter_order must be positive")
        self.n_components = int(n_components)
        self.filter_order = int(filter_order)
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.filters_: FloatArray | None = None
        self.class_names_: tuple[str, ...] | None = None
        self.fs_: float | None = None

    def fit(
        self,
        trials: ArrayLike,
        labels: ArrayLike,
        fs: float,
        class_names: tuple[str, ...],
    ) -> "FilterBankCSP":
        x = _as_trials(trials)
        y = np.asarray(labels, dtype=int)
        if y.shape != (x.shape[0],):
            raise ValueError(f"labels must have shape ({x.shape[0]},), got {y.shape}")
        if fs <= 0:
            raise ValueError("fs must be positive")
        _validate_bands(self.bands, fs)

        n_classes = len(class_names)
        filters = np.zeros(
            (
                len(self.bands),
                n_classes,
                2 * self.n_components,
                x.shape[1],
            ),
            dtype=np.float64,
        )
        for band_index, band in enumerate(self.bands):
            band_trials = _bandpass_trials(
                x,
                fs=fs,
                band=band,
                order=self.filter_order,
            )
            covariances = _trial_covariances(band_trials)
            for class_index in range(n_classes):
                positive = covariances[y == class_index]
                negative = covariances[y != class_index]
                if len(positive) == 0 or len(negative) == 0:
                    raise ValueError(
                        f"class {class_names[class_index]!r} cannot form one-vs-rest CSP"
                    )
                cov_pos = _shrink_covariance(positive.mean(axis=0), self.covariance_shrinkage)
                cov_neg = _shrink_covariance(negative.mean(axis=0), self.covariance_shrinkage)
                composite = cov_pos + cov_neg + np.eye(cov_pos.shape[0]) * 1e-9
                eigenvalues, eigenvectors = eigh(cov_pos, composite)
                order = np.argsort(eigenvalues)
                selected = np.concatenate(
                    [
                        order[: self.n_components],
                        order[-self.n_components :],
                    ]
                )
                filters[band_index, class_index] = eigenvectors[:, selected].T
        self.filters_ = filters
        self.class_names_ = tuple(class_names)
        self.fs_ = float(fs)
        return self

    def transform(self, trials: ArrayLike) -> FBCSPFeatureSet:
        if self.filters_ is None or self.class_names_ is None or self.fs_ is None:
            raise RuntimeError("FilterBankCSP must be fitted before transform")
        x = _as_trials(trials)
        n_trials = x.shape[0]
        n_bands, n_classes, n_filters, _ = self.filters_.shape
        tensor = np.zeros((n_trials, n_bands, n_classes, n_filters), dtype=np.float64)
        for band_index, band in enumerate(self.bands):
            band_trials = _bandpass_trials(
                x,
                fs=self.fs_,
                band=band,
                order=self.filter_order,
            )
            for class_index in range(n_classes):
                spatial = csp_spatial_project(
                    self.filters_[band_index, class_index],
                    band_trials,
                    name=(
                        "fbcsp_spatial_projection_"
                        f"b{band_index}_c{class_index}"
                    ),
                )
                variances = np.var(spatial, axis=2) + 1e-10
                normalized = variances / variances.sum(axis=1, keepdims=True)
                tensor[:, band_index, class_index] = np.log(normalized)
        vector = tensor.reshape(n_trials, -1)
        return FBCSPFeatureSet(
            vector=vector,
            tensor=tensor,
            feature_names=self._feature_names(),
            bands=self.bands,
            class_names=self.class_names_,
        )

    def fit_transform(
        self,
        trials: ArrayLike,
        labels: ArrayLike,
        fs: float,
        class_names: tuple[str, ...],
    ) -> FBCSPFeatureSet:
        return self.fit(trials, labels, fs, class_names).transform(trials)

    def _feature_names(self) -> tuple[str, ...]:
        if self.class_names_ is None:
            raise RuntimeError("FilterBankCSP must be fitted before feature names are available")
        names = []
        for low, high in self.bands:
            for class_name in self.class_names_:
                for component in range(2 * self.n_components):
                    names.append(f"{low:.0f}-{high:.0f}Hz_{class_name}_csp{component}")
        return tuple(names)


def _as_trials(trials: ArrayLike) -> FloatArray:
    x = np.asarray(trials, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"trials must have shape (N, C, T), got {x.shape}")
    return x


def _validate_bands(bands: tuple[tuple[float, float], ...], fs: float) -> None:
    for band in bands:
        if not 0.0 < band[0] < band[1] < fs / 2:
            raise ValueError(f"invalid band {band} for fs={fs}")


def _bandpass_trials(
    trials: FloatArray,
    fs: float,
    band: tuple[float, float],
    order: int,
) -> FloatArray:
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    return signal_sosfiltfilt(
        sos,
        trials,
        axis=2,
        name=f"fbcsp_filter_bank_{band[0]:g}_{band[1]:g}hz_sosfiltfilt",
    )


def _trial_covariances(trials: FloatArray) -> FloatArray:
    covariances = np.zeros((trials.shape[0], trials.shape[1], trials.shape[1]), dtype=np.float64)
    for index, trial in enumerate(trials):
        centered = trial - trial.mean(axis=1, keepdims=True)
        covariance = covariance_gram(centered, name="fbcsp_trial_covariance") / max(
            1,
            centered.shape[1] - 1,
        )
        trace = float(np.trace(covariance))
        covariances[index] = covariance / trace if trace > 1e-12 else covariance
    return covariances


def _shrink_covariance(covariance: FloatArray, shrinkage: float) -> FloatArray:
    diagonal_target = np.eye(covariance.shape[0]) * (np.trace(covariance) / covariance.shape[0])
    return (1.0 - shrinkage) * covariance + shrinkage * diagonal_target
