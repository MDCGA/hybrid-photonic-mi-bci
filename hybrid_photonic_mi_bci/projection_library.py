"""Candidate projection matrix library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ProjectionCandidate:
    """One candidate calibration/projection state."""

    weights: FloatArray
    label: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 2:
            raise ValueError(f"candidate weights must have shape (M, D), got {weights.shape}")
        object.__setattr__(self, "weights", weights)


class ProjectionLibrary:
    """A collection of candidate matrices scanned per EEG window.

    The current BCICIV baseline uses ``2 x 8`` matrices because it compresses
    eight EEG features into a two-dimensional discriminant space. That is an
    algorithm choice, not a hard limit of the photonic backend.
    """

    def __init__(self, candidates: Iterable[ProjectionCandidate]):
        self._candidates = tuple(candidates)
        if not self._candidates:
            raise ValueError("ProjectionLibrary requires at least one candidate")
        first_shape = self._candidates[0].weights.shape
        for candidate in self._candidates:
            if candidate.weights.shape != first_shape:
                raise ValueError("all projection candidates must have the same matrix shape")

    @property
    def candidates(self) -> tuple[ProjectionCandidate, ...]:
        return self._candidates

    @property
    def size(self) -> int:
        return len(self._candidates)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(candidate.label for candidate in self._candidates)

    @property
    def weights(self) -> FloatArray:
        return np.stack([candidate.weights for candidate in self._candidates], axis=0)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> ProjectionCandidate:
        return self._candidates[index]

    @classmethod
    def from_array(
        cls,
        weights: ArrayLike,
        labels: Iterable[str] | None = None,
        metadata: Iterable[Mapping[str, object]] | None = None,
    ) -> "ProjectionLibrary":
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.ndim != 3:
            raise ValueError(f"weights must have shape (N, M, D), got {weights_arr.shape}")

        labels_tuple = tuple(labels) if labels is not None else tuple(
            f"candidate_{i:03d}" for i in range(weights_arr.shape[0])
        )
        metadata_tuple = tuple(metadata) if metadata is not None else tuple(
            {} for _ in range(weights_arr.shape[0])
        )
        if len(labels_tuple) != weights_arr.shape[0]:
            raise ValueError("labels length must match number of candidates")
        if len(metadata_tuple) != weights_arr.shape[0]:
            raise ValueError("metadata length must match number of candidates")

        return cls(
            ProjectionCandidate(weights_arr[i], labels_tuple[i], metadata_tuple[i])
            for i in range(weights_arr.shape[0])
        )

    @classmethod
    def random_around(
        cls,
        base_weights: ArrayLike,
        n_candidates: int,
        noise_scale: float = 0.05,
        seed: int | None = None,
    ) -> "ProjectionLibrary":
        """Create a small candidate bank around one baseline matrix.

        This is useful before real offline-trained candidate matrices exist.
        """

        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        base = np.asarray(base_weights, dtype=np.float64)
        if base.ndim != 2:
            raise ValueError(f"base_weights must have shape (M, D), got {base.shape}")

        rng = np.random.default_rng(seed)
        weights = base + rng.normal(scale=noise_scale, size=(n_candidates, *base.shape))
        weights[0] = base
        return cls.from_array(
            weights,
            labels=[f"perturb_{i:03d}" if i else "baseline" for i in range(n_candidates)],
            metadata=[{"source": "random_around", "noise_scale": noise_scale} for _ in range(n_candidates)],
        )
