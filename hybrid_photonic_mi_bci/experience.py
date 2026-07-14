"""Experience-library retrieval and candidate linear-head scanning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backends import candidate_probability_fusion, prototype_distances
from .backends import MVMBackend, TiledMVMBackend
from .evaluation import softmax
from .linear_models import LinearHead, ShrinkageLDA


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
ScanProgressCallback = Callable[[int, int, FloatArray], None]


@dataclass(frozen=True)
class ExperienceEntry:
    entry_id: str
    centroid: FloatArray
    head: LinearHead
    source: str
    train_indices: IntArray
    train_accuracy: float


@dataclass(frozen=True)
class ExperienceScanResult:
    selected_entries: tuple[ExperienceEntry, ...]
    retrieval_weights: FloatArray
    candidate_scores: FloatArray
    fused_scores: FloatArray
    tile_count_per_window: int


def build_bootstrap_experience_library(
    embeddings: ArrayLike,
    labels: ArrayLike,
    class_names: tuple[str, ...],
    n_entries: int = 32,
    sample_fraction: float = 0.75,
    shrinkage: float = 0.15,
    seed: int = 13,
) -> tuple[ExperienceEntry, ...]:
    """Build a first experience library from class-balanced bootstrap subsets."""

    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    entries: list[ExperienceEntry] = []
    n_classes = len(class_names)
    class_indices = [np.flatnonzero(y == class_index) for class_index in range(n_classes)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("every class needs at least one sample for experience entries")
    per_class = [max(2, int(len(indices) * sample_fraction)) for indices in class_indices]
    for entry_index in range(n_entries):
        sampled = []
        for indices, count in zip(class_indices, per_class):
            sampled.extend(rng.choice(indices, size=count, replace=True).tolist())
        train_indices = rng.permutation(np.asarray(sampled, dtype=int))
        lda = ShrinkageLDA(shrinkage=shrinkage).fit(
            x[train_indices],
            y[train_indices],
            class_names=class_names,
        )
        scores = lda.scores(x[train_indices])
        accuracy = float((scores.argmax(axis=1) == y[train_indices]).mean())
        entries.append(
            ExperienceEntry(
                entry_id=f"bootstrap_{entry_index:03d}",
                centroid=x[train_indices].mean(axis=0),
                head=lda.head,
                source="bootstrap",
                train_indices=train_indices,
                train_accuracy=accuracy,
            )
        )
    return tuple(entries)


def retrieve_top_k(
    entries: tuple[ExperienceEntry, ...],
    calibration_embeddings: ArrayLike,
    k: int = 8,
) -> tuple[tuple[ExperienceEntry, ...], FloatArray, FloatArray]:
    """Retrieve nearest experience entries by centroid distance."""

    if not entries:
        raise ValueError("experience library is empty")
    query = np.asarray(calibration_embeddings, dtype=np.float64).mean(axis=0)
    centroids = np.stack([entry.centroid for entry in entries], axis=0)
    distances = prototype_distances(
        query[None, :],
        centroids,
        name="experience_retrieval_centroid_distance",
    )[0]
    order = np.argsort(distances)[: min(k, len(entries))]
    selected = tuple(entries[int(index)] for index in order)
    selected_distances = distances[order]
    logits = -selected_distances / (selected_distances.std() + 1e-6)
    weights = np.exp(logits - logits.max())
    weights = weights / weights.sum()
    return selected, weights.astype(np.float64), selected_distances.astype(np.float64)


def scan_experience_heads(
    entries: tuple[ExperienceEntry, ...],
    retrieval_weights: ArrayLike,
    embeddings: ArrayLike,
    backend: MVMBackend | None = None,
    progress_callback: ScanProgressCallback | None = None,
) -> ExperienceScanResult:
    """Scan candidate linear heads and fuse their probability outputs."""

    selected_weights = np.asarray(retrieval_weights, dtype=np.float64)
    x = np.asarray(embeddings, dtype=np.float64)
    if len(entries) != len(selected_weights):
        raise ValueError("entries and retrieval_weights length mismatch")
    backend = backend or TiledMVMBackend(tile_shape=(2, 8))
    weights = np.stack([entry.head.weights for entry in entries], axis=0)
    bias = np.stack([entry.head.bias for entry in entries], axis=0)
    augmented_weights = np.concatenate([weights, bias[:, :, None]], axis=2)
    candidate_scores = np.zeros((x.shape[0], len(entries), weights.shape[1]), dtype=np.float64)
    tile_count = 0
    for index, feature in enumerate(x):
        augmented_feature = np.concatenate([feature, np.ones(1, dtype=np.float64)])
        candidate_scores[index] = backend.scan(augmented_weights, augmented_feature)
        if hasattr(backend, "last_tile_count"):
            tile_count = int(getattr(backend, "last_tile_count"))
        if progress_callback is not None:
            window_probs = softmax(candidate_scores[index])
            fused_probability = np.sum(selected_weights[:, None] * window_probs, axis=0)
            progress_callback(index + 1, x.shape[0], fused_probability.astype(np.float64))
    candidate_probs = softmax(candidate_scores.reshape(-1, candidate_scores.shape[-1])).reshape(
        candidate_scores.shape
    )
    fused_probs = candidate_probability_fusion(
        selected_weights,
        candidate_probs,
        name="experience_probability_fusion",
    )
    fused_scores = np.log(fused_probs + 1e-12)
    return ExperienceScanResult(
        selected_entries=entries,
        retrieval_weights=selected_weights,
        candidate_scores=candidate_scores,
        fused_scores=fused_scores,
        tile_count_per_window=tile_count,
    )
