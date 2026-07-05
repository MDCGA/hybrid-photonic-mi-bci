"""Shared helpers for feature-level replay experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .backends import NumpyMVMBackend
from .calibration import ConfidenceSelector, EpsilonGreedyBandit, ProbabilityFusionSelector
from .decision import DecisionConfig, PrototypeDecisionHead
from .features import Standardizer
from .pipeline import HybridBCIPipeline
from .projection_library import ProjectionLibrary


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class PipelineBuildConfig:
    """Configuration for feature-level candidate scan pipelines."""

    n_train: int = 120
    n_candidates: int = 32
    candidate_noise: float = 0.02
    library_kind: str = "perturb"
    epsilon: float = 0.08
    temperature: float = 0.8
    reject_threshold: float = 0.34
    margin_threshold: float = 0.0
    selector_kind: str = "bandit"
    confidence_weight: float = 0.25
    selector_margin_weight: float = 0.10
    rejected_penalty: float = 0.50
    fusion_value_weight: float = 0.75
    prototype_kind: str = "candidate"
    seed: int = 13


def build_pipeline_from_features(
    features: FloatArray,
    labels: IntArray,
    class_names: tuple[str, ...],
    config: PipelineBuildConfig,
) -> HybridBCIPipeline:
    """Build the hybrid candidate-scan pipeline from fixed 8-D features."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or x.shape[1] != 8:
        raise ValueError(f"features must have shape (N, 8), got {x.shape}")
    if y.shape != (x.shape[0],):
        raise ValueError(f"labels must have shape ({x.shape[0]},), got {y.shape}")
    if config.n_train <= 0 or config.n_train >= len(y):
        raise ValueError("n_train must leave at least one replay sample")

    standardizer = Standardizer().fit(x[: config.n_train])
    x_train = standardizer.transform(x[: config.n_train])
    y_train = y[: config.n_train]
    n_classes = len(class_names)

    base_w = least_squares_projection(x_train, y_train, n_classes=n_classes)
    library = build_projection_library(
        x_train=x_train,
        y_train=y_train,
        base_weights=base_w,
        n_candidates=config.n_candidates,
        candidate_noise=config.candidate_noise,
        library_kind=config.library_kind,
        seed=config.seed,
        n_classes=n_classes,
    )

    if config.prototype_kind == "shared":
        projected_train = x_train @ base_w.T
        prototypes = class_prototypes(projected_train, y_train, n_classes=n_classes)
    elif config.prototype_kind == "candidate":
        projected_by_candidate = np.einsum("nij,sj->nsi", library.weights, x_train)
        prototypes = np.stack(
            [
                class_prototypes(projected_by_candidate[i], y_train, n_classes=n_classes)
                for i in range(len(library))
            ],
            axis=0,
        )
    else:
        raise ValueError(f"unknown prototype_kind: {config.prototype_kind}")

    decision_head = PrototypeDecisionHead(
        prototypes=prototypes,
        config=DecisionConfig(
            class_names=class_names,
            temperature=config.temperature,
            reject_threshold=config.reject_threshold,
            margin_threshold=config.margin_threshold,
        ),
    )
    selector = build_selector(config=config, n_candidates=len(library))
    return HybridBCIPipeline(
        projection_library=library,
        backend=NumpyMVMBackend(),
        decision_head=decision_head,
        selector=selector,
        standardizer=standardizer,
    )


def build_selector(
    config: PipelineBuildConfig,
    n_candidates: int,
) -> EpsilonGreedyBandit | ConfidenceSelector | ProbabilityFusionSelector:
    """Create a candidate selector/fusion policy."""

    if config.selector_kind == "bandit":
        return EpsilonGreedyBandit(
            n_candidates=n_candidates,
            epsilon=config.epsilon,
            seed=config.seed + 6,
            confidence_weight=config.confidence_weight,
            margin_weight=config.selector_margin_weight,
            rejected_penalty=config.rejected_penalty,
        )
    if config.selector_kind == "confidence":
        return ConfidenceSelector()
    if config.selector_kind == "fusion":
        return ProbabilityFusionSelector(
            n_candidates=n_candidates,
            reject_threshold=config.reject_threshold,
            margin_threshold=config.margin_threshold,
            value_weight=config.fusion_value_weight,
            confidence_weight=config.confidence_weight,
            margin_weight=config.selector_margin_weight,
            rejected_penalty=config.rejected_penalty,
        )
    raise ValueError(f"unknown selector_kind: {config.selector_kind}")


def class_targets(n_classes: int) -> FloatArray:
    """Return 2-D target coordinates for least-squares projection fitting."""

    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if n_classes == 2:
        return np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    angles = 2 * np.pi * np.arange(n_classes) / n_classes
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float64)


def least_squares_projection(
    features: FloatArray,
    labels: IntArray,
    n_classes: int,
) -> FloatArray:
    """Fit the current baseline's two-dimensional discriminant projection."""

    targets = class_targets(n_classes)[labels]
    weights, *_ = np.linalg.lstsq(features, targets, rcond=None)
    return weights.T


def class_prototypes(
    projected_features: FloatArray,
    labels: IntArray,
    n_classes: int,
) -> FloatArray:
    """Compute one 2-D class prototype per class."""

    return np.stack(
        [projected_features[labels == class_idx].mean(axis=0) for class_idx in range(n_classes)],
        axis=0,
    )


def build_projection_library(
    x_train: FloatArray,
    y_train: IntArray,
    base_weights: FloatArray,
    n_candidates: int,
    candidate_noise: float,
    library_kind: str,
    seed: int,
    n_classes: int,
) -> ProjectionLibrary:
    """Build a candidate bank around one fitted baseline projection."""

    if library_kind == "perturb":
        return ProjectionLibrary.random_around(
            base_weights=base_weights,
            n_candidates=n_candidates,
            noise_scale=candidate_noise,
            seed=seed,
        )
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")

    rng = np.random.default_rng(seed)
    weights = [base_weights]
    labels = ["baseline"]
    metadata = [{"source": "baseline"}]
    while len(weights) < n_candidates:
        if library_kind == "bootstrap":
            candidate = _bootstrap_projection(x_train, y_train, rng, n_classes=n_classes)
            label = f"bootstrap_{len(weights):03d}"
            source = "bootstrap"
        elif library_kind == "mixed":
            if len(weights) % 2:
                candidate = _bootstrap_projection(x_train, y_train, rng, n_classes=n_classes)
                label = f"bootstrap_{len(weights):03d}"
                source = "bootstrap"
            else:
                candidate = base_weights.copy()
                label = f"perturb_{len(weights):03d}"
                source = "perturb"
        else:
            raise ValueError(f"unknown library_kind: {library_kind}")

        if candidate_noise > 0:
            candidate = candidate + rng.normal(scale=candidate_noise, size=candidate.shape)
        weights.append(candidate)
        labels.append(label)
        metadata.append({"source": source, "candidate_noise": candidate_noise})

    return ProjectionLibrary.from_array(np.stack(weights, axis=0), labels=labels, metadata=metadata)


def warmup_selector(
    pipeline: HybridBCIPipeline,
    features: FloatArray,
    labels: IntArray,
    n_trials: int,
) -> int:
    """Update the selector with labeled calibration windows."""

    count = min(max(0, n_trials), len(labels))
    for x, y in zip(features[:count], labels[:count]):
        output = pipeline.predict_window(x)
        pipeline.update_from_label(output, int(y))
    return count


def run_replay(
    pipeline: HybridBCIPipeline,
    features: FloatArray,
    labels: IntArray,
    start_index: int,
    update_online: bool = True,
) -> dict[str, object]:
    """Run labeled replay and return generic metrics."""

    n_classes = len(pipeline.decision_head.config.class_names)
    confusion = np.zeros((n_classes, n_classes + 1), dtype=int)
    correct = 0
    rejected = 0
    predictions: list[str] = []
    outputs = []

    for x, y in zip(features[start_index:], labels[start_index:]):
        output = pipeline.predict_window(x)
        if update_online:
            pipeline.update_from_label(output, int(y))
        outputs.append(output)
        if output.rejected:
            rejected += 1
            predictions.append("reject")
            confusion[int(y), n_classes] += 1
        else:
            predictions.append(output.predicted_class or "reject")
            correct += int(output.predicted_index == int(y))
            confusion[int(y), int(output.predicted_index)] += 1

    total = len(labels[start_index:])
    accepted = total - rejected
    recalls = []
    for class_index in range(n_classes):
        class_total = confusion[class_index].sum()
        recalls.append(confusion[class_index, class_index] / class_total if class_total else 0.0)
    return {
        "total": total,
        "correct": correct,
        "rejected": rejected,
        "accepted_accuracy": correct / accepted if accepted else 0.0,
        "command_accuracy": correct / total if total else 0.0,
        "balanced_command_accuracy": float(np.mean(recalls)),
        "reject_rate": rejected / total if total else 0.0,
        "predictions": predictions,
        "confusion": confusion,
        "outputs": outputs,
    }


def _bootstrap_projection(
    x_train: FloatArray,
    y_train: IntArray,
    rng: np.random.Generator,
    n_classes: int,
) -> FloatArray:
    indices: list[int] = []
    for class_index in range(n_classes):
        class_indices = np.flatnonzero(y_train == class_index)
        sampled = rng.choice(class_indices, size=len(class_indices), replace=True)
        indices.extend(sampled.tolist())
    shuffled = rng.permutation(np.asarray(indices, dtype=int))
    return least_squares_projection(x_train[shuffled], y_train[shuffled], n_classes=n_classes)
