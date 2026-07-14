"""Experience-library retrieval with tiled photonic candidate-head scan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backends import TiledMVMBackend, prototype_distances
from ..compute_accounting import (
    LinearComputeLedger,
    add_candidate_scan_events,
    add_centroid_retrieval_event,
    add_lda_fit_events,
    add_linear_scores_event,
    compact_summary_fields,
    events_from_dicts,
)
from ..experience import (
    ExperienceEntry,
    build_bootstrap_experience_library,
    scan_experience_heads,
)
from ..evaluation import softmax
from ..linear_models import LinearHead, ShrinkageLDA
from ..progress import ConsoleProgressBar
from .common import (
    FBCSPDesignConfig,
    FBCSPPreparedData,
    calibrated_reject_threshold,
    evaluate_scores,
    prepare_fbcsp_data,
    save_json,
    save_npz,
    summary_from_metrics,
)
from .small_network_line import SmallNetworkLineResult, run_small_network_line


FloatArray = NDArray[np.float64]


@dataclass
class ExperiencePhotonicLineResult:
    prepared: FBCSPPreparedData
    small_network: SmallNetworkLineResult
    library: tuple[ExperienceEntry, ...]
    selected_entries: tuple[ExperienceEntry, ...]
    retrieval_weights: FloatArray
    retrieval_distances: FloatArray
    selected_calibration_accuracy: FloatArray
    selected_calibration_confidence: FloatArray
    train_fused_scores: FloatArray
    eval_fused_scores: FloatArray
    eval_candidate_scores: FloatArray
    eval_labels: NDArray[np.int_]
    metrics: dict[str, Any]
    reject_threshold: float
    tile_count_per_window: int
    compute_summary: dict[str, Any]
    compute_events: list[dict[str, Any]]
    summary: dict[str, Any]


def run_experience_photonic_line(
    config: FBCSPDesignConfig | None = None,
    prepared: FBCSPPreparedData | None = None,
    small_network: SmallNetworkLineResult | None = None,
    save: bool = True,
    show_progress: bool = False,
) -> ExperiencePhotonicLineResult:
    """Run the mainline system from the design document."""

    cfg = config or FBCSPDesignConfig()
    data = prepared or prepare_fbcsp_data(cfg)
    small = small_network or run_small_network_line(cfg, prepared=data, save=save)
    bootstrap_library = build_bootstrap_experience_library(
        embeddings=small.train_embeddings,
        labels=data.train_labels,
        class_names=data.dataset.class_names,
        n_entries=cfg.experience_entries,
        sample_fraction=cfg.experience_sample_fraction,
        seed=cfg.seed,
    )
    library = _build_anchor_entries(small, data) + bootstrap_library
    calibration_embeddings = small.replay_embeddings[data.split.calibration_replay]
    selected, retrieval_weights, retrieval_distances, calibration_accuracy, calibration_confidence = _select_candidates(
        library,
        calibration_embeddings=calibration_embeddings,
        calibration_labels=data.replay_labels[data.split.calibration_replay],
        k=cfg.experience_top_k,
        anchor_prior_strength=cfg.experience_anchor_prior,
    )
    backend = TiledMVMBackend(tile_shape=cfg.tile_shape)
    train_scan = scan_experience_heads(
        selected,
        retrieval_weights,
        embeddings=small.train_embeddings,
        backend=backend,
    )
    reject_threshold = calibrated_reject_threshold(train_scan.fused_scores, cfg)
    eval_embeddings = small.replay_embeddings[data.split.evaluation_replay]
    eval_labels = data.replay_labels[data.split.evaluation_replay]
    live_tracker = _LiveAccuracyProgress(
        labels=eval_labels,
        reject_threshold=reject_threshold,
        margin_threshold=cfg.margin_threshold,
        enabled=show_progress,
    )
    eval_scan = scan_experience_heads(
        selected,
        retrieval_weights,
        embeddings=eval_embeddings,
        backend=backend,
        progress_callback=live_tracker.update if show_progress else None,
    )
    live_tracker.close()
    metrics = evaluate_scores(
        eval_scan.fused_scores,
        eval_labels,
        class_names=data.dataset.class_names,
        reject_threshold=reject_threshold,
        margin_threshold=cfg.margin_threshold,
    )
    metrics.update(_cumulative_decision_traces(metrics["correct_trace"], metrics["rejected_mask"]))
    ledger = _account_experience_compute(
        small=small,
        data=data,
        bootstrap_library=bootstrap_library,
        selected=selected,
        train_windows=small.train_embeddings.shape[0],
        calibration_windows=calibration_embeddings.shape[0],
        eval_windows=eval_embeddings.shape[0],
    )
    compute_summary = ledger.summary()
    summary = summary_from_metrics(
        "FBCSP + MLP embedding + library + photonic scan",
        metrics,
        extra={
            "raw_fbcsp_dim": int(data.train_features_raw.shape[1]),
            "selected_features": int(len(data.selected_indices)),
            "embedding_dim": int(small.train_embeddings.shape[1]),
            "experience_entries": int(len(library)),
            "top_k": int(len(selected)),
            "anchor_prior": float(cfg.experience_anchor_prior),
            "calibration_trials": int(len(data.split.calibration_replay)),
            "tile_shape": cfg.tile_shape,
            "tile_evaluations_per_window": int(eval_scan.tile_count_per_window),
            "reject_threshold": reject_threshold,
            **compact_summary_fields(compute_summary),
        },
    )
    result = ExperiencePhotonicLineResult(
        prepared=data,
        small_network=small,
        library=library,
        selected_entries=selected,
        retrieval_weights=retrieval_weights,
        retrieval_distances=retrieval_distances,
        selected_calibration_accuracy=calibration_accuracy,
        selected_calibration_confidence=calibration_confidence,
        train_fused_scores=train_scan.fused_scores,
        eval_fused_scores=eval_scan.fused_scores,
        eval_candidate_scores=eval_scan.candidate_scores,
        eval_labels=eval_labels,
        metrics=metrics,
        reject_threshold=reject_threshold,
        tile_count_per_window=eval_scan.tile_count_per_window,
        compute_summary=compute_summary,
        compute_events=ledger.to_events(),
        summary=summary,
    )
    if save:
        save_experience_result(result, cfg.metrics_path / "experience_photonic")
    return result


def save_experience_result(result: ExperiencePhotonicLineResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = result.prepared
    save_json(
        output_dir / "summary.json",
        {
            "line": "FBCSP + small-network embedding + experience retrieval + photonic scan",
            "summary": result.summary,
            "calibration_protocol": {
                "calibration_trials_per_subject": data.config.calibration_trials_per_subject,
                "calibration_trials_total": int(len(data.split.calibration_replay)),
                "online_evaluation_trials": int(len(result.eval_labels)),
                "note": "Calibration windows query the experience library; online metrics exclude them.",
            },
            "compute_accounting": {
                "summary": result.compute_summary,
                "events_file": "../compute_accounting.json",
            },
            "selected_entries": [
                {
                    "entry_id": entry.entry_id,
                    "source": entry.source,
                    "train_accuracy": entry.train_accuracy,
                    "calibration_accuracy": float(result.selected_calibration_accuracy[index]),
                    "calibration_confidence": float(result.selected_calibration_confidence[index]),
                    "weight": float(result.retrieval_weights[index]),
                    "distance": float(result.retrieval_distances[index]),
                }
                for index, entry in enumerate(result.selected_entries)
            ],
        "library": {
                "entries": int(len(result.library)),
                "train_accuracy_mean": float(np.mean([entry.train_accuracy for entry in result.library])),
                "train_accuracy_std": float(np.std([entry.train_accuracy for entry in result.library])),
            },
            "metrics": {
                key: value
                for key, value in result.metrics.items()
                if key
                not in {
                    "probabilities",
                    "predicted",
                    "rejected_mask",
                    "confidence",
                    "margin",
                    "decision_labels",
                    "correct_trace",
                    "rolling_command_accuracy",
                    "rolling_reject_rate",
                    "cumulative_command_accuracy",
                    "cumulative_accepted_accuracy",
                    "cumulative_reject_rate",
                }
            },
        },
    )
    save_npz(
        output_dir / "arrays.npz",
        train_fused_scores=result.train_fused_scores,
        eval_fused_scores=result.eval_fused_scores,
        eval_candidate_scores=result.eval_candidate_scores,
        eval_labels=result.eval_labels,
        probabilities=result.metrics["probabilities"],
        predicted=result.metrics["predicted"],
        rejected=result.metrics["rejected_mask"],
        confidence=result.metrics["confidence"],
        margin=result.metrics["margin"],
        correct_trace=result.metrics["correct_trace"],
        rolling_command_accuracy=result.metrics["rolling_command_accuracy"],
        rolling_reject_rate=result.metrics["rolling_reject_rate"],
        cumulative_command_accuracy=result.metrics["cumulative_command_accuracy"],
        cumulative_accepted_accuracy=result.metrics["cumulative_accepted_accuracy"],
        cumulative_reject_rate=result.metrics["cumulative_reject_rate"],
        confusion=result.metrics["confusion"],
        retrieval_weights=result.retrieval_weights,
        retrieval_distances=result.retrieval_distances,
        selected_calibration_accuracy=result.selected_calibration_accuracy,
        selected_calibration_confidence=result.selected_calibration_confidence,
        selected_train_accuracy=np.asarray(
            [entry.train_accuracy for entry in result.selected_entries],
            dtype=np.float64,
        ),
        library_train_accuracy=np.asarray(
            [entry.train_accuracy for entry in result.library],
            dtype=np.float64,
        ),
        calibration_replay_indices=result.prepared.split.calibration_replay,
        evaluation_replay_indices=result.prepared.split.evaluation_replay,
    )


def _build_anchor_entries(
    small: SmallNetworkLineResult,
    data: FBCSPPreparedData,
) -> tuple[ExperienceEntry, ...]:
    """Add global embedding heads that every deployment can scan as anchors."""

    labels = data.train_labels
    indices = np.arange(len(labels), dtype=int)
    mlp_head = LinearHead(
        weights=small.classifier_weights,
        bias=small.classifier_bias,
        class_names=data.dataset.class_names,
    )
    mlp_scores = mlp_head.scores(small.train_embeddings)
    mlp_entry = ExperienceEntry(
        entry_id="anchor_mlp_classifier",
        centroid=small.train_embeddings.mean(axis=0),
        head=mlp_head,
        source="small_network_classifier",
        train_indices=indices,
        train_accuracy=float((mlp_scores.argmax(axis=1) == labels).mean()),
    )
    lda = ShrinkageLDA(shrinkage=0.15).fit(
        small.train_embeddings,
        labels,
        class_names=data.dataset.class_names,
    )
    lda_scores = lda.scores(small.train_embeddings)
    lda_entry = ExperienceEntry(
        entry_id="anchor_embedding_lda",
        centroid=small.train_embeddings.mean(axis=0),
        head=lda.head,
        source="global_embedding_lda",
        train_indices=indices,
        train_accuracy=float((lda_scores.argmax(axis=1) == labels).mean()),
    )
    return (mlp_entry, lda_entry)


def _account_experience_compute(
    *,
    small: SmallNetworkLineResult,
    data: FBCSPPreparedData,
    bootstrap_library: tuple[ExperienceEntry, ...],
    selected: tuple[ExperienceEntry, ...],
    train_windows: int,
    calibration_windows: int,
    eval_windows: int,
) -> LinearComputeLedger:
    ledger = LinearComputeLedger(events_from_dicts(small.compute_events))
    n_classes = len(data.dataset.class_names)
    embedding_dim = small.train_embeddings.shape[1]
    add_linear_scores_event(
        ledger,
        name="experience anchor MLP head train scores",
        n_samples=train_windows,
        n_features=embedding_dim,
        n_outputs=n_classes,
        stage="fit",
    )
    add_lda_fit_events(
        ledger,
        prefix="experience anchor embedding LDA",
        n_samples=train_windows,
        n_features=embedding_dim,
        n_classes=n_classes,
        stage="fit",
    )
    add_linear_scores_event(
        ledger,
        name="experience anchor embedding LDA train scores",
        n_samples=train_windows,
        n_features=embedding_dim,
        n_outputs=n_classes,
        stage="fit",
    )
    total_bootstrap_samples = int(
        sum(len(entry.train_indices) for entry in bootstrap_library)
    )
    ledger.add(
        "experience bootstrap LDA pooled covariance centered.T @ centered",
        total_bootstrap_samples * embedding_dim * embedding_dim,
        photonic=True,
        stage="fit",
        category="lda_fit_covariance",
        implementation="simulated_photonic_matmul",
        details={
            "entries": int(len(bootstrap_library)),
            "total_bootstrap_samples": total_bootstrap_samples,
            "features": int(embedding_dim),
            "classes": int(n_classes),
        },
    )
    ledger.add(
        "experience bootstrap LDA weights/bias means @ inv_cov",
        len(bootstrap_library) * 2 * n_classes * embedding_dim * embedding_dim,
        photonic=True,
        stage="fit",
        category="lda_fit_parameters",
        implementation="simulated_photonic_matmul",
        details={
            "entries": int(len(bootstrap_library)),
            "features": int(embedding_dim),
            "classes": int(n_classes),
            "note": "Aggregated over all bootstrap heads.",
        },
    )
    ledger.add(
        "experience bootstrap LDA train scores",
        total_bootstrap_samples * (embedding_dim + 1) * n_classes,
        photonic=True,
        stage="fit",
        category="linear_head_scores",
        implementation="simulated_photonic_augmented_matmul",
        details={
            "entries": int(len(bootstrap_library)),
            "total_bootstrap_samples": total_bootstrap_samples,
            "features": int(embedding_dim),
            "augmented_features": int(embedding_dim) + 1,
            "classes": int(n_classes),
            "note": "Bias is counted as an augmented constant-one input channel.",
        },
    )
    add_linear_scores_event(
        ledger,
        name="experience selected heads calibration scores",
        n_samples=calibration_windows * len(selected),
        n_features=embedding_dim,
        n_outputs=n_classes,
        stage="calibration",
    )
    add_centroid_retrieval_event(
        ledger,
        name="experience calibration-to-library centroid retrieval",
        n_queries=1,
        n_centroids=len(bootstrap_library) + 2,
        n_features=embedding_dim,
        stage="calibration",
    )
    add_candidate_scan_events(
        ledger,
        prefix="experience train threshold scan",
        n_windows=train_windows,
        n_candidates=len(selected),
        n_features=embedding_dim,
        n_classes=n_classes,
        stage="calibration",
    )
    add_candidate_scan_events(
        ledger,
        prefix="experience online evaluation scan",
        n_windows=eval_windows,
        n_candidates=len(selected),
        n_features=embedding_dim,
        n_classes=n_classes,
        stage="inference",
    )
    return ledger


def _select_candidates(
    entries: tuple[ExperienceEntry, ...],
    calibration_embeddings: FloatArray,
    calibration_labels: NDArray[np.int_] | None,
    k: int,
    anchor_prior_strength: float,
) -> tuple[tuple[ExperienceEntry, ...], FloatArray, FloatArray, FloatArray, FloatArray]:
    if not entries:
        raise ValueError("experience library is empty")
    query = np.asarray(calibration_embeddings, dtype=np.float64).mean(axis=0)
    centroids = np.stack([entry.centroid for entry in entries], axis=0)
    distances = prototype_distances(
        query[None, :],
        centroids,
        name="experience_mainline_retrieval_centroid_distance",
    )[0]
    anchor_indices = [
        index for index, entry in enumerate(entries) if entry.entry_id.startswith("anchor_")
    ]
    variant_indices = [
        index for index in range(len(entries)) if index not in set(anchor_indices)
    ]
    remaining = max(0, int(k) - len(anchor_indices))
    variant_order = np.asarray(variant_indices, dtype=int)[
        np.argsort(distances[variant_indices])[:remaining]
    ]
    order = np.asarray([*anchor_indices, *variant_order.tolist()], dtype=int)
    selected = tuple(entries[int(index)] for index in order)
    selected_distances = distances[order]
    selected_accuracy = np.asarray([entry.train_accuracy for entry in selected], dtype=np.float64)
    anchor_prior = np.asarray(
        [1.0 if entry.entry_id.startswith("anchor_") else 0.0 for entry in selected],
        dtype=np.float64,
    )
    calibration_accuracy, calibration_confidence = _calibration_stats(
        selected,
        calibration_embeddings,
        calibration_labels,
    )
    distance_scale = selected_distances.std() + 1e-6
    distance_term = -(selected_distances - selected_distances.min()) / distance_scale
    logits = (
        0.25 * distance_term
        + 0.75 * selected_accuracy
        + 3.0 * calibration_accuracy
        + 0.50 * calibration_confidence
        + float(anchor_prior_strength) * anchor_prior
    )
    weights = np.exp(logits - logits.max())
    weights = weights / weights.sum()
    return (
        selected,
        weights.astype(np.float64),
        selected_distances.astype(np.float64),
        calibration_accuracy.astype(np.float64),
        calibration_confidence.astype(np.float64),
    )


def _calibration_stats(
    entries: tuple[ExperienceEntry, ...],
    calibration_embeddings: FloatArray,
    calibration_labels: NDArray[np.int_] | None,
) -> tuple[FloatArray, FloatArray]:
    if calibration_labels is None or len(calibration_labels) == 0:
        ones = np.ones(len(entries), dtype=np.float64)
        return ones, ones
    labels = np.asarray(calibration_labels, dtype=int)
    accuracy = np.zeros(len(entries), dtype=np.float64)
    confidence = np.zeros(len(entries), dtype=np.float64)
    for index, entry in enumerate(entries):
        scores = entry.head.scores(calibration_embeddings)
        probabilities = softmax(scores)
        predicted = probabilities.argmax(axis=1)
        accuracy[index] = float((predicted == labels).mean())
        confidence[index] = float(probabilities[np.arange(len(labels)), labels].mean())
    return accuracy, confidence


class _LiveAccuracyProgress:
    def __init__(
        self,
        *,
        labels: NDArray[np.int_],
        reject_threshold: float,
        margin_threshold: float,
        enabled: bool,
    ) -> None:
        self.labels = np.asarray(labels, dtype=int)
        self.reject_threshold = float(reject_threshold)
        self.margin_threshold = float(margin_threshold)
        self.correct = 0
        self.rejected = 0
        self.progress = ConsoleProgressBar(
            "experience online scan",
            len(self.labels),
            enabled=enabled,
        )

    def update(self, current: int, total: int, probability: FloatArray) -> None:
        del total
        probability = np.asarray(probability, dtype=np.float64)
        order = np.argsort(probability)[::-1]
        predicted = int(order[0])
        second = int(order[1]) if len(order) > 1 else predicted
        confidence = float(probability[predicted])
        margin = float(probability[predicted] - probability[second])
        rejected = (
            confidence < self.reject_threshold
            or margin < self.margin_threshold
        )
        self.correct += int(not rejected and predicted == int(self.labels[current - 1]))
        self.rejected += int(rejected)
        accepted = max(1, current - self.rejected)
        suffix = (
            f"acc={self.correct / current:.3f} "
            f"accepted_acc={self.correct / accepted:.3f} "
            f"reject={self.rejected / current:.3f}"
        )
        self.progress.update(current, suffix=suffix)

    def close(self) -> None:
        self.progress.close()


def _cumulative_decision_traces(
    correct_trace: NDArray[np.float64],
    rejected_mask: NDArray[np.bool_],
) -> dict[str, FloatArray]:
    correct = np.asarray(correct_trace, dtype=np.float64)
    rejected = np.asarray(rejected_mask, dtype=bool)
    count = np.arange(1, len(correct) + 1, dtype=np.float64)
    rejected_count = np.cumsum(rejected.astype(np.float64))
    accepted_count = np.maximum(1.0, count - rejected_count)
    return {
        "cumulative_command_accuracy": np.cumsum(correct) / count,
        "cumulative_accepted_accuracy": np.cumsum(correct) / accepted_count,
        "cumulative_reject_rate": rejected_count / count,
    }
