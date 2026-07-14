"""BNCI2014_004 evaluation for the three FBCSP design lines."""

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
    add_feature_standardization_event,
    add_fbcsp_events,
    add_fbcsp_transform_events,
    add_lda_fit_events,
    add_linear_scores_event,
    add_mlp_forward_event,
    add_mlp_training_event,
    compact_summary_fields,
    events_from_dicts,
    summarize_lines,
)
from ..datasets.bnci2014_004 import (
    DEFAULT_SUBJECTS,
    calibration_eval_split,
    load_subject_history_and_target,
)
from ..evaluation import softmax
from ..experience import ExperienceEntry, build_bootstrap_experience_library, scan_experience_heads
from ..fbcsp import FilterBankCSP
from ..linear_models import FeatureStandardizer, LinearHead, ShrinkageLDA, select_fisher_features
from ..progress import ProgressLogger
from ..small_networks import SmallMLPConfig, train_small_mlp
from .common import evaluate_scores, save_json, save_npz


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

LINES = (
    "FBCSP + shrinkage LDA",
    "FBCSP + small MLP embedding",
    "FBCSP + MLP + library + photonic scan",
)


@dataclass(frozen=True)
class BNCI004PersonalizationConfig:
    """BNCI2014_004 protocol used to compare the three design lines.

    The historical name is kept so existing scripts still work. The workflow now
    performs a direct three-line comparison instead of reporting a fourth
    "personalization" line.
    """

    data_dir: Path | str = Path("Dataset/BNCI2014_004")
    metrics_dir: Path | str = Path("artifacts/metrics/bnci2014_004_personalization")
    subjects: tuple[int, ...] = DEFAULT_SUBJECTS
    calibration_trials_per_class: int = 8
    bands: tuple[tuple[float, float], ...] = (
        (8.0, 12.0),
        (12.0, 16.0),
        (16.0, 20.0),
        (20.0, 24.0),
        (24.0, 28.0),
        (28.0, 32.0),
    )
    filter_order: int = 3
    csp_components: int = 1
    csp_shrinkage: float = 0.15
    selected_features: int = 12
    reject_target_rate: float = 0.02
    margin_threshold: float = 0.0
    mlp_epochs: int = 120
    mlp_hidden_dim: int = 32
    mlp_embedding_dim: int = 16
    mlp_dropout: float = 0.20
    experience_entries: int = 32
    experience_sample_fraction: float = 0.70
    top_k: int = 8
    experience_anchor_prior: float = 1.0
    tile_shape: tuple[int, int] = (2, 8)
    seed: int = 13

    @property
    def metrics_path(self) -> Path:
        return Path(self.metrics_dir)


def run_bnci2014_004_personalization(
    config: BNCI004PersonalizationConfig | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Run the BNCI2014_004 three-line design comparison."""

    cfg = config or BNCI004PersonalizationConfig()
    total_steps = len(cfg.subjects) + 2
    progress = ProgressLogger(
        "bnci2014_004_three_line_comparison",
        cfg.metrics_path / "run_progress.json" if save else None,
    )
    subject_results = []
    for index, subject in enumerate(cfg.subjects, start=1):
        with progress.step(f"run subject {subject}", index=index, total=total_steps):
            subject_results.append(_run_subject(subject, cfg))
    with progress.step("aggregate subject and compute summaries", index=len(cfg.subjects) + 1, total=total_steps):
        subject_rows = [row for result in subject_results for row in result["rows"]]
        summary_rows: list[dict[str, Any]] = []
        compute_lines = []
        for line in LINES:
            line_rows = [row for row in subject_rows if row["line"] == line]
            ledger = LinearComputeLedger()
            for result in subject_results:
                ledger.extend(events_from_dicts(result["compute_events"][line]))
            compute_summary = ledger.summary()
            summary_rows.append(_aggregate_line(line_rows, compute_summary))
            compute_lines.append(
                {
                    "line": line,
                    "summary": compute_summary,
                    "events": ledger.to_events(),
                }
            )
    payload = {
        "config": cfg,
        "summary": summary_rows,
        "subject_rows": subject_rows,
        "compute_accounting": summarize_lines(compute_lines),
    }
    if save:
        with progress.step("save BNCI metrics", index=total_steps, total=total_steps):
            _save_results(payload, cfg.metrics_path)
    progress.write()
    return payload


def _run_subject(subject: int, cfg: BNCI004PersonalizationConfig) -> dict[str, Any]:
    history, target = load_subject_history_and_target(cfg.data_dir, subject)
    calibration_idx, eval_idx = calibration_eval_split(
        target.labels,
        trials_per_class=cfg.calibration_trials_per_class,
    )
    fbcsp = FilterBankCSP(
        bands=cfg.bands,
        n_components=cfg.csp_components,
        filter_order=cfg.filter_order,
        covariance_shrinkage=cfg.csp_shrinkage,
    )
    history_fbcsp = fbcsp.fit_transform(history.trials, history.labels, history.fs, history.class_names)
    target_fbcsp = fbcsp.transform(target.trials)
    selected = select_fisher_features(
        history_fbcsp.vector,
        history.labels,
        n_classes=len(history.class_names),
        n_features=cfg.selected_features,
    )
    standardizer = FeatureStandardizer()
    x_history = standardizer.fit_transform(history_fbcsp.vector[:, selected])
    x_target = standardizer.transform(target_fbcsp.vector[:, selected])
    y_eval = target.labels[eval_idx]

    rows: list[dict[str, Any]] = []
    compute_events: dict[str, list[dict[str, Any]]] = {}

    reference = _reference_line(
        subject=subject,
        cfg=cfg,
        x_history=x_history,
        y_history=history.labels,
        x_eval=x_target[eval_idx],
        y_eval=y_eval,
        class_names=history.class_names,
        trial_shape=history.trials.shape,
        eval_windows=len(eval_idx),
    )
    rows.append(reference["row"])
    compute_events[reference["row"]["line"]] = reference["compute_events"]

    small = _small_network_line(
        subject=subject,
        cfg=cfg,
        x_history=x_history,
        y_history=history.labels,
        x_target=x_target,
        eval_idx=eval_idx,
        y_eval=y_eval,
        class_names=history.class_names,
        trial_shape=history.trials.shape,
        eval_windows=len(eval_idx),
    )
    rows.append(small["row"])
    compute_events[small["row"]["line"]] = small["compute_events"]

    mainline = _mainline(
        subject=subject,
        cfg=cfg,
        x_history=x_history,
        y_history=history.labels,
        x_target=x_target,
        calibration_idx=calibration_idx,
        y_cal=target.labels[calibration_idx],
        eval_idx=eval_idx,
        y_eval=y_eval,
        class_names=history.class_names,
        trial_shape=history.trials.shape,
        small=small,
    )
    rows.append(mainline["row"])
    compute_events[mainline["row"]["line"]] = mainline["compute_events"]
    return {"subject": subject, "rows": rows, "compute_events": compute_events}


def _reference_line(
    *,
    subject: int,
    cfg: BNCI004PersonalizationConfig,
    x_history: FloatArray,
    y_history: IntArray,
    x_eval: FloatArray,
    y_eval: IntArray,
    class_names: tuple[str, ...],
    trial_shape: tuple[int, int, int],
    eval_windows: int,
) -> dict[str, Any]:
    line = LINES[0]
    lda = ShrinkageLDA(shrinkage=0.20).fit(x_history, y_history, class_names)
    train_scores = lda.scores(x_history)
    eval_scores = lda.scores(x_eval)
    metrics = evaluate_scores(
        eval_scores,
        y_eval,
        class_names=class_names,
        reject_threshold=_reject_threshold(train_scores, cfg.reject_target_rate),
        margin_threshold=cfg.margin_threshold,
    )
    ledger = _base_fbcsp_ledger(subject, "reference", cfg, trial_shape, len(class_names), eval_windows)
    add_lda_fit_events(
        ledger,
        prefix=f"subject {subject} reference LDA",
        n_samples=x_history.shape[0],
        n_features=x_history.shape[1],
        n_classes=len(class_names),
    )
    add_linear_scores_event(
        ledger,
        name=f"subject {subject} reference train scores",
        n_samples=x_history.shape[0],
        n_features=x_history.shape[1],
        n_outputs=len(class_names),
        stage="calibration",
    )
    add_linear_scores_event(
        ledger,
        name=f"subject {subject} reference eval scores",
        n_samples=x_eval.shape[0],
        n_features=x_eval.shape[1],
        n_outputs=len(class_names),
        stage="inference",
    )
    return {"row": _subject_row(subject, line, metrics, eval_windows), "compute_events": ledger.to_events()}


def _small_network_line(
    *,
    subject: int,
    cfg: BNCI004PersonalizationConfig,
    x_history: FloatArray,
    y_history: IntArray,
    x_target: FloatArray,
    eval_idx: IntArray,
    y_eval: IntArray,
    class_names: tuple[str, ...],
    trial_shape: tuple[int, int, int],
    eval_windows: int,
) -> dict[str, Any]:
    line = LINES[1]
    n_classes = len(class_names)
    mlp = train_small_mlp(
        train_features=x_history,
        train_labels=y_history,
        replay_features=x_target,
        n_classes=n_classes,
        config=SmallMLPConfig(
            hidden_dim=cfg.mlp_hidden_dim,
            embedding_dim=cfg.mlp_embedding_dim,
            dropout=cfg.mlp_dropout,
            epochs=cfg.mlp_epochs,
            seed=cfg.seed + subject,
        ),
    )
    metrics = evaluate_scores(
        mlp.replay_scores[eval_idx],
        y_eval,
        class_names=class_names,
        reject_threshold=_reject_threshold(mlp.train_scores, cfg.reject_target_rate),
        margin_threshold=cfg.margin_threshold,
    )
    ledger = _base_fbcsp_ledger(subject, "small-network", cfg, trial_shape, n_classes, eval_windows)
    _add_mlp_compute(
        ledger,
        prefix=f"subject {subject} small-network",
        cfg=cfg,
        train_windows=x_history.shape[0],
        inference_windows=eval_windows,
        input_dim=x_history.shape[1],
        n_classes=n_classes,
    )
    return {
        "row": _subject_row(subject, line, metrics, eval_windows),
        "compute_events": ledger.to_events(),
        "mlp": mlp,
        "classifier_weights": mlp.model.classifier.weight.detach().cpu().numpy().astype(np.float64),
        "classifier_bias": mlp.model.classifier.bias.detach().cpu().numpy().astype(np.float64),
    }


def _mainline(
    *,
    subject: int,
    cfg: BNCI004PersonalizationConfig,
    x_history: FloatArray,
    y_history: IntArray,
    x_target: FloatArray,
    calibration_idx: IntArray,
    y_cal: IntArray,
    eval_idx: IntArray,
    y_eval: IntArray,
    class_names: tuple[str, ...],
    trial_shape: tuple[int, int, int],
    small: dict[str, Any],
) -> dict[str, Any]:
    line = LINES[2]
    n_classes = len(class_names)
    mlp = small["mlp"]
    library = _build_embedding_library(
        mlp.train_embeddings,
        y_history,
        class_names,
        classifier_weights=small["classifier_weights"],
        classifier_bias=small["classifier_bias"],
        cfg=cfg,
        subject=subject,
    )
    selected, weights = _select_candidates(
        library,
        calibration_embeddings=mlp.replay_embeddings[calibration_idx],
        calibration_labels=y_cal,
        top_k=cfg.top_k,
        anchor_prior_strength=cfg.experience_anchor_prior,
    )
    backend = TiledMVMBackend(tile_shape=cfg.tile_shape)
    train_scan = scan_experience_heads(selected, weights, mlp.train_embeddings, backend=backend)
    eval_scan = scan_experience_heads(selected, weights, mlp.replay_embeddings[eval_idx], backend=backend)
    metrics = evaluate_scores(
        eval_scan.fused_scores,
        y_eval,
        class_names=class_names,
        reject_threshold=_reject_threshold(train_scan.fused_scores, cfg.reject_target_rate),
        margin_threshold=cfg.margin_threshold,
    )
    ledger = _base_fbcsp_ledger(subject, "mainline", cfg, trial_shape, n_classes, len(eval_idx))
    add_fbcsp_transform_events(
        ledger,
        prefix=f"subject {subject} mainline calibration",
        n_trials=len(calibration_idx),
        n_bands=len(cfg.bands),
        n_classes=n_classes,
        n_filters=2 * cfg.csp_components,
        n_channels=trial_shape[1],
        n_samples=trial_shape[2],
        filter_order=cfg.filter_order,
        stage="calibration",
    )
    add_feature_standardization_event(
        ledger,
        name=f"subject {subject} mainline calibration standardization affine",
        n_samples=len(calibration_idx),
        n_features=cfg.selected_features,
        stage="calibration",
    )
    _add_mlp_compute(
        ledger,
        prefix=f"subject {subject} mainline",
        cfg=cfg,
        train_windows=x_history.shape[0],
        inference_windows=len(eval_idx),
        input_dim=x_history.shape[1],
        n_classes=n_classes,
    )
    add_mlp_forward_event(
        ledger,
        name=f"subject {subject} mainline calibration MLP forward",
        n_samples=len(calibration_idx),
        input_dim=x_history.shape[1],
        hidden_dim=cfg.mlp_hidden_dim,
        embedding_dim=cfg.mlp_embedding_dim,
        n_classes=n_classes,
        stage="calibration",
    )
    _add_library_compute(
        ledger,
        subject=subject,
        train_windows=mlp.train_embeddings.shape[0],
        embedding_dim=mlp.train_embeddings.shape[1],
        n_classes=n_classes,
        bootstrap_entries=tuple(entry for entry in library if entry.source == "bootstrap"),
    )
    add_centroid_retrieval_event(
        ledger,
        name=f"subject {subject} calibration-to-library centroid retrieval",
        n_queries=1,
        n_centroids=len(library),
        n_features=mlp.train_embeddings.shape[1],
        stage="calibration",
    )
    add_linear_scores_event(
        ledger,
        name=f"subject {subject} selected heads calibration scores",
        n_samples=len(calibration_idx) * len(selected),
        n_features=mlp.train_embeddings.shape[1],
        n_outputs=n_classes,
        stage="calibration",
    )
    add_candidate_scan_events(
        ledger,
        prefix=f"subject {subject} train threshold scan",
        n_windows=mlp.train_embeddings.shape[0],
        n_candidates=len(selected),
        n_features=mlp.train_embeddings.shape[1],
        n_classes=n_classes,
        stage="calibration",
    )
    add_candidate_scan_events(
        ledger,
        prefix=f"subject {subject} online eval scan",
        n_windows=len(eval_idx),
        n_candidates=len(selected),
        n_features=mlp.train_embeddings.shape[1],
        n_classes=n_classes,
        stage="inference",
    )
    row = _subject_row(subject, line, metrics, len(eval_idx))
    row.update(
        {
            "calibration_trials_total": int(len(calibration_idx)),
            "top_k": int(len(selected)),
            "tile_evaluations_per_window": int(eval_scan.tile_count_per_window),
        }
    )
    return {"row": row, "compute_events": ledger.to_events()}


def _base_fbcsp_ledger(
    subject: int,
    label: str,
    cfg: BNCI004PersonalizationConfig,
    trial_shape: tuple[int, int, int],
    n_classes: int,
    eval_windows: int,
) -> LinearComputeLedger:
    ledger = LinearComputeLedger()
    add_fbcsp_events(
        ledger,
        prefix=f"subject {subject} {label} FBCSP",
        n_train=trial_shape[0],
        n_replay=eval_windows,
        n_bands=len(cfg.bands),
        n_classes=n_classes,
        n_filters=2 * cfg.csp_components,
        n_channels=trial_shape[1],
        n_samples=trial_shape[2],
        filter_order=cfg.filter_order,
    )
    add_feature_standardization_event(
        ledger,
        name=f"subject {subject} {label} selected FBCSP standardization affine",
        n_samples=trial_shape[0] + eval_windows,
        n_features=cfg.selected_features,
        stage="preprocessing",
    )
    return ledger


def _add_mlp_compute(
    ledger: LinearComputeLedger,
    *,
    prefix: str,
    cfg: BNCI004PersonalizationConfig,
    train_windows: int,
    inference_windows: int,
    input_dim: int,
    n_classes: int,
) -> None:
    add_mlp_training_event(
        ledger,
        prefix=prefix,
        n_samples=train_windows,
        input_dim=input_dim,
        hidden_dim=cfg.mlp_hidden_dim,
        embedding_dim=cfg.mlp_embedding_dim,
        n_classes=n_classes,
        epochs=cfg.mlp_epochs,
    )
    add_mlp_forward_event(
        ledger,
        name=f"{prefix} train MLP forward",
        n_samples=train_windows,
        input_dim=input_dim,
        hidden_dim=cfg.mlp_hidden_dim,
        embedding_dim=cfg.mlp_embedding_dim,
        n_classes=n_classes,
        stage="calibration",
    )
    add_mlp_forward_event(
        ledger,
        name=f"{prefix} eval MLP forward",
        n_samples=inference_windows,
        input_dim=input_dim,
        hidden_dim=cfg.mlp_hidden_dim,
        embedding_dim=cfg.mlp_embedding_dim,
        n_classes=n_classes,
        stage="inference",
    )


def _build_embedding_library(
    embeddings: FloatArray,
    labels: IntArray,
    class_names: tuple[str, ...],
    *,
    classifier_weights: FloatArray,
    classifier_bias: FloatArray,
    cfg: BNCI004PersonalizationConfig,
    subject: int,
) -> tuple[ExperienceEntry, ...]:
    mlp_head = LinearHead(classifier_weights, classifier_bias, class_names)
    mlp_scores = mlp_head.scores(embeddings)
    mlp_entry = ExperienceEntry(
        entry_id="anchor_mlp_classifier",
        centroid=embeddings.mean(axis=0),
        head=mlp_head,
        source="small_network_classifier",
        train_indices=np.arange(len(labels), dtype=int),
        train_accuracy=float((mlp_scores.argmax(axis=1) == labels).mean()),
    )
    lda = ShrinkageLDA(shrinkage=0.20).fit(embeddings, labels, class_names=class_names)
    lda_scores = lda.scores(embeddings)
    lda_entry = ExperienceEntry(
        entry_id="anchor_embedding_lda",
        centroid=embeddings.mean(axis=0),
        head=lda.head,
        source="global_embedding_lda",
        train_indices=np.arange(len(labels), dtype=int),
        train_accuracy=float((lda_scores.argmax(axis=1) == labels).mean()),
    )
    bootstrap = build_bootstrap_experience_library(
        embeddings,
        labels,
        class_names=class_names,
        n_entries=cfg.experience_entries,
        sample_fraction=cfg.experience_sample_fraction,
        shrinkage=0.20,
        seed=cfg.seed + subject,
    )
    return (mlp_entry, lda_entry, *bootstrap)


def _select_candidates(
    entries: tuple[ExperienceEntry, ...],
    *,
    calibration_embeddings: FloatArray,
    calibration_labels: IntArray,
    top_k: int,
    anchor_prior_strength: float,
) -> tuple[tuple[ExperienceEntry, ...], FloatArray]:
    query = calibration_embeddings.mean(axis=0)
    centroids = np.stack([entry.centroid for entry in entries], axis=0)
    distances = prototype_distances(
        query[None, :],
        centroids,
        name="bnci_retrieval_centroid_distance",
    )[0]
    anchor_indices = [index for index, entry in enumerate(entries) if entry.entry_id.startswith("anchor_")]
    rest_indices = [index for index in range(len(entries)) if index not in set(anchor_indices)]
    rest_order = np.asarray(rest_indices, dtype=int)[np.argsort(distances[rest_indices])]
    order = np.asarray([*anchor_indices, *rest_order[: max(0, top_k - len(anchor_indices))]], dtype=int)
    selected = tuple(entries[int(index)] for index in order)
    selected_distances = distances[order]
    cal_acc = []
    cal_conf = []
    for entry in selected:
        scores = entry.head.scores(calibration_embeddings)
        probs = softmax(scores)
        cal_acc.append(float((probs.argmax(axis=1) == calibration_labels).mean()))
        cal_conf.append(float(probs[np.arange(len(calibration_labels)), calibration_labels].mean()))
    distance_term = -(selected_distances - selected_distances.min()) / (selected_distances.std() + 1e-6)
    train_acc = np.asarray([entry.train_accuracy for entry in selected], dtype=np.float64)
    anchor_prior = np.asarray([1.0 if entry.entry_id.startswith("anchor_") else 0.0 for entry in selected])
    logits = (
        0.35 * distance_term
        + 2.0 * np.asarray(cal_acc)
        + 0.5 * np.asarray(cal_conf)
        + 0.5 * train_acc
        + float(anchor_prior_strength) * anchor_prior
    )
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return selected, weights.astype(np.float64)


def _add_library_compute(
    ledger: LinearComputeLedger,
    *,
    subject: int,
    train_windows: int,
    embedding_dim: int,
    n_classes: int,
    bootstrap_entries: tuple[ExperienceEntry, ...],
) -> None:
    add_linear_scores_event(
        ledger,
        name=f"subject {subject} anchor MLP train scores",
        n_samples=train_windows,
        n_features=embedding_dim,
        n_outputs=n_classes,
        stage="fit",
    )
    add_lda_fit_events(
        ledger,
        prefix=f"subject {subject} anchor embedding LDA",
        n_samples=train_windows,
        n_features=embedding_dim,
        n_classes=n_classes,
    )
    add_linear_scores_event(
        ledger,
        name=f"subject {subject} anchor embedding LDA train scores",
        n_samples=train_windows,
        n_features=embedding_dim,
        n_outputs=n_classes,
        stage="fit",
    )
    total_bootstrap_samples = int(sum(len(entry.train_indices) for entry in bootstrap_entries))
    ledger.add(
        f"subject {subject} bootstrap LDA pooled covariance centered.T @ centered",
        total_bootstrap_samples * embedding_dim * embedding_dim,
        photonic=True,
        stage="fit",
        category="lda_fit_covariance",
        implementation="simulated_photonic_matmul",
        details={"entries": len(bootstrap_entries), "total_bootstrap_samples": total_bootstrap_samples},
    )
    ledger.add(
        f"subject {subject} bootstrap LDA weights/bias means @ inv_cov",
        len(bootstrap_entries) * 2 * n_classes * embedding_dim * embedding_dim,
        photonic=True,
        stage="fit",
        category="lda_fit_parameters",
        implementation="simulated_photonic_matmul",
        details={"entries": len(bootstrap_entries), "features": embedding_dim, "classes": n_classes},
    )
    ledger.add(
        f"subject {subject} bootstrap train scores",
        total_bootstrap_samples * (embedding_dim + 1) * n_classes,
        photonic=True,
        stage="fit",
        category="linear_head_scores",
        implementation="simulated_photonic_augmented_matmul",
        details={
            "entries": len(bootstrap_entries),
            "total_bootstrap_samples": total_bootstrap_samples,
            "features": embedding_dim,
            "augmented_features": embedding_dim + 1,
            "classes": n_classes,
            "note": "Bias is counted as an augmented constant-one input channel.",
        },
    )


def _reject_threshold(scores: FloatArray, target_rate: float) -> float:
    if target_rate <= 0.0:
        return 0.0
    confidence = softmax(np.asarray(scores, dtype=np.float64)).max(axis=1)
    return float(np.quantile(confidence, min(max(float(target_rate), 0.0), 0.95)))


def _subject_row(
    subject: int,
    line: str,
    metrics: dict[str, Any],
    eval_windows: int,
) -> dict[str, Any]:
    return {
        "subject": int(subject),
        "line": line,
        "eval_windows": int(eval_windows),
        "command_accuracy": float(metrics["command_accuracy"]),
        "balanced_command_accuracy": float(metrics["balanced_command_accuracy"]),
        "accepted_accuracy": float(metrics["accepted_accuracy"]),
        "reject_rate": float(metrics["reject_rate"]),
        "confusion": metrics["confusion"],
    }


def _aggregate_line(rows: list[dict[str, Any]], compute_summary: dict[str, Any]) -> dict[str, Any]:
    command = np.asarray([row["command_accuracy"] for row in rows], dtype=np.float64)
    balanced = np.asarray([row["balanced_command_accuracy"] for row in rows], dtype=np.float64)
    accepted = np.asarray([row["accepted_accuracy"] for row in rows], dtype=np.float64)
    reject = np.asarray([row["reject_rate"] for row in rows], dtype=np.float64)
    row = {
        "line": rows[0]["line"],
        "subjects": int(len(rows)),
        "total": int(sum(row["eval_windows"] for row in rows)),
        "command_accuracy": float(command.mean()),
        "command_accuracy_std": float(command.std(ddof=0)),
        "balanced_command_accuracy": float(balanced.mean()),
        "balanced_command_accuracy_std": float(balanced.std(ddof=0)),
        "accepted_accuracy": float(accepted.mean()),
        "accepted_accuracy_std": float(accepted.std(ddof=0)),
        "reject_rate": float(reject.mean()),
        "reject_rate_std": float(reject.std(ddof=0)),
        "tile_evaluations_per_window": int(max(row.get("tile_evaluations_per_window", 0) for row in rows)),
    }
    row.update(compact_summary_fields(compute_summary))
    return row


def _save_results(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "summary.json",
        {
            "dataset": "BNCI2014_004",
            "protocol": (
                "sessions 1-2 train/history, session 3 target; first calibration "
                "windows per class are used only for mainline library retrieval; "
                "all lines share the same held-out evaluation windows"
            ),
            "config": payload["config"],
            "rows": payload["summary"],
            "subject_rows_file": "subject_rows.json",
            "compute_accounting_file": "compute_accounting.json",
            "progress_file": "run_progress.json",
        },
    )
    save_json(output_dir / "subject_rows.json", {"rows": payload["subject_rows"]})
    save_json(output_dir / "compute_accounting.json", payload["compute_accounting"])
    save_npz(
        output_dir / "arrays.npz",
        subjects=np.asarray([row["subject"] for row in payload["subject_rows"]], dtype=int),
        line=np.asarray([row["line"] for row in payload["subject_rows"]]),
        command_accuracy=np.asarray([row["command_accuracy"] for row in payload["subject_rows"]], dtype=np.float64),
        balanced_command_accuracy=np.asarray(
            [row["balanced_command_accuracy"] for row in payload["subject_rows"]],
            dtype=np.float64,
        ),
        reject_rate=np.asarray([row["reject_rate"] for row in payload["subject_rows"]], dtype=np.float64),
    )
