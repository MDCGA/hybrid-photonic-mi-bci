"""Single-subject personalization experiment on BNCI2014_004."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..datasets.bnci2014_004 import (
    DEFAULT_SUBJECTS,
    calibration_eval_split,
    load_subject_history_and_target,
)
from ..evaluation import classification_metrics, rolling_mean, softmax
from ..experience import ExperienceEntry, build_bootstrap_experience_library, scan_experience_heads
from ..fbcsp import FilterBankCSP
from ..linear_models import FeatureStandardizer, LinearHead, ShrinkageLDA, select_fisher_features
from .common import save_json, save_npz


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class BNCI004PersonalizationConfig:
    data_dir: Path | str = Path("Dataset/BNCI2014_004")
    metrics_dir: Path | str = Path("artifacts/metrics/bnci2014_004_personalization")
    subjects: tuple[int, ...] = DEFAULT_SUBJECTS
    calibration_trials_per_class: tuple[int, ...] = (2, 4, 8, 12, 16)
    bands: tuple[tuple[float, float], ...] = (
        (8.0, 12.0),
        (12.0, 16.0),
        (16.0, 20.0),
        (20.0, 24.0),
        (24.0, 28.0),
        (28.0, 32.0),
    )
    csp_components: int = 1
    selected_features: int = 12
    experience_entries: int = 32
    top_k: int = 8
    seed: int = 13

    @property
    def metrics_path(self) -> Path:
        return Path(self.metrics_dir)


@dataclass
class SubjectPersonalizationResult:
    subject: int
    rows: list[dict[str, Any]]


def run_bnci2014_004_personalization(
    config: BNCI004PersonalizationConfig | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Run before/after personalization curves for all configured subjects."""

    cfg = config or BNCI004PersonalizationConfig()
    subject_results = [_run_subject(subject, cfg) for subject in cfg.subjects]
    rows = [row for result in subject_results for row in result.rows]
    summary = _summarize(rows, cfg)
    payload = {"config": cfg, "rows": rows, "summary": summary}
    if save:
        _save_results(payload, cfg.metrics_path)
    return payload


def _run_subject(subject: int, cfg: BNCI004PersonalizationConfig) -> SubjectPersonalizationResult:
    history, target = load_subject_history_and_target(cfg.data_dir, subject)
    fbcsp = FilterBankCSP(
        bands=cfg.bands,
        n_components=cfg.csp_components,
        covariance_shrinkage=0.15,
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

    anchor = ShrinkageLDA(shrinkage=0.20).fit(x_history, history.labels, history.class_names)
    library = _build_library_with_anchor(
        x_history,
        history.labels,
        history.class_names,
        anchor.head,
        n_entries=cfg.experience_entries,
        seed=cfg.seed + subject,
    )
    rows = []
    for k in cfg.calibration_trials_per_class:
        calibration_idx, eval_idx = calibration_eval_split(target.labels, trials_per_class=k)
        x_cal = x_target[calibration_idx]
        y_cal = target.labels[calibration_idx]
        x_eval = x_target[eval_idx]
        y_eval = target.labels[eval_idx]

        before_scores = anchor.scores(x_eval)
        before_metrics = _score_metrics(before_scores, y_eval, n_classes=2)

        calibration_head = ShrinkageLDA(shrinkage=0.40).fit(x_cal, y_cal, history.class_names)
        cal_only_scores = calibration_head.scores(x_eval)
        cal_only_metrics = _score_metrics(cal_only_scores, y_eval, n_classes=2)

        selected_entries, weights, distances, candidate_cal_acc = _select_with_calibration(
            library,
            x_cal,
            y_cal,
            k=cfg.top_k,
        )
        scan = scan_experience_heads(selected_entries, weights, x_eval)
        after_metrics = _score_metrics(scan.fused_scores, y_eval, n_classes=2)

        rows.append(
            {
                "subject": subject,
                "calibration_trials_per_class": int(k),
                "calibration_trials_total": int(len(calibration_idx)),
                "eval_trials": int(len(eval_idx)),
                "before_accuracy": before_metrics["command_accuracy"],
                "calibration_only_accuracy": cal_only_metrics["command_accuracy"],
                "experience_after_accuracy": after_metrics["command_accuracy"],
                "experience_gain_vs_before": after_metrics["command_accuracy"] - before_metrics["command_accuracy"],
                "experience_gain_vs_calibration_only": (
                    after_metrics["command_accuracy"] - cal_only_metrics["command_accuracy"]
                ),
                "before_confusion": before_metrics["confusion"],
                "calibration_only_confusion": cal_only_metrics["confusion"],
                "experience_confusion": after_metrics["confusion"],
                "candidate_weights": weights,
                "candidate_distances": distances,
                "candidate_calibration_accuracy": candidate_cal_acc,
                "candidate_ids": [entry.entry_id for entry in selected_entries],
                "rolling_before": _rolling_correct(before_scores, y_eval),
                "rolling_after": _rolling_correct(scan.fused_scores, y_eval),
            }
        )
    return SubjectPersonalizationResult(subject=subject, rows=rows)


def _build_library_with_anchor(
    features: FloatArray,
    labels: IntArray,
    class_names: tuple[str, ...],
    anchor_head: LinearHead,
    n_entries: int,
    seed: int,
) -> tuple[ExperienceEntry, ...]:
    library = build_bootstrap_experience_library(
        features,
        labels,
        class_names=class_names,
        n_entries=n_entries,
        sample_fraction=0.70,
        shrinkage=0.20,
        seed=seed,
    )
    anchor_scores = anchor_head.scores(features)
    anchor_entry = ExperienceEntry(
        entry_id="anchor_history_lda",
        centroid=features.mean(axis=0),
        head=anchor_head,
        source="target_history",
        train_indices=np.arange(len(labels), dtype=int),
        train_accuracy=float((anchor_scores.argmax(axis=1) == labels).mean()),
    )
    return (anchor_entry, *library)


def _select_with_calibration(
    entries: tuple[ExperienceEntry, ...],
    calibration_features: FloatArray,
    calibration_labels: IntArray,
    k: int,
) -> tuple[tuple[ExperienceEntry, ...], FloatArray, FloatArray, FloatArray]:
    query = calibration_features.mean(axis=0)
    centroids = np.stack([entry.centroid for entry in entries], axis=0)
    distances = np.linalg.norm(centroids - query[None, :], axis=1)
    anchor_indices = [index for index, entry in enumerate(entries) if entry.entry_id.startswith("anchor_")]
    rest_indices = [index for index in range(len(entries)) if index not in set(anchor_indices)]
    rest_order = np.asarray(rest_indices, dtype=int)[np.argsort(distances[rest_indices])]
    order = np.asarray([*anchor_indices, *rest_order[: max(0, k - len(anchor_indices))]], dtype=int)
    selected = tuple(entries[int(index)] for index in order)
    selected_distances = distances[order]
    cal_acc = []
    cal_conf = []
    for entry in selected:
        scores = entry.head.scores(calibration_features)
        probs = softmax(scores)
        cal_acc.append(float((probs.argmax(axis=1) == calibration_labels).mean()))
        cal_conf.append(float(probs[np.arange(len(calibration_labels)), calibration_labels].mean()))
    cal_acc_arr = np.asarray(cal_acc, dtype=np.float64)
    cal_conf_arr = np.asarray(cal_conf, dtype=np.float64)
    anchor_prior = np.asarray(
        [1.0 if entry.entry_id.startswith("anchor_") else 0.0 for entry in selected],
        dtype=np.float64,
    )
    distance_term = -(selected_distances - selected_distances.min()) / (selected_distances.std() + 1e-6)
    train_acc = np.asarray([entry.train_accuracy for entry in selected], dtype=np.float64)
    logits = 0.35 * distance_term + 2.0 * cal_acc_arr + 0.5 * cal_conf_arr + 0.5 * train_acc + anchor_prior
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return selected, weights, selected_distances, cal_acc_arr


def _score_metrics(scores: FloatArray, labels: IntArray, n_classes: int) -> dict[str, Any]:
    predicted = np.asarray(scores).argmax(axis=1)
    rejected = np.zeros(len(labels), dtype=bool)
    return classification_metrics(labels, predicted, rejected, n_classes=n_classes)


def _rolling_correct(scores: FloatArray, labels: IntArray) -> FloatArray:
    correct = (np.asarray(scores).argmax(axis=1) == labels).astype(np.float64)
    return rolling_mean(correct, window=min(20, len(correct)))


def _summarize(rows: list[dict[str, Any]], cfg: BNCI004PersonalizationConfig) -> list[dict[str, Any]]:
    summary = []
    for k in cfg.calibration_trials_per_class:
        subset = [row for row in rows if row["calibration_trials_per_class"] == k]
        before = np.asarray([row["before_accuracy"] for row in subset], dtype=np.float64)
        cal = np.asarray([row["calibration_only_accuracy"] for row in subset], dtype=np.float64)
        after = np.asarray([row["experience_after_accuracy"] for row in subset], dtype=np.float64)
        summary.append(
            {
                "calibration_trials_per_class": int(k),
                "subjects": int(len(subset)),
                "before_mean": float(before.mean()),
                "calibration_only_mean": float(cal.mean()),
                "experience_after_mean": float(after.mean()),
                "gain_vs_before_mean": float((after - before).mean()),
                "gain_vs_calibration_only_mean": float((after - cal).mean()),
                "subjects_improved_vs_before": int(np.sum(after > before)),
                "subjects_improved_vs_calibration_only": int(np.sum(after > cal)),
            }
        )
    return summary


def _save_results(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = payload["rows"]
    summary = payload["summary"]
    save_json(output_dir / "summary.json", {"summary": summary})
    save_json(
        output_dir / "rows.json",
        {
            "rows": [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "before_confusion",
                        "calibration_only_confusion",
                        "experience_confusion",
                        "candidate_weights",
                        "candidate_distances",
                        "candidate_calibration_accuracy",
                        "rolling_before",
                        "rolling_after",
                    }
                }
                for row in rows
            ]
        },
    )
    save_npz(
        output_dir / "arrays.npz",
        subjects=np.asarray([row["subject"] for row in rows], dtype=int),
        calibration_trials_per_class=np.asarray(
            [row["calibration_trials_per_class"] for row in rows],
            dtype=int,
        ),
        before_accuracy=np.asarray([row["before_accuracy"] for row in rows], dtype=np.float64),
        calibration_only_accuracy=np.asarray(
            [row["calibration_only_accuracy"] for row in rows],
            dtype=np.float64,
        ),
        experience_after_accuracy=np.asarray(
            [row["experience_after_accuracy"] for row in rows],
            dtype=np.float64,
        ),
        gain_vs_before=np.asarray([row["experience_gain_vs_before"] for row in rows], dtype=np.float64),
        gain_vs_calibration_only=np.asarray(
            [row["experience_gain_vs_calibration_only"] for row in rows],
            dtype=np.float64,
        ),
    )
