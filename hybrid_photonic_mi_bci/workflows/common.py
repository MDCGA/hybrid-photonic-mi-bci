"""Shared utilities for FBCSP design-line experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from ..datasets import DEFAULT_MOTOR_CHANNELS, DEFAULT_SUBJECTS, DEFAULT_WINDOW
from ..datasets import BCICIVTrials, load_pooled_subject_trials
from ..evaluation import decisions_from_scores, rolling_mean, softmax
from ..fbcsp import DEFAULT_FILTER_BANK, FBCSPFeatureSet, FilterBankCSP
from ..linear_models import FeatureStandardizer, fisher_scores, select_fisher_features


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class ReplaySplit:
    """Absolute and replay-relative indices for train/calibration/evaluation."""

    train: IntArray
    replay: IntArray
    calibration_abs: IntArray
    evaluation_abs: IntArray
    calibration_replay: IntArray
    evaluation_replay: IntArray
    replay_per_subject: int


@dataclass(frozen=True)
class FBCSPDesignConfig:
    """Configuration shared by the design-line workflows."""

    data_dir: Path | str = Path("Dataset/BCICIV_1_asc")
    metrics_dir: Path | str = Path("artifacts/metrics/fbcsp_design")
    subjects: tuple[str, ...] = DEFAULT_SUBJECTS
    channels: tuple[str, ...] = DEFAULT_MOTOR_CHANNELS
    window: tuple[float, float] = DEFAULT_WINDOW
    bands: tuple[tuple[float, float], ...] = DEFAULT_FILTER_BANK
    n_train_per_subject: int = 120
    calibration_trials_per_subject: int = 6
    csp_components: int = 2
    csp_shrinkage: float = 0.10
    selected_features: int = 32
    reject_target_rate: float = 0.02
    fixed_reject_threshold: float | None = None
    margin_threshold: float = 0.0
    seed: int = 13
    mlp_epochs: int = 220
    mlp_hidden_dim: int = 64
    mlp_embedding_dim: int = 32
    mlp_dropout: float = 0.20
    experience_entries: int = 64
    experience_top_k: int = 8
    experience_sample_fraction: float = 0.75
    experience_anchor_prior: float = 5.0
    tile_shape: tuple[int, int] = (2, 8)

    @property
    def n_train_total(self) -> int:
        return self.n_train_per_subject * len(self.subjects)

    @property
    def metrics_path(self) -> Path:
        return Path(self.metrics_dir)


@dataclass
class FBCSPPreparedData:
    """Prepared pooled trials and train-fitted FBCSP features."""

    config: FBCSPDesignConfig
    dataset: BCICIVTrials
    split: ReplaySplit
    train_features_raw: FloatArray
    replay_features_raw: FloatArray
    train_tensor: FloatArray
    replay_tensor: FloatArray
    selected_indices: IntArray
    fisher_scores: FloatArray
    feature_names: tuple[str, ...]

    @property
    def train_labels(self) -> IntArray:
        return self.dataset.labels[self.split.train]

    @property
    def replay_labels(self) -> IntArray:
        return self.dataset.labels[self.split.replay]

    @property
    def evaluation_labels(self) -> IntArray:
        return self.dataset.labels[self.split.evaluation_abs]

    @property
    def selected_feature_names(self) -> tuple[str, ...]:
        return tuple(self.feature_names[int(index)] for index in self.selected_indices)

    def selected_train_features(self) -> FloatArray:
        return self.train_features_raw[:, self.selected_indices]

    def selected_replay_features(self) -> FloatArray:
        return self.replay_features_raw[:, self.selected_indices]


def prepare_fbcsp_data(config: FBCSPDesignConfig) -> FBCSPPreparedData:
    """Load BCICIV_1_asc, fit FBCSP on train trials, and select features."""

    dataset = load_pooled_subject_trials(
        data_dir=config.data_dir,
        subjects=config.subjects,
        n_train_per_subject=config.n_train_per_subject,
        channels=config.channels,
        window=config.window,
    )
    split = make_replay_split(
        total_trials=len(dataset.labels),
        n_subjects=len(config.subjects),
        n_train_per_subject=config.n_train_per_subject,
        calibration_trials_per_subject=config.calibration_trials_per_subject,
    )
    train_trials = dataset.trials[split.train]
    replay_trials = dataset.trials[split.replay]
    train_labels = dataset.labels[split.train]
    fbcsp = FilterBankCSP(
        bands=config.bands,
        n_components=config.csp_components,
        covariance_shrinkage=config.csp_shrinkage,
    )
    train_set = fbcsp.fit_transform(
        train_trials,
        train_labels,
        fs=dataset.fs,
        class_names=dataset.class_names,
    )
    replay_set = fbcsp.transform(replay_trials)
    selected_indices = select_fisher_features(
        train_set.vector,
        train_labels,
        n_classes=len(dataset.class_names),
        n_features=config.selected_features,
    )
    fisher = fisher_scores(
        train_set.vector,
        train_labels,
        n_classes=len(dataset.class_names),
    )
    return FBCSPPreparedData(
        config=config,
        dataset=dataset,
        split=split,
        train_features_raw=train_set.vector,
        replay_features_raw=replay_set.vector,
        train_tensor=train_set.tensor,
        replay_tensor=replay_set.tensor,
        selected_indices=selected_indices,
        fisher_scores=fisher,
        feature_names=train_set.feature_names,
    )


def make_replay_split(
    total_trials: int,
    n_subjects: int,
    n_train_per_subject: int,
    calibration_trials_per_subject: int,
) -> ReplaySplit:
    """Build train/replay/calibration/evaluation indices for pooled BCICIV files."""

    n_train_total = n_subjects * n_train_per_subject
    if not 0 < n_train_total < total_trials:
        raise ValueError("train split must leave replay samples")
    n_replay = total_trials - n_train_total
    if n_replay % n_subjects:
        raise ValueError("pooled replay split must have equal subject blocks")
    replay_per_subject = n_replay // n_subjects
    if calibration_trials_per_subject >= replay_per_subject:
        raise ValueError("calibration trials per subject must leave evaluation trials")
    train = np.arange(n_train_total, dtype=int)
    replay = np.arange(n_train_total, total_trials, dtype=int)
    calibration = []
    evaluation = []
    for subject_index in range(n_subjects):
        start = n_train_total + subject_index * replay_per_subject
        calibration.extend(range(start, start + calibration_trials_per_subject))
        evaluation.extend(range(start + calibration_trials_per_subject, start + replay_per_subject))
    calibration_abs = np.asarray(calibration, dtype=int)
    evaluation_abs = np.asarray(evaluation, dtype=int)
    return ReplaySplit(
        train=train,
        replay=replay,
        calibration_abs=calibration_abs,
        evaluation_abs=evaluation_abs,
        calibration_replay=calibration_abs - n_train_total,
        evaluation_replay=evaluation_abs - n_train_total,
        replay_per_subject=replay_per_subject,
    )


def standardize_selected_features(
    prepared: FBCSPPreparedData,
) -> tuple[FeatureStandardizer, FloatArray, FloatArray]:
    """Fit a standardizer on selected train FBCSP features."""

    standardizer = FeatureStandardizer()
    train = standardizer.fit_transform(prepared.selected_train_features())
    replay = standardizer.transform(prepared.selected_replay_features())
    return standardizer, train, replay


def calibrated_reject_threshold(
    train_scores: FloatArray,
    config: FBCSPDesignConfig,
) -> float:
    """Choose a confidence threshold from train scores only."""

    if config.fixed_reject_threshold is not None:
        return float(config.fixed_reject_threshold)
    target = float(config.reject_target_rate)
    if target <= 0.0:
        return 0.0
    confidence = softmax(np.asarray(train_scores, dtype=np.float64)).max(axis=1)
    return float(np.quantile(confidence, min(max(target, 0.0), 0.95)))


def evaluate_scores(
    scores: FloatArray,
    labels: IntArray,
    class_names: tuple[str, ...],
    reject_threshold: float,
    margin_threshold: float,
    rolling_window: int = 40,
) -> dict[str, Any]:
    """Convert scores to decisions, metrics, and rolling traces."""

    decisions = decisions_from_scores(
        scores=scores,
        class_names=class_names,
        reject_threshold=reject_threshold,
        margin_threshold=margin_threshold,
    )
    from ..evaluation import classification_metrics

    metrics = classification_metrics(
        y_true=labels,
        predicted=decisions["predicted"],
        rejected=decisions["rejected"],
        n_classes=len(class_names),
    )
    correct = (
        (np.asarray(decisions["predicted"], dtype=int) == np.asarray(labels, dtype=int))
        & ~np.asarray(decisions["rejected"], dtype=bool)
    )
    metrics.update(
        {
            "probabilities": decisions["probabilities"],
            "predicted": decisions["predicted"],
            "rejected_mask": decisions["rejected"],
            "confidence": decisions["confidence"],
            "margin": decisions["margin"],
            "decision_labels": decisions["labels"],
            "correct_trace": correct.astype(np.float64),
            "rolling_command_accuracy": rolling_mean(correct.astype(np.float64), rolling_window),
            "rolling_reject_rate": rolling_mean(
                np.asarray(decisions["rejected"], dtype=np.float64),
                rolling_window,
            ),
        }
    )
    return metrics


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    return value


def summary_from_metrics(name: str, metrics: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "line": name,
        "total": int(metrics["total"]),
        "command_accuracy": float(metrics["command_accuracy"]),
        "balanced_command_accuracy": float(metrics["balanced_command_accuracy"]),
        "accepted_accuracy": float(metrics["accepted_accuracy"]),
        "reject_rate": float(metrics["reject_rate"]),
    }
    if extra:
        row.update(extra)
    return row


def class_count_dict(labels: Iterable[int], class_names: tuple[str, ...]) -> dict[str, int]:
    labels_arr = np.asarray(tuple(labels), dtype=int)
    return {
        class_names[index]: int(np.sum(labels_arr == index))
        for index in range(len(class_names))
    }


def feature_rank_table(prepared: FBCSPPreparedData, top_n: int = 24) -> list[dict[str, Any]]:
    rows = []
    for rank, index in enumerate(prepared.selected_indices[:top_n], start=1):
        rows.append({"rank": rank, "index": int(index), "feature": prepared.feature_names[int(index)]})
    return rows


def feature_set_to_arrays(feature_set: FBCSPFeatureSet) -> dict[str, Any]:
    return {
        "vector": feature_set.vector,
        "tensor": feature_set.tensor,
        "feature_names": np.asarray(feature_set.feature_names),
        "bands": np.asarray(feature_set.bands, dtype=np.float64),
        "class_names": np.asarray(feature_set.class_names),
    }
