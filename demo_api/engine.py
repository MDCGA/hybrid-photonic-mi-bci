"""Subject-personalized single-window inference engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from examples.run_single_window_inference import (
    PreparedRuntime,
    build_experience_entries,
    calibrate_reject_threshold,
    run_one_online_forward,
    timed,
)
from hybrid_photonic_mi_bci.backends import (
    NumpyMatrixOpsBackend,
    ScipySignalOpsBackend,
    get_matrix_ops_backend,
    use_matrix_ops_backend,
    use_signal_ops_backend,
)
from hybrid_photonic_mi_bci.datasets import DEFAULT_SUBJECTS, load_subject_trials
from hybrid_photonic_mi_bci.fbcsp import FilterBankCSP
from hybrid_photonic_mi_bci.linear_models import FeatureStandardizer, select_fisher_features
from hybrid_photonic_mi_bci.small_networks import SmallMLPConfig, train_small_mlp
from hybrid_photonic_mi_bci.workflows import FBCSPDesignConfig
from pure_runtime import PurePhotonicScanRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("uvicorn.error")
GLOBAL_CLASS_NAMES = ("left", "right", "foot")


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    title: str
    description: str
    timing_key: str | None = None


@dataclass(frozen=True)
class PersonalizedRuntime:
    subject: str
    training_subjects: tuple[str, ...]
    prepared: PreparedRuntime
    calibration_indices: np.ndarray
    evaluation_indices: np.ndarray
    setup_timings: dict[str, float]


STAGES = (
    StageDefinition("window", "EEG window", "Read the selected subject EEG evaluation window"),
    StageDefinition("fbcsp", "FBCSP feature extraction", "Apply the other-subject fitted filter bank and CSP projection", "fbcsp_transform_one_window"),
    StageDefinition("standardize", "Feature standardization", "Apply the other-subject fitted feature standardizer", "standardize_one_window"),
    StageDefinition("embedding", "Compact MLP encoding", "Run the compact MLP trained on other subjects", "small_mlp_forward_one_window"),
    StageDefinition("photonic_scan", "Personalized experience scan", "Scan experience heads selected by this subject's calibration windows", "pure_runtime_photonic_scan_one_window"),
    StageDefinition("decision", "Fusion and decision", "Apply this subject's calibrated rejection threshold"),
)


class LiveInferenceEngine:
    """Build leave-one-subject-out bases and personalize them with target calibration."""

    def __init__(self) -> None:
        self.config = self._build_config()
        self.subjects = tuple(self.config.subjects)
        self.subject_trials: dict[str, object] = {}
        self.runtimes: dict[str, PersonalizedRuntime] = {}
        self.load_timings: dict[str, float] = {}
        self._execution_lock = Lock()

    def prepare(self) -> None:
        data_dir = Path(self.config.data_dir)
        missing = [
            subject
            for subject in self.subjects
            if not (data_dir / f"BCICIV_calib_ds1{subject}_cnt.txt").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"BCICIV_1_asc subjects are missing under {data_dir}: {missing}. "
                "Run: python scripts/download_datasets.py --dataset bciciv-1-asc"
            )

        LOGGER.info("subject_data_load_started data_dir=%s subjects=%s", data_dir, self.subjects)
        timings: dict[str, float] = {}
        with self._execution_lock, use_matrix_ops_backend(
            NumpyMatrixOpsBackend()
        ), use_signal_ops_backend(ScipySignalOpsBackend()):
            for subject in self.subjects:
                self.subject_trials[subject] = timed(
                    f"load_subject_{subject}",
                    timings,
                    lambda subject=subject: load_subject_trials(
                        data_dir=data_dir,
                        subject=subject,
                        channels=self.config.channels,
                        window=self.config.window,
                    ),
                )
            self.load_timings = timings
            self._prepare_subject_locked(self.subjects[0])
        LOGGER.info(
            "subject_data_load_completed subjects=%d load_ms=%.3f default_subject=%s",
            len(self.subjects),
            sum(timings.values()) * 1000.0,
            self.subjects[0],
        )

    def metadata(self, subject: str | None = None) -> dict[str, object]:
        selected_subject = self._normalize_subject(subject or self.subjects[0])
        target = self._subject_dataset(selected_subject)
        evaluation_count = self._evaluation_count(target)
        runtime = self.runtimes.get(selected_subject)
        backend = get_matrix_ops_backend()
        return {
            "service": "hybrid-photonic-mi-bci-demo",
            "mode": "subject_personalized_inference",
            "mode_label": "Subject-personalized BCICIV single-window",
            "dataset": "BCICIV_1_asc",
            "subjects": list(self.subjects),
            "subject": selected_subject,
            "training_subjects": [item for item in self.subjects if item != selected_subject],
            "calibration_windows": self.config.calibration_trials_per_subject,
            "evaluation_count": evaluation_count,
            "default_evaluation_index": 0,
            "runtime_ready": runtime is not None,
            "class_names": [*GLOBAL_CLASS_NAMES, "reject"],
            "channel_names": list(target.channel_names),
            "sample_rate_hz": float(target.fs),
            "window_seconds": [float(value) for value in target.window],
            "backend": getattr(backend, "execution_backend", type(backend).__name__),
            "reject_threshold": None if runtime is None else runtime.prepared.reject_threshold,
            "selected_entries": [] if runtime is None else [
                entry.entry_id for entry in (runtime.prepared.runtime.selected_entries or ())
            ],
            "setup_timings_ms": {} if runtime is None else {
                name: round(elapsed * 1000.0, 3)
                for name, elapsed in runtime.setup_timings.items()
            },
            "stages": [self._stage_payload(stage) for stage in STAGES],
        }

    def window_info(self, subject: str, evaluation_index: int) -> dict[str, object]:
        selected_subject = self._normalize_subject(subject)
        target = self._subject_dataset(selected_subject)
        evaluation_indices = self._evaluation_indices(target)
        trial_index = self._trial_index(evaluation_indices, evaluation_index)
        target_labels = self._global_labels(target)
        true_index = int(target_labels[trial_index])
        return {
            "subject": selected_subject,
            "evaluation_index": evaluation_index,
            "subject_trial_index": trial_index,
            "true_label": GLOBAL_CLASS_NAMES[true_index],
            "signal": self._signal_payload(target, target.trials[trial_index]),
        }

    def infer(self, subject: str, evaluation_index: int) -> dict[str, object]:
        selected_subject = self._normalize_subject(subject)
        personalized = self._get_or_prepare_runtime(selected_subject)
        trial_index = self._trial_index(personalized.evaluation_indices, evaluation_index)
        prepared = personalized.prepared
        true_index = int(prepared.dataset.labels[trial_index])
        trial = prepared.dataset.trials[trial_index : trial_index + 1]
        timings: dict[str, float] = {}
        tile_counts: dict[str, int] = {}

        with self._execution_lock:
            output = run_one_online_forward(prepared, trial, timings, tile_counts)

        decision = output.decisions[0]
        probabilities = output.probabilities[0]
        backend = get_matrix_ops_backend()
        precision_summary = getattr(backend, "precision_summary", lambda: [])()
        return {
            "subject": selected_subject,
            "training_subjects": list(personalized.training_subjects),
            "calibration_windows": len(personalized.calibration_indices),
            "evaluation_index": evaluation_index,
            "subject_trial_index": trial_index,
            "true_label": GLOBAL_CLASS_NAMES[true_index],
            "predicted_label": decision.label,
            "predicted_index": decision.predicted_index,
            "rejected": decision.rejected,
            "confidence": decision.confidence,
            "margin": decision.margin,
            "probabilities": [
                {"label": label, "value": float(probabilities[index])}
                for index, label in enumerate(GLOBAL_CLASS_NAMES)
            ],
            "online_timings_ms": {
                name: round(elapsed * 1000.0, 3)
                for name, elapsed in timings.items()
            },
            "online_total_ms": round(sum(timings.values()) * 1000.0, 3),
            "online_tile_counts": tile_counts,
            "tile_count": sum(tile_counts.values()),
            "runtime_tile_count_per_window": output.tile_count_per_window,
            "reject_threshold": prepared.reject_threshold,
            "selected_entries": [
                entry.entry_id for entry in (prepared.runtime.selected_entries or ())
            ],
            "backend": getattr(backend, "execution_backend", type(backend).__name__),
            "precision": [self._precision_payload(item) for item in precision_summary],
        }

    def _get_or_prepare_runtime(self, subject: str) -> PersonalizedRuntime:
        existing = self.runtimes.get(subject)
        if existing is not None:
            return existing
        with self._execution_lock, use_matrix_ops_backend(
            NumpyMatrixOpsBackend()
        ), use_signal_ops_backend(ScipySignalOpsBackend()):
            existing = self.runtimes.get(subject)
            if existing is not None:
                return existing
            return self._prepare_subject_locked(subject)

    def _prepare_subject_locked(self, subject: str) -> PersonalizedRuntime:
        training_subjects = tuple(item for item in self.subjects if item != subject)
        target = self._subject_dataset(subject)
        target_labels = self._global_labels(target)
        train_trials = np.concatenate(
            [
                self._subject_dataset(item).trials[: self.config.n_train_per_subject]
                for item in training_subjects
            ],
            axis=0,
        )
        train_labels = np.concatenate(
            [
                self._global_labels(self._subject_dataset(item))[
                    : self.config.n_train_per_subject
                ]
                for item in training_subjects
            ],
            axis=0,
        )
        calibration_count = self.config.calibration_trials_per_subject
        if calibration_count >= len(target.trials) - self.config.n_train_per_subject:
            raise ValueError(f"subject {subject} has no evaluation trials after calibration")
        calibration_indices, evaluation_indices = self._personalized_indices(target)
        calibration_trials = target.trials[calibration_indices]

        LOGGER.info(
            "personalized_runtime_prepare_started subject=%s training_subjects=%s train_windows=%d calibration_windows=%d evaluation_windows=%d",
            subject,
            training_subjects,
            len(train_trials),
            calibration_count,
            len(evaluation_indices),
        )
        timings: dict[str, float] = {}
        fbcsp = FilterBankCSP(
            bands=self.config.bands,
            n_components=self.config.csp_components,
            filter_order=self.config.filter_order,
            covariance_shrinkage=self.config.csp_shrinkage,
        )
        train_set = timed(
            "fit_transform_fbcsp_other_subjects",
            timings,
            lambda: fbcsp.fit_transform(
                train_trials,
                train_labels,
                fs=target.fs,
                class_names=GLOBAL_CLASS_NAMES,
            ),
        )
        target_calibration_set = timed(
            "transform_fbcsp_target_calibration",
            timings,
            lambda: fbcsp.transform(calibration_trials),
        )
        selected_indices = timed(
            "select_fbcsp_features",
            timings,
            lambda: select_fisher_features(
                train_set.vector,
                train_labels,
                n_classes=len(GLOBAL_CLASS_NAMES),
                n_features=self.config.selected_features,
            ),
        )
        standardizer = FeatureStandardizer()
        train_features = timed(
            "standardize_other_subject_features",
            timings,
            lambda: standardizer.fit_transform(train_set.vector[:, selected_indices]),
        )
        target_calibration_features = timed(
            "standardize_target_calibration_features",
            timings,
            lambda: standardizer.transform(
                target_calibration_set.vector[:, selected_indices]
            ),
        )
        mlp = timed(
            "train_small_mlp_other_subjects",
            timings,
            lambda: train_small_mlp(
                train_features=train_features,
                train_labels=train_labels,
                replay_features=target_calibration_features,
                n_classes=len(GLOBAL_CLASS_NAMES),
                config=SmallMLPConfig(
                    hidden_dim=self.config.mlp_hidden_dim,
                    embedding_dim=self.config.mlp_embedding_dim,
                    dropout=self.config.mlp_dropout,
                    epochs=self.config.mlp_epochs,
                    seed=self.config.seed,
                ),
            ),
        )
        entries = timed(
            "build_other_subject_experience_library",
            timings,
            lambda: build_experience_entries(
                embeddings=mlp.train_embeddings,
                labels=train_labels,
                class_names=GLOBAL_CLASS_NAMES,
                mlp_model=mlp.model,
                n_entries=self.config.experience_entries,
                sample_fraction=self.config.experience_sample_fraction,
                seed=self.config.seed,
            ),
        )
        runtime = PurePhotonicScanRuntime(
            entries=entries,
            class_names=GLOBAL_CLASS_NAMES,
            top_k=self.config.experience_top_k,
            tile_shape=self.config.tile_shape,
            reject_threshold=0.0,
            margin_threshold=self.config.margin_threshold,
        )
        calibration_embeddings = mlp.replay_embeddings
        timed(
            "select_personalized_top_k",
            timings,
            lambda: runtime.calibrate(calibration_embeddings),
        )
        reject_threshold = timed(
            "calibrate_subject_reject_threshold",
            timings,
            lambda: calibrate_reject_threshold(
                runtime,
                train_embeddings=calibration_embeddings,
                target_rate=self.config.reject_target_rate,
                fixed_threshold=self.config.fixed_reject_threshold,
            ),
        )
        runtime.reject_threshold = reject_threshold
        target_dataset = replace(
            target,
            labels=target_labels,
            class_names=GLOBAL_CLASS_NAMES,
        )
        prepared = PreparedRuntime(
            dataset=target_dataset,
            split=None,
            fbcsp=fbcsp,
            selected_indices=selected_indices,
            standardizer=standardizer,
            mlp_model=mlp.model,
            runtime=runtime,
            reject_threshold=reject_threshold,
        )
        personalized = PersonalizedRuntime(
            subject=subject,
            training_subjects=training_subjects,
            prepared=prepared,
            calibration_indices=calibration_indices,
            evaluation_indices=evaluation_indices,
            setup_timings=timings,
        )
        self.runtimes[subject] = personalized
        LOGGER.info(
            "personalized_runtime_prepare_completed subject=%s setup_ms=%.3f selected_entries=%s reject_threshold=%.6f",
            subject,
            sum(timings.values()) * 1000.0,
            [entry.entry_id for entry in (runtime.selected_entries or ())],
            reject_threshold,
        )
        return personalized

    def _subject_dataset(self, subject: str):
        try:
            return self.subject_trials[subject]
        except KeyError as exc:
            raise RuntimeError("subject data has not been loaded") from exc

    def _normalize_subject(self, subject: str) -> str:
        normalized = subject.strip().lower()
        if normalized not in self.subjects:
            raise ValueError(f"subject must be one of {self.subjects}, got {subject!r}")
        return normalized

    def _evaluation_count(self, target) -> int:
        return len(self._evaluation_indices(target))

    def _evaluation_indices(self, target) -> np.ndarray:
        return self._personalized_indices(target)[1]

    def _personalized_indices(self, target) -> tuple[np.ndarray, np.ndarray]:
        labels = self._global_labels(target)
        replay_indices = np.arange(
            self.config.n_train_per_subject,
            len(target.trials),
            dtype=int,
        )
        replay_classes = np.unique(labels[replay_indices])
        calibration_count = self.config.calibration_trials_per_subject
        per_class = calibration_count // len(replay_classes)
        remainder = calibration_count % len(replay_classes)
        calibration: list[int] = []
        for class_offset, class_index in enumerate(replay_classes):
            required = per_class + int(class_offset < remainder)
            candidates = replay_indices[labels[replay_indices] == class_index]
            if len(candidates) <= required:
                raise ValueError(
                    f"subject {target.subject} class {class_index} has too few replay trials"
                )
            calibration.extend(candidates[:required].tolist())
        calibration_indices = np.asarray(sorted(calibration), dtype=int)
        calibration_set = set(calibration_indices.tolist())
        evaluation_indices = np.asarray(
            [index for index in replay_indices if int(index) not in calibration_set],
            dtype=int,
        )
        return calibration_indices, evaluation_indices

    @staticmethod
    def _trial_index(evaluation_indices: np.ndarray, evaluation_index: int) -> int:
        if not 0 <= evaluation_index < len(evaluation_indices):
            raise ValueError(
                f"evaluation_index must be in [0, {len(evaluation_indices) - 1}], got {evaluation_index}"
            )
        return int(evaluation_indices[evaluation_index])

    @staticmethod
    def _global_labels(dataset) -> np.ndarray:
        mapping = {name: index for index, name in enumerate(GLOBAL_CLASS_NAMES)}
        return np.asarray(
            [mapping[dataset.class_names[int(label)]] for label in dataset.labels],
            dtype=int,
        )

    @staticmethod
    def _signal_payload(dataset, trial: np.ndarray) -> dict[str, object]:
        return {
            "sample_rate_hz": float(dataset.fs),
            "duration_seconds": float(trial.shape[-1] / dataset.fs),
            "source": f"BCICIV_1_asc/{dataset.subject}",
            "channels": [
                {"name": name, "values": trial[index].astype(float).tolist()}
                for index, name in enumerate(dataset.channel_names)
            ],
        }

    @staticmethod
    def _stage_payload(stage: StageDefinition) -> dict[str, object]:
        return {
            "id": stage.stage_id,
            "title": stage.title,
            "description": stage.description,
            "timing_key": stage.timing_key,
        }

    @staticmethod
    def _precision_payload(item: dict[str, Any]) -> dict[str, object]:
        return {
            "policy": item["policy"],
            "bits": item["current_bits"],
            "escalations": item["escalations"],
            "max_shadow_error": round(float(item["max_8bit_shadow_error"]), 4),
        }

    @staticmethod
    def _build_config() -> FBCSPDesignConfig:
        configured_path = Path(os.getenv("DEMO_DATA_DIR", "Dataset/BCICIV_1_asc"))
        data_dir = configured_path if configured_path.is_absolute() else PROJECT_ROOT / configured_path
        return FBCSPDesignConfig(data_dir=data_dir)
