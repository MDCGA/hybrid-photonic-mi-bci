"""Multi-dataset subject-personalized single-window inference engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Callable, TypeVar

import numpy as np

from examples.run_single_window_inference import (
    PreparedRuntime,
    build_experience_entries,
    calibrate_reject_threshold,
    run_one_online_forward,
)
from hybrid_photonic_mi_bci.backends import (
    NumpyMatrixOpsBackend,
    ScipySignalOpsBackend,
    get_matrix_ops_backend,
    use_matrix_ops_backend,
    use_signal_ops_backend,
)
from hybrid_photonic_mi_bci.datasets import (
    DEFAULT_SUBJECTS as BCICIV_DEFAULT_SUBJECTS,
    load_subject_trials,
)
from hybrid_photonic_mi_bci.datasets.bnci2014_004 import (
    DEFAULT_SUBJECTS as BNCI_DEFAULT_SUBJECTS,
    calibration_eval_split,
    load_subject_history_and_target,
)
from hybrid_photonic_mi_bci.fbcsp import FilterBankCSP
from hybrid_photonic_mi_bci.linear_models import FeatureStandardizer, select_fisher_features
from hybrid_photonic_mi_bci.small_networks import SmallMLPConfig, train_small_mlp
from hybrid_photonic_mi_bci.workflows import FBCSPDesignConfig
from hybrid_photonic_mi_bci.workflows.bnci2014_004_personalization import (
    BNCI004PersonalizationConfig,
)
from pure_runtime import PurePhotonicScanRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("uvicorn.error")

BCICIV_DATASET_KEY = "bciciv_1_asc"
BNCI_DATASET_KEY = "bnci2014_004"
DEFAULT_DATASET_KEY = BCICIV_DATASET_KEY
BCICIV_GLOBAL_CLASS_NAMES = ("left", "right", "foot")
BNCI_CLASS_NAMES = ("left", "right")
FFT_DISPLAY_MAX_HZ = 40.0
FFT_DISPLAY_FLOOR_DB = -60.0

DATASET_ALIASES = {
    "1": BCICIV_DATASET_KEY,
    "dataset1": BCICIV_DATASET_KEY,
    "dataset_1": BCICIV_DATASET_KEY,
    "bciciv": BCICIV_DATASET_KEY,
    "bciciv1": BCICIV_DATASET_KEY,
    "bciciv_1": BCICIV_DATASET_KEY,
    "bciciv1asc": BCICIV_DATASET_KEY,
    "bciciv_1_asc": BCICIV_DATASET_KEY,
    "bciciv-1-asc": BCICIV_DATASET_KEY,
    "2": BNCI_DATASET_KEY,
    "dataset2": BNCI_DATASET_KEY,
    "dataset_2": BNCI_DATASET_KEY,
    "bnci": BNCI_DATASET_KEY,
    "bnci004": BNCI_DATASET_KEY,
    "bnci2014_004": BNCI_DATASET_KEY,
    "bnci2014-004": BNCI_DATASET_KEY,
    "bnci2014004": BNCI_DATASET_KEY,
    "bciciv2b": BNCI_DATASET_KEY,
    "bciciv_2b": BNCI_DATASET_KEY,
    "2b": BNCI_DATASET_KEY,
}

T = TypeVar("T")


def _normalized_fft_spectrum(
    trial: np.ndarray,
    sample_rate_hz: float,
    *,
    max_frequency_hz: float = FFT_DISPLAY_MAX_HZ,
    floor_db: float = FFT_DISPLAY_FLOOR_DB,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a Hann-windowed, per-channel normalized one-sided FFT in dB."""

    samples = np.asarray(trial, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError("trial must have shape (channels, samples) with at least two samples")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if max_frequency_hz <= 0:
        raise ValueError("max_frequency_hz must be positive")
    if floor_db >= 0:
        raise ValueError("floor_db must be negative")

    centered = samples - samples.mean(axis=1, keepdims=True)
    tapered = centered * np.hanning(samples.shape[1])[None, :]
    magnitudes = np.abs(np.fft.rfft(tapered, axis=1))
    frequencies = np.fft.rfftfreq(samples.shape[1], d=1.0 / sample_rate_hz)
    frequency_mask = frequencies <= min(max_frequency_hz, sample_rate_hz / 2.0)
    frequencies = frequencies[frequency_mask]
    magnitudes = magnitudes[:, frequency_mask]

    channel_peaks = magnitudes.max(axis=1, keepdims=True)
    channel_peaks = np.where(channel_peaks > 0.0, channel_peaks, 1.0)
    minimum_ratio = 10.0 ** (floor_db / 20.0)
    normalized = np.maximum(magnitudes / channel_peaks, minimum_ratio)
    return frequencies, np.maximum(20.0 * np.log10(normalized), floor_db)


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    title: str
    description: str
    timing_key: str | None = None


@dataclass(frozen=True)
class DatasetDefinition:
    key: str
    name: str
    label: str
    mode_label: str
    subjects: tuple[str, ...]
    default_subject: str
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeInputs:
    dataset_key: str
    dataset_name: str
    dataset_label: str
    subject: str
    subject_label: str
    training_subjects: tuple[str, ...]
    training_summary: str
    calibration_summary: str
    target: object
    train_trials: np.ndarray
    train_labels: np.ndarray
    target_labels: np.ndarray
    calibration_indices: np.ndarray
    evaluation_indices: np.ndarray
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class PersonalizedRuntime:
    dataset_key: str
    subject: str
    training_subjects: tuple[str, ...]
    prepared: PreparedRuntime
    calibration_indices: np.ndarray
    evaluation_indices: np.ndarray
    setup_timings: dict[str, float]
    training_summary: str
    calibration_summary: str


STAGES = (
    StageDefinition("window", "EEG window", "Read the selected EEG evaluation window"),
    StageDefinition(
        "fbcsp",
        "FBCSP feature extraction",
        "Apply filter-bank CSP to the selected raw EEG window",
        "fbcsp_transform_one_window",
    ),
    StageDefinition(
        "standardize",
        "Feature standardization",
        "Apply the train-fitted feature standardizer",
        "standardize_one_window",
    ),
    StageDefinition(
        "embedding",
        "Compact MLP encoding",
        "Run the compact MLP embedding model",
        "small_mlp_forward_one_window",
    ),
    StageDefinition(
        "photonic_scan",
        "Personalized experience scan",
        "Scan experience heads selected by calibration windows",
        "pure_runtime_photonic_scan_one_window",
    ),
    StageDefinition("decision", "Fusion and decision", "Apply calibrated rejection logic"),
)


class LiveInferenceEngine:
    """Build and cache single-window demo runtimes for BCICIV and BNCI datasets."""

    def __init__(self) -> None:
        self.bciciv_config = self._build_bciciv_config()
        self.bnci_config = self._build_bnci_config()
        self.dataset_definitions = self._build_dataset_definitions()
        self._bciciv_trials: dict[str, object] = {}
        self._bnci_sessions: dict[str, tuple[object, object]] = {}
        self.runtimes: dict[tuple[str, str], PersonalizedRuntime] = {}
        self.load_timings: dict[str, dict[str, float]] = {
            BCICIV_DATASET_KEY: {},
            BNCI_DATASET_KEY: {},
        }
        self._execution_lock = Lock()

    def prepare(self) -> None:
        """Load the default dataset and warm the default subject runtime."""

        LOGGER.info(
            "demo_engine_prepare_started default_dataset=%s datasets=%s",
            DEFAULT_DATASET_KEY,
            [item["key"] for item in self.dataset_options()],
        )
        self._log_dataset_file_status()
        with self._execution_lock, use_matrix_ops_backend(
            NumpyMatrixOpsBackend()
        ), use_signal_ops_backend(ScipySignalOpsBackend()):
            self._ensure_bciciv_loaded()
            default_subject = self.dataset_definitions[BCICIV_DATASET_KEY].default_subject
            self._prepare_subject_locked(BCICIV_DATASET_KEY, default_subject)
        LOGGER.info(
            "demo_engine_prepare_completed default_dataset=%s cached_bciciv_subjects=%d cached_runtimes=%d",
            DEFAULT_DATASET_KEY,
            len(self._bciciv_trials),
            len(self.runtimes),
        )

    def dataset_options(self) -> list[dict[str, object]]:
        return [
            {
                "key": definition.key,
                "name": definition.name,
                "label": definition.label,
                "mode_label": definition.mode_label,
                "subjects": list(definition.subjects),
                "default_subject": definition.default_subject,
                "command_class_names": list(definition.class_names),
                "available": not self._dataset_file_missing(definition.key),
            }
            for definition in self.dataset_definitions.values()
        ]

    def runtime_ready(self, dataset: str | None, subject: str | None) -> bool:
        dataset_key = self._normalize_dataset(dataset)
        selected_subject = self._normalize_subject(dataset_key, subject)
        return (dataset_key, selected_subject) in self.runtimes

    def metadata(
        self,
        subject: str | None = None,
        dataset: str | None = None,
    ) -> dict[str, object]:
        dataset_key = self._normalize_dataset(dataset)
        selected_subject = self._normalize_subject(dataset_key, subject)
        definition = self.dataset_definitions[dataset_key]
        target = self._target_dataset(dataset_key, selected_subject)
        target_labels = self._target_labels(dataset_key, target)
        calibration_indices, evaluation_indices = self._calibration_and_evaluation_indices(
            dataset_key,
            target,
            target_labels,
        )
        runtime = self.runtimes.get((dataset_key, selected_subject))
        backend = get_matrix_ops_backend()
        training_subjects = self._training_subjects(dataset_key, selected_subject)
        return {
            "service": "hybrid-photonic-mi-bci-demo",
            "mode": "multi_dataset_subject_personalized_inference",
            "mode_label": definition.mode_label,
            "dataset_key": dataset_key,
            "dataset": definition.name,
            "dataset_label": definition.label,
            "datasets": self.dataset_options(),
            "subjects": list(definition.subjects),
            "subject": selected_subject,
            "subject_label": self._subject_label(dataset_key, selected_subject),
            "training_subjects": list(training_subjects),
            "training_summary": self._training_summary(dataset_key, selected_subject),
            "calibration_summary": self._calibration_summary(
                dataset_key,
                selected_subject,
                len(calibration_indices),
            ),
            "calibration_windows": int(len(calibration_indices)),
            "evaluation_count": int(len(evaluation_indices)),
            "default_evaluation_index": 0,
            "runtime_ready": runtime is not None,
            "class_names": [*definition.class_names, "reject"],
            "command_class_names": list(definition.class_names),
            "channel_names": list(
                self._display_channel_names(dataset_key, target.channel_names)
            ),
            "sample_rate_hz": float(target.fs),
            "window_seconds": [float(value) for value in target.window],
            "backend": getattr(backend, "execution_backend", type(backend).__name__),
            "reject_threshold": None if runtime is None else runtime.prepared.reject_threshold,
            "selected_entries": []
            if runtime is None
            else [entry.entry_id for entry in (runtime.prepared.runtime.selected_entries or ())],
            "setup_timings_ms": {}
            if runtime is None
            else {
                name: round(elapsed * 1000.0, 3)
                for name, elapsed in runtime.setup_timings.items()
            },
            "stages": [self._stage_payload(stage) for stage in STAGES],
        }

    def window_info(
        self,
        subject: str | None,
        evaluation_index: int,
        dataset: str | None = None,
    ) -> dict[str, object]:
        dataset_key = self._normalize_dataset(dataset)
        selected_subject = self._normalize_subject(dataset_key, subject)
        definition = self.dataset_definitions[dataset_key]
        target = self._target_dataset(dataset_key, selected_subject)
        target_labels = self._target_labels(dataset_key, target)
        _calibration_indices, evaluation_indices = self._calibration_and_evaluation_indices(
            dataset_key,
            target,
            target_labels,
        )
        trial_index = self._trial_index(evaluation_indices, evaluation_index)
        true_index = int(target_labels[trial_index])
        true_label = definition.class_names[true_index]
        LOGGER.info(
            "window_selected dataset=%s subject=%s evaluation_index=%d subject_trial_index=%d true_label=%s signal_shape=%s",
            dataset_key,
            selected_subject,
            evaluation_index,
            trial_index,
            true_label,
            tuple(target.trials[trial_index].shape),
        )
        return {
            "dataset_key": dataset_key,
            "dataset": definition.name,
            "dataset_label": definition.label,
            "subject": selected_subject,
            "subject_label": self._subject_label(dataset_key, selected_subject),
            "evaluation_index": evaluation_index,
            "subject_trial_index": trial_index,
            "true_label": true_label,
            "signal": self._signal_payload(dataset_key, target, target.trials[trial_index]),
        }

    def infer(
        self,
        subject: str | None,
        evaluation_index: int,
        dataset: str | None = None,
    ) -> dict[str, object]:
        dataset_key = self._normalize_dataset(dataset)
        selected_subject = self._normalize_subject(dataset_key, subject)
        personalized = self._get_or_prepare_runtime(dataset_key, selected_subject)
        trial_index = self._trial_index(personalized.evaluation_indices, evaluation_index)
        prepared = personalized.prepared
        true_index = int(prepared.dataset.labels[trial_index])
        true_label = prepared.dataset.class_names[true_index]
        trial = prepared.dataset.trials[trial_index : trial_index + 1]
        timings: dict[str, float] = {}
        tile_counts: dict[str, int] = {}

        LOGGER.info(
            "online_forward_started dataset=%s subject=%s evaluation_index=%d subject_trial_index=%d true_label=%s",
            dataset_key,
            selected_subject,
            evaluation_index,
            trial_index,
            true_label,
        )
        with self._execution_lock:
            output = self._timed_step(
                dataset_key,
                selected_subject,
                "online_forward_total",
                {},
                lambda: run_one_online_forward(prepared, trial, timings, tile_counts),
            )

        decision = output.decisions[0]
        probabilities = output.probabilities[0]
        backend = get_matrix_ops_backend()
        precision_summary = getattr(backend, "precision_summary", lambda: [])()
        online_total_ms = round(sum(timings.values()) * 1000.0, 3)
        tile_count = int(sum(tile_counts.values()))
        LOGGER.info(
            "online_forward_completed dataset=%s subject=%s evaluation_index=%d predicted_label=%s rejected=%s confidence=%.6f margin=%.6f online_ms=%.3f tiles=%d timings_ms=%s tile_counts=%s",
            dataset_key,
            selected_subject,
            evaluation_index,
            decision.label,
            decision.rejected,
            decision.confidence,
            decision.margin,
            online_total_ms,
            tile_count,
            {name: round(elapsed * 1000.0, 3) for name, elapsed in timings.items()},
            tile_counts,
        )
        return {
            "dataset_key": dataset_key,
            "dataset": self.dataset_definitions[dataset_key].name,
            "dataset_label": self.dataset_definitions[dataset_key].label,
            "subject": selected_subject,
            "subject_label": self._subject_label(dataset_key, selected_subject),
            "training_subjects": list(personalized.training_subjects),
            "training_summary": personalized.training_summary,
            "calibration_summary": personalized.calibration_summary,
            "calibration_windows": int(len(personalized.calibration_indices)),
            "evaluation_index": evaluation_index,
            "subject_trial_index": trial_index,
            "true_label": true_label,
            "predicted_label": decision.label,
            "predicted_index": decision.predicted_index,
            "rejected": decision.rejected,
            "confidence": decision.confidence,
            "margin": decision.margin,
            "probabilities": [
                {"label": label, "value": float(probabilities[index])}
                for index, label in enumerate(prepared.dataset.class_names)
            ],
            "online_timings_ms": {
                name: round(elapsed * 1000.0, 3)
                for name, elapsed in timings.items()
            },
            "online_total_ms": online_total_ms,
            "online_tile_counts": tile_counts,
            "tile_count": tile_count,
            "runtime_tile_count_per_window": output.tile_count_per_window,
            "reject_threshold": prepared.reject_threshold,
            "selected_entries": [
                entry.entry_id for entry in (prepared.runtime.selected_entries or ())
            ],
            "backend": getattr(backend, "execution_backend", type(backend).__name__),
            "precision": [self._precision_payload(item) for item in precision_summary],
        }

    def _get_or_prepare_runtime(
        self,
        dataset_key: str,
        subject: str,
    ) -> PersonalizedRuntime:
        cache_key = (dataset_key, subject)
        existing = self.runtimes.get(cache_key)
        if existing is not None:
            LOGGER.info(
                "personalized_runtime_cache_hit dataset=%s subject=%s",
                dataset_key,
                subject,
            )
            return existing
        with self._execution_lock, use_matrix_ops_backend(
            NumpyMatrixOpsBackend()
        ), use_signal_ops_backend(ScipySignalOpsBackend()):
            existing = self.runtimes.get(cache_key)
            if existing is not None:
                return existing
            return self._prepare_subject_locked(dataset_key, subject)

    def _prepare_subject_locked(
        self,
        dataset_key: str,
        subject: str,
    ) -> PersonalizedRuntime:
        inputs = self._build_runtime_inputs(dataset_key, subject)
        config = self._config(dataset_key)
        calibration_trials = inputs.target.trials[inputs.calibration_indices]
        LOGGER.info(
            "personalized_runtime_prepare_started dataset=%s subject=%s training_subjects=%s train_windows=%d calibration_windows=%d evaluation_windows=%d class_names=%s channels=%s samples_per_window=%d",
            dataset_key,
            subject,
            inputs.training_subjects,
            len(inputs.train_trials),
            len(inputs.calibration_indices),
            len(inputs.evaluation_indices),
            inputs.class_names,
            self._display_channel_names(dataset_key, inputs.target.channel_names),
            inputs.target.trials.shape[-1],
        )
        timings: dict[str, float] = {}
        fbcsp = FilterBankCSP(
            bands=config.bands,
            n_components=config.csp_components,
            filter_order=config.filter_order,
            covariance_shrinkage=config.csp_shrinkage,
        )
        train_set = self._timed_step(
            dataset_key,
            subject,
            "fit_transform_fbcsp_train",
            timings,
            lambda: fbcsp.fit_transform(
                inputs.train_trials,
                inputs.train_labels,
                fs=inputs.target.fs,
                class_names=inputs.class_names,
            ),
        )
        target_calibration_set = self._timed_step(
            dataset_key,
            subject,
            "transform_fbcsp_calibration",
            timings,
            lambda: fbcsp.transform(calibration_trials),
        )
        selected_indices = self._timed_step(
            dataset_key,
            subject,
            "select_fbcsp_features",
            timings,
            lambda: select_fisher_features(
                train_set.vector,
                inputs.train_labels,
                n_classes=len(inputs.class_names),
                n_features=config.selected_features,
            ),
        )
        standardizer = FeatureStandardizer()
        train_features = self._timed_step(
            dataset_key,
            subject,
            "standardize_train_features",
            timings,
            lambda: standardizer.fit_transform(train_set.vector[:, selected_indices]),
        )
        target_calibration_features = self._timed_step(
            dataset_key,
            subject,
            "standardize_calibration_features",
            timings,
            lambda: standardizer.transform(
                target_calibration_set.vector[:, selected_indices]
            ),
        )
        mlp = self._timed_step(
            dataset_key,
            subject,
            "train_small_mlp",
            timings,
            lambda: train_small_mlp(
                train_features=train_features,
                train_labels=inputs.train_labels,
                replay_features=target_calibration_features,
                n_classes=len(inputs.class_names),
                config=SmallMLPConfig(
                    hidden_dim=config.mlp_hidden_dim,
                    embedding_dim=config.mlp_embedding_dim,
                    dropout=config.mlp_dropout,
                    epochs=config.mlp_epochs,
                    seed=self._seed(dataset_key, subject),
                ),
            ),
        )
        entries = self._timed_step(
            dataset_key,
            subject,
            "build_experience_library",
            timings,
            lambda: build_experience_entries(
                embeddings=mlp.train_embeddings,
                labels=inputs.train_labels,
                class_names=inputs.class_names,
                mlp_model=mlp.model,
                n_entries=config.experience_entries,
                sample_fraction=config.experience_sample_fraction,
                seed=self._seed(dataset_key, subject),
            ),
        )
        runtime = PurePhotonicScanRuntime(
            entries=entries,
            class_names=inputs.class_names,
            top_k=self._experience_top_k(dataset_key),
            tile_shape=config.tile_shape,
            reject_threshold=0.0,
            margin_threshold=config.margin_threshold,
        )
        calibration_embeddings = mlp.replay_embeddings
        self._timed_step(
            dataset_key,
            subject,
            "select_personalized_top_k",
            timings,
            lambda: runtime.calibrate(calibration_embeddings),
        )
        reject_threshold = self._timed_step(
            dataset_key,
            subject,
            "calibrate_reject_threshold",
            timings,
            lambda: calibrate_reject_threshold(
                runtime,
                train_embeddings=calibration_embeddings,
                target_rate=config.reject_target_rate,
                fixed_threshold=self._fixed_reject_threshold(dataset_key),
            ),
        )
        runtime.reject_threshold = reject_threshold
        target_dataset = replace(
            inputs.target,
            labels=inputs.target_labels,
            class_names=inputs.class_names,
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
            dataset_key=dataset_key,
            subject=subject,
            training_subjects=inputs.training_subjects,
            prepared=prepared,
            calibration_indices=inputs.calibration_indices,
            evaluation_indices=inputs.evaluation_indices,
            setup_timings=timings,
            training_summary=inputs.training_summary,
            calibration_summary=inputs.calibration_summary,
        )
        self.runtimes[(dataset_key, subject)] = personalized
        LOGGER.info(
            "personalized_runtime_prepare_completed dataset=%s subject=%s setup_ms=%.3f selected_entries=%s reject_threshold=%.6f cache_size=%d",
            dataset_key,
            subject,
            sum(timings.values()) * 1000.0,
            [entry.entry_id for entry in (runtime.selected_entries or ())],
            reject_threshold,
            len(self.runtimes),
        )
        return personalized

    def _build_runtime_inputs(self, dataset_key: str, subject: str) -> RuntimeInputs:
        definition = self.dataset_definitions[dataset_key]
        target = self._target_dataset(dataset_key, subject)
        target_labels = self._target_labels(dataset_key, target)
        calibration_indices, evaluation_indices = self._calibration_and_evaluation_indices(
            dataset_key,
            target,
            target_labels,
        )
        if dataset_key == BCICIV_DATASET_KEY:
            config = self.bciciv_config
            training_subjects = tuple(item for item in definition.subjects if item != subject)
            train_trials = np.concatenate(
                [
                    self._bciciv_trials[item].trials[: config.n_train_per_subject]
                    for item in training_subjects
                ],
                axis=0,
            )
            train_labels = np.concatenate(
                [
                    self._target_labels(BCICIV_DATASET_KEY, self._bciciv_trials[item])[
                        : config.n_train_per_subject
                    ]
                    for item in training_subjects
                ],
                axis=0,
            )
        else:
            history, _target = self._ensure_bnci_subject_loaded(subject)
            training_subjects = self._training_subjects(dataset_key, subject)
            train_trials = history.trials
            train_labels = history.labels
        if not len(evaluation_indices):
            raise ValueError(f"{definition.name} subject {subject} has no evaluation windows")
        return RuntimeInputs(
            dataset_key=dataset_key,
            dataset_name=definition.name,
            dataset_label=definition.label,
            subject=subject,
            subject_label=self._subject_label(dataset_key, subject),
            training_subjects=training_subjects,
            training_summary=self._training_summary(dataset_key, subject),
            calibration_summary=self._calibration_summary(
                dataset_key,
                subject,
                len(calibration_indices),
            ),
            target=target,
            train_trials=np.asarray(train_trials, dtype=np.float64),
            train_labels=np.asarray(train_labels, dtype=int),
            target_labels=target_labels,
            calibration_indices=calibration_indices,
            evaluation_indices=evaluation_indices,
            class_names=definition.class_names,
        )

    def _ensure_bciciv_loaded(self) -> None:
        if len(self._bciciv_trials) == len(self.dataset_definitions[BCICIV_DATASET_KEY].subjects):
            return
        data_dir = Path(self.bciciv_config.data_dir)
        missing = self._bciciv_missing_files()
        if missing:
            raise FileNotFoundError(
                f"BCICIV_1_asc subjects are missing under {data_dir}: {missing[:5]}. "
                "Run: python scripts/download_datasets.py --dataset bciciv-1-asc"
            )

        LOGGER.info(
            "dataset_load_started dataset=%s data_dir=%s subjects=%s",
            BCICIV_DATASET_KEY,
            data_dir,
            self.dataset_definitions[BCICIV_DATASET_KEY].subjects,
        )
        timings = self.load_timings[BCICIV_DATASET_KEY]
        with use_matrix_ops_backend(NumpyMatrixOpsBackend()), use_signal_ops_backend(
            ScipySignalOpsBackend()
        ):
            for subject in self.dataset_definitions[BCICIV_DATASET_KEY].subjects:
                if subject in self._bciciv_trials:
                    continue
                self._bciciv_trials[subject] = self._timed_load(
                    BCICIV_DATASET_KEY,
                    subject,
                    f"load_subject_{subject}",
                    timings,
                    lambda subject=subject: load_subject_trials(
                        data_dir=data_dir,
                        subject=subject,
                        channels=self.bciciv_config.channels,
                        window=self.bciciv_config.window,
                    ),
                )
        LOGGER.info(
            "dataset_load_completed dataset=%s subjects=%d load_ms=%.3f",
            BCICIV_DATASET_KEY,
            len(self._bciciv_trials),
            sum(timings.values()) * 1000.0,
        )

    def _ensure_bnci_subject_loaded(self, subject: str) -> tuple[object, object]:
        cached = self._bnci_sessions.get(subject)
        if cached is not None:
            return cached
        data_dir = Path(self.bnci_config.data_dir)
        missing = self._bnci_missing_files(subjects=(subject,))
        if missing:
            raise FileNotFoundError(
                f"BNCI2014_004 subject {subject} is missing labeled GDF files under {data_dir}: {missing}. "
                "Run: python scripts/download_datasets.py --dataset bnci2014-004"
            )
        timings = self.load_timings[BNCI_DATASET_KEY]
        loaded = self._timed_load(
            BNCI_DATASET_KEY,
            subject,
            f"load_subject_{subject}",
            timings,
            lambda: load_subject_history_and_target(
                data_dir=data_dir,
                subject=int(subject),
                history_sessions=(1, 2),
                target_session=3,
            ),
        )
        self._bnci_sessions[subject] = loaded
        return loaded

    def _target_dataset(self, dataset_key: str, subject: str):
        if dataset_key == BCICIV_DATASET_KEY:
            self._ensure_bciciv_loaded()
            try:
                return self._bciciv_trials[subject]
            except KeyError as exc:
                raise RuntimeError("BCICIV subject data has not been loaded") from exc
        _history, target = self._ensure_bnci_subject_loaded(subject)
        return target

    def _target_labels(self, dataset_key: str, target) -> np.ndarray:
        if dataset_key == BCICIV_DATASET_KEY:
            return self._bciciv_global_labels(target)
        return np.asarray(target.labels, dtype=int)

    def _calibration_and_evaluation_indices(
        self,
        dataset_key: str,
        target,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if dataset_key == BCICIV_DATASET_KEY:
            return self._bciciv_personalized_indices(target, labels)
        return calibration_eval_split(
            labels,
            trials_per_class=self.bnci_config.calibration_trials_per_class,
            n_classes=len(self.dataset_definitions[BNCI_DATASET_KEY].class_names),
        )

    def _bciciv_personalized_indices(
        self,
        target,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        replay_indices = np.arange(
            self.bciciv_config.n_train_per_subject,
            len(target.trials),
            dtype=int,
        )
        replay_classes = np.unique(labels[replay_indices])
        calibration_count = self.bciciv_config.calibration_trials_per_subject
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

    def _training_subjects(self, dataset_key: str, subject: str) -> tuple[str, ...]:
        if dataset_key == BCICIV_DATASET_KEY:
            return tuple(
                item
                for item in self.dataset_definitions[BCICIV_DATASET_KEY].subjects
                if item != subject
            )
        return (f"B{int(subject):02d}01T", f"B{int(subject):02d}02T")

    def _training_summary(self, dataset_key: str, subject: str) -> str:
        if dataset_key == BCICIV_DATASET_KEY:
            return "基础训练：" + "、".join(self._training_subjects(dataset_key, subject))
        return f"历史训练：B{int(subject):02d} session 01T-02T"

    def _calibration_summary(
        self,
        dataset_key: str,
        subject: str,
        calibration_windows: int,
    ) -> str:
        if dataset_key == BCICIV_DATASET_KEY:
            return f"个体校准：受试者 {subject}（{calibration_windows} 个窗口）"
        return f"目标校准：B{int(subject):02d} session 03T（{calibration_windows} 个窗口）"

    def _subject_label(self, dataset_key: str, subject: str) -> str:
        if dataset_key == BCICIV_DATASET_KEY:
            return subject
        return f"B{int(subject):02d}"

    def _signal_payload(
        self,
        dataset_key: str,
        dataset,
        trial: np.ndarray,
    ) -> dict[str, object]:
        channel_names = self._display_channel_names(dataset_key, dataset.channel_names)
        frequencies, spectrum_db = _normalized_fft_spectrum(trial, float(dataset.fs))
        LOGGER.info(
            "signal_payload_created dataset=%s subject=%s channels=%d samples=%d sample_rate_hz=%.3f fft_bins=%d fft_range_hz=0-%.3f",
            dataset_key,
            dataset.subject,
            len(channel_names),
            trial.shape[-1],
            float(dataset.fs),
            len(frequencies),
            float(frequencies[-1]),
        )
        return {
            "sample_rate_hz": float(dataset.fs),
            "duration_seconds": float(trial.shape[-1] / dataset.fs),
            "source": f"{self.dataset_definitions[dataset_key].name}/{self._subject_label(dataset_key, str(dataset.subject))}",
            "channels": [
                {"name": name, "values": trial[index].astype(float).tolist()}
                for index, name in enumerate(channel_names)
            ],
            "spectrum": {
                "method": "demean_hann_rfft",
                "unit": "dB_relative_to_channel_peak",
                "floor_db": FFT_DISPLAY_FLOOR_DB,
                "max_frequency_hz": float(frequencies[-1]),
                "frequencies_hz": np.round(frequencies, 4).tolist(),
                "channels": [
                    {
                        "name": name,
                        "values_db": np.round(spectrum_db[index], 3).tolist(),
                    }
                    for index, name in enumerate(channel_names)
                ],
            },
        }

    def _timed_step(
        self,
        dataset_key: str,
        subject: str,
        name: str,
        timings: dict[str, float],
        func: Callable[[], T],
    ) -> T:
        LOGGER.info(
            "runtime_step_started dataset=%s subject=%s step=%s",
            dataset_key,
            subject,
            name,
        )
        started_at = perf_counter()
        result = func()
        elapsed = perf_counter() - started_at
        timings[name] = elapsed
        LOGGER.info(
            "runtime_step_completed dataset=%s subject=%s step=%s elapsed_ms=%.3f result=%s",
            dataset_key,
            subject,
            name,
            elapsed * 1000.0,
            self._summarize_value(result),
        )
        return result

    def _timed_load(
        self,
        dataset_key: str,
        subject: str,
        name: str,
        timings: dict[str, float],
        func: Callable[[], T],
    ) -> T:
        LOGGER.info(
            "dataset_subject_load_started dataset=%s subject=%s step=%s",
            dataset_key,
            subject,
            name,
        )
        started_at = perf_counter()
        result = func()
        elapsed = perf_counter() - started_at
        timings[name] = elapsed
        LOGGER.info(
            "dataset_subject_load_completed dataset=%s subject=%s step=%s elapsed_ms=%.3f result=%s",
            dataset_key,
            subject,
            name,
            elapsed * 1000.0,
            self._summarize_value(result),
        )
        return result

    def _dataset_file_missing(self, dataset_key: str) -> bool:
        if dataset_key == BCICIV_DATASET_KEY:
            return bool(self._bciciv_missing_files())
        return bool(self._bnci_missing_files())

    def _bciciv_missing_files(self) -> list[str]:
        data_dir = Path(self.bciciv_config.data_dir)
        missing: list[str] = []
        for subject in self.dataset_definitions[BCICIV_DATASET_KEY].subjects:
            for suffix in ("cnt", "mrk", "nfo"):
                path = data_dir / f"BCICIV_calib_ds1{subject}_{suffix}.txt"
                if not path.is_file():
                    missing.append(str(path))
        return missing

    def _bnci_missing_files(self, subjects: tuple[str, ...] | None = None) -> list[str]:
        data_dir = Path(self.bnci_config.data_dir) / "gdf"
        subject_list = subjects or self.dataset_definitions[BNCI_DATASET_KEY].subjects
        missing: list[str] = []
        for subject in subject_list:
            for session in (1, 2, 3):
                path = data_dir / f"B{int(subject):02d}{session:02d}T.gdf"
                if not path.is_file():
                    missing.append(str(path))
        return missing

    def _log_dataset_file_status(self) -> None:
        for definition in self.dataset_definitions.values():
            missing = (
                self._bciciv_missing_files()
                if definition.key == BCICIV_DATASET_KEY
                else self._bnci_missing_files()
            )
            if missing:
                LOGGER.warning(
                    "dataset_files_missing dataset=%s missing_count=%d first_missing=%s",
                    definition.key,
                    len(missing),
                    missing[0],
                )
            else:
                LOGGER.info(
                    "dataset_files_available dataset=%s data_dir=%s subjects=%d",
                    definition.key,
                    self._config(definition.key).data_dir,
                    len(definition.subjects),
                )

    def _normalize_dataset(self, dataset: str | None) -> str:
        value = (dataset or DEFAULT_DATASET_KEY).strip().lower().replace(" ", "")
        key = DATASET_ALIASES.get(value, value)
        if key not in self.dataset_definitions:
            raise ValueError(
                f"dataset must be one of {tuple(self.dataset_definitions)}, got {dataset!r}"
            )
        return key

    def _normalize_subject(self, dataset_key: str, subject: str | None) -> str:
        definition = self.dataset_definitions[dataset_key]
        value = (subject or definition.default_subject).strip().lower()
        if dataset_key == BCICIV_DATASET_KEY:
            if value.startswith("ds1"):
                value = value[-1]
            if value not in definition.subjects:
                raise ValueError(f"subject must be one of {definition.subjects}, got {subject!r}")
            return value
        if value.startswith("b") and value[1:].isdigit():
            value = value[1:]
        try:
            normalized = str(int(value))
        except ValueError as exc:
            raise ValueError(
                f"subject must be one of {definition.subjects}, got {subject!r}"
            ) from exc
        if normalized not in definition.subjects:
            raise ValueError(f"subject must be one of {definition.subjects}, got {subject!r}")
        return normalized

    def _config(self, dataset_key: str):
        if dataset_key == BCICIV_DATASET_KEY:
            return self.bciciv_config
        return self.bnci_config

    def _experience_top_k(self, dataset_key: str) -> int:
        if dataset_key == BCICIV_DATASET_KEY:
            return int(self.bciciv_config.experience_top_k)
        return int(self.bnci_config.top_k)

    def _fixed_reject_threshold(self, dataset_key: str) -> float | None:
        if dataset_key == BCICIV_DATASET_KEY:
            return self.bciciv_config.fixed_reject_threshold
        return None

    def _seed(self, dataset_key: str, subject: str) -> int:
        if dataset_key == BCICIV_DATASET_KEY:
            return int(self.bciciv_config.seed)
        return int(self.bnci_config.seed + int(subject))

    @staticmethod
    def _trial_index(evaluation_indices: np.ndarray, evaluation_index: int) -> int:
        if not 0 <= evaluation_index < len(evaluation_indices):
            raise ValueError(
                f"evaluation_index must be in [0, {len(evaluation_indices) - 1}], got {evaluation_index}"
            )
        return int(evaluation_indices[evaluation_index])

    @staticmethod
    def _bciciv_global_labels(dataset) -> np.ndarray:
        mapping = {name: index for index, name in enumerate(BCICIV_GLOBAL_CLASS_NAMES)}
        return np.asarray(
            [mapping[dataset.class_names[int(label)]] for label in dataset.labels],
            dtype=int,
        )

    @staticmethod
    def _display_channel_names(
        dataset_key: str,
        channel_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        if dataset_key == BNCI_DATASET_KEY:
            return tuple(name.replace("EEG:", "") for name in channel_names)
        return tuple(channel_names)

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
    def _summarize_value(value: object) -> str:
        if isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        if isinstance(value, tuple):
            parts = ", ".join(LiveInferenceEngine._summarize_value(item) for item in value[:3])
            suffix = "" if len(value) <= 3 else f", ... len={len(value)}"
            return f"tuple({parts}{suffix})"
        if hasattr(value, "vector") and hasattr(value, "tensor"):
            return (
                f"{type(value).__name__}(vector_shape={value.vector.shape}, "
                f"tensor_shape={value.tensor.shape})"
            )
        if hasattr(value, "trials") and hasattr(value, "labels"):
            return (
                f"{type(value).__name__}(trials_shape={value.trials.shape}, "
                f"labels={len(value.labels)}, fs={getattr(value, 'fs', 'n/a')})"
            )
        if hasattr(value, "train_embeddings") and hasattr(value, "replay_embeddings"):
            return (
                f"{type(value).__name__}(train_embeddings={value.train_embeddings.shape}, "
                f"replay_embeddings={value.replay_embeddings.shape})"
            )
        if isinstance(value, (list, tuple)):
            return f"{type(value).__name__}(len={len(value)})"
        return type(value).__name__

    @staticmethod
    def _build_bciciv_config() -> FBCSPDesignConfig:
        configured_path = Path(
            os.getenv("DEMO_BCICIV_DATA_DIR", os.getenv("DEMO_DATA_DIR", "Dataset/BCICIV_1_asc"))
        )
        data_dir = configured_path if configured_path.is_absolute() else PROJECT_ROOT / configured_path
        return FBCSPDesignConfig(data_dir=data_dir)

    @staticmethod
    def _build_bnci_config() -> BNCI004PersonalizationConfig:
        configured_path = Path(os.getenv("DEMO_BNCI_DATA_DIR", "Dataset/BNCI2014_004"))
        data_dir = configured_path if configured_path.is_absolute() else PROJECT_ROOT / configured_path
        return BNCI004PersonalizationConfig(data_dir=data_dir)

    def _build_dataset_definitions(self) -> dict[str, DatasetDefinition]:
        return {
            BCICIV_DATASET_KEY: DatasetDefinition(
                key=BCICIV_DATASET_KEY,
                name="BCICIV_1_asc",
                label="数据集1 / BCICIV_1_asc",
                mode_label="数据集1：跨受试者个体化单窗口推理",
                subjects=tuple(str(subject) for subject in BCICIV_DEFAULT_SUBJECTS),
                default_subject=str(BCICIV_DEFAULT_SUBJECTS[0]),
                class_names=BCICIV_GLOBAL_CLASS_NAMES,
            ),
            BNCI_DATASET_KEY: DatasetDefinition(
                key=BNCI_DATASET_KEY,
                name="BNCI2014_004",
                label="数据集2 / BNCI2014_004",
                mode_label="数据集2：2b 历史会话到目标会话单窗口推理",
                subjects=tuple(str(subject) for subject in BNCI_DEFAULT_SUBJECTS),
                default_subject=str(BNCI_DEFAULT_SUBJECTS[0]),
                class_names=BNCI_CLASS_NAMES,
            ),
        }
