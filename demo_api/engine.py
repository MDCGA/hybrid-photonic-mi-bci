"""Live single-window inference engine backed by the reference example."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from examples.run_single_window_inference import (
    PreparedRuntime,
    prepare_runtime,
    run_one_online_forward,
)
from hybrid_photonic_mi_bci.backends import (
    NumpyMatrixOpsBackend,
    ScipySignalOpsBackend,
    get_matrix_ops_backend,
    use_matrix_ops_backend,
    use_signal_ops_backend,
)
from hybrid_photonic_mi_bci.workflows import FBCSPDesignConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    title: str
    description: str
    timing_key: str | None = None


STAGES = (
    StageDefinition("window", "EEG window", "Read the selected real EEG evaluation window"),
    StageDefinition("fbcsp", "FBCSP feature extraction", "Filter bank, CSP projection, and feature selection", "fbcsp_transform_one_window"),
    StageDefinition("standardize", "Feature standardization", "Apply the train-fitted feature standardizer", "standardize_one_window"),
    StageDefinition("embedding", "Compact MLP encoding", "Run the trained compact MLP forward pass", "small_mlp_forward_one_window"),
    StageDefinition("photonic_scan", "Photonic experience scan", "Scan calibrated Top-K experience heads", "pure_runtime_photonic_scan_one_window"),
    StageDefinition("decision", "Fusion and decision", "Fuse real class probabilities and apply rejection"),
)


class LiveInferenceEngine:
    """Prepare once and run the exact single-window online interface on demand."""

    def __init__(self) -> None:
        self.prepared: PreparedRuntime | None = None
        self.setup_timings: dict[str, float] = {}
        self.config = self._build_config()
        self._inference_lock = Lock()

    def prepare(self) -> None:
        data_dir = Path(self.config.data_dir)
        required = data_dir / "BCICIV_calib_ds1a_cnt.txt"
        if not required.is_file():
            raise FileNotFoundError(
                f"BCICIV_1_asc is missing under {data_dir}. "
                "Run: python scripts/download_datasets.py --dataset bciciv-1-asc"
            )

        timings: dict[str, float] = {}
        LOGGER.info("runtime_prepare_started data_dir=%s", data_dir)
        with use_matrix_ops_backend(NumpyMatrixOpsBackend()), use_signal_ops_backend(
            ScipySignalOpsBackend()
        ):
            prepared = prepare_runtime(self.config, timings)
        self.prepared = prepared
        self.setup_timings = timings
        LOGGER.info(
            "runtime_prepare_completed evaluation_windows=%d setup_ms=%.3f",
            len(prepared.split.evaluation_abs),
            sum(timings.values()) * 1000.0,
        )

    def metadata(self) -> dict[str, object]:
        prepared = self._require_prepared()
        backend = get_matrix_ops_backend()
        return {
            "service": "hybrid-photonic-mi-bci-demo",
            "mode": "live_inference",
            "mode_label": "BCICIV single-window",
            "dataset": "BCICIV_1_asc",
            "subject": prepared.dataset.subject,
            "evaluation_count": len(prepared.split.evaluation_abs),
            "default_evaluation_index": 0,
            "class_names": [*prepared.dataset.class_names, "reject"],
            "channel_names": list(prepared.dataset.channel_names),
            "sample_rate_hz": float(prepared.dataset.fs),
            "window_seconds": [float(value) for value in prepared.dataset.window],
            "backend": getattr(
                backend,
                "execution_backend",
                type(backend).__name__,
            ),
            "setup_timings_ms": {
                name: round(elapsed * 1000.0, 3)
                for name, elapsed in self.setup_timings.items()
            },
            "stages": [self._stage_payload(stage) for stage in STAGES],
        }

    def window_info(self, evaluation_index: int) -> dict[str, object]:
        prepared = self._require_prepared()
        absolute_index = self._absolute_index(evaluation_index)
        true_index = int(prepared.dataset.labels[absolute_index])
        trial = prepared.dataset.trials[absolute_index]
        return {
            "evaluation_index": evaluation_index,
            "absolute_trial_index": absolute_index,
            "true_label": prepared.dataset.class_names[true_index],
            "signal": self._signal_payload(trial),
        }

    def infer(self, evaluation_index: int) -> dict[str, object]:
        prepared = self._require_prepared()
        absolute_index = self._absolute_index(evaluation_index)
        true_index = int(prepared.dataset.labels[absolute_index])
        trial = prepared.dataset.trials[absolute_index : absolute_index + 1]
        timings: dict[str, float] = {}
        tile_counts: dict[str, int] = {}

        with self._inference_lock:
            output = run_one_online_forward(prepared, trial, timings, tile_counts)

        decision = output.decisions[0]
        probabilities = output.probabilities[0]
        backend = get_matrix_ops_backend()
        precision_summary = getattr(backend, "precision_summary", lambda: [])()
        return {
            "evaluation_index": evaluation_index,
            "absolute_trial_index": absolute_index,
            "true_label": prepared.dataset.class_names[true_index],
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
            "online_total_ms": round(sum(timings.values()) * 1000.0, 3),
            "online_tile_counts": tile_counts,
            "tile_count": sum(tile_counts.values()),
            "runtime_tile_count_per_window": output.tile_count_per_window,
            "reject_threshold": prepared.reject_threshold,
            "selected_entries": [
                entry.entry_id for entry in (prepared.runtime.selected_entries or ())
            ],
            "backend": getattr(
                backend,
                "execution_backend",
                type(backend).__name__,
            ),
            "precision": [self._precision_payload(item) for item in precision_summary],
        }

    def _absolute_index(self, evaluation_index: int) -> int:
        prepared = self._require_prepared()
        evaluation_count = len(prepared.split.evaluation_abs)
        if not 0 <= evaluation_index < evaluation_count:
            raise ValueError(
                f"evaluation_index must be in [0, {evaluation_count - 1}], "
                f"got {evaluation_index}"
            )
        return int(prepared.split.evaluation_abs[evaluation_index])

    def _signal_payload(self, trial: np.ndarray) -> dict[str, object]:
        prepared = self._require_prepared()
        channels = [
            {"name": name, "values": trial[index].astype(float).tolist()}
            for index, name in enumerate(prepared.dataset.channel_names)
        ]
        return {
            "sample_rate_hz": float(prepared.dataset.fs),
            "duration_seconds": float(trial.shape[-1] / prepared.dataset.fs),
            "source": "BCICIV_1_asc",
            "channels": channels,
        }

    def _require_prepared(self) -> PreparedRuntime:
        if self.prepared is None:
            raise RuntimeError("live inference runtime has not been prepared")
        return self.prepared

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
