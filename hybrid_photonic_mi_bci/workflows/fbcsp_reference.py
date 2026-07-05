"""Traditional reference line: FBCSP + selected features + shrinkage LDA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..linear_models import ShrinkageLDA
from .common import (
    FBCSPDesignConfig,
    FBCSPPreparedData,
    calibrated_reject_threshold,
    class_count_dict,
    evaluate_scores,
    feature_rank_table,
    prepare_fbcsp_data,
    save_json,
    save_npz,
    standardize_selected_features,
    summary_from_metrics,
)


FloatArray = NDArray[np.float64]


@dataclass
class FBCSPReferenceResult:
    prepared: FBCSPPreparedData
    train_features: FloatArray
    replay_features: FloatArray
    train_scores: FloatArray
    replay_scores: FloatArray
    metrics: dict[str, Any]
    reject_threshold: float
    summary: dict[str, Any]


def run_fbcsp_reference(
    config: FBCSPDesignConfig | None = None,
    prepared: FBCSPPreparedData | None = None,
    save: bool = True,
) -> FBCSPReferenceResult:
    """Run and optionally save the FBCSP + shrinkage LDA reference baseline."""

    cfg = config or FBCSPDesignConfig()
    data = prepared or prepare_fbcsp_data(cfg)
    _standardizer, train_features, replay_features = standardize_selected_features(data)
    lda = ShrinkageLDA(shrinkage=0.15).fit(
        train_features,
        data.train_labels,
        class_names=data.dataset.class_names,
    )
    train_scores = lda.scores(train_features)
    replay_scores = lda.scores(replay_features)
    reject_threshold = calibrated_reject_threshold(train_scores, cfg)
    metrics = evaluate_scores(
        replay_scores,
        data.replay_labels,
        class_names=data.dataset.class_names,
        reject_threshold=reject_threshold,
        margin_threshold=cfg.margin_threshold,
    )
    summary = summary_from_metrics(
        "FBCSP + shrinkage LDA",
        metrics,
        extra={
            "raw_fbcsp_dim": int(data.train_features_raw.shape[1]),
            "selected_features": int(len(data.selected_indices)),
            "reject_threshold": reject_threshold,
            "calibration_trials": 0,
            "tile_evaluations_per_window": 0,
        },
    )
    result = FBCSPReferenceResult(
        prepared=data,
        train_features=train_features,
        replay_features=replay_features,
        train_scores=train_scores,
        replay_scores=replay_scores,
        metrics=metrics,
        reject_threshold=reject_threshold,
        summary=summary,
    )
    if save:
        save_reference_result(result, cfg.metrics_path / "reference")
    return result


def save_reference_result(result: FBCSPReferenceResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = result.prepared
    save_json(
        output_dir / "summary.json",
        {
            "line": "FBCSP + shrinkage LDA",
            "summary": result.summary,
            "config": data.config,
            "dataset": {
                "name": "BCICIV_1_asc",
                "subjects": data.config.subjects,
                "classes": data.dataset.class_names,
                "channels": data.dataset.channel_names,
                "fs": data.dataset.fs,
                "window": data.dataset.window,
                "train_trials": int(len(data.split.train)),
                "replay_trials": int(len(data.split.replay)),
                "train_class_counts": class_count_dict(data.train_labels, data.dataset.class_names),
                "replay_class_counts": class_count_dict(data.replay_labels, data.dataset.class_names),
            },
            "feature_ranking": feature_rank_table(data, top_n=len(data.selected_indices)),
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
                }
            },
        },
    )
    save_npz(
        output_dir / "arrays.npz",
        train_features=result.train_features,
        replay_features=result.replay_features,
        train_scores=result.train_scores,
        replay_scores=result.replay_scores,
        replay_labels=data.replay_labels,
        probabilities=result.metrics["probabilities"],
        predicted=result.metrics["predicted"],
        rejected=result.metrics["rejected_mask"],
        confidence=result.metrics["confidence"],
        margin=result.metrics["margin"],
        correct_trace=result.metrics["correct_trace"],
        rolling_command_accuracy=result.metrics["rolling_command_accuracy"],
        rolling_reject_rate=result.metrics["rolling_reject_rate"],
        confusion=result.metrics["confusion"],
        selected_indices=data.selected_indices,
        selected_feature_names=np.asarray(data.selected_feature_names),
        train_fbcsp_tensor=data.train_tensor,
        replay_fbcsp_tensor=data.replay_tensor,
    )
