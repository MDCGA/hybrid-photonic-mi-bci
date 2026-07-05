"""FBCSP + compact neural embedding line."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..small_networks import SmallMLPConfig, train_small_mlp
from .common import (
    FBCSPDesignConfig,
    FBCSPPreparedData,
    calibrated_reject_threshold,
    evaluate_scores,
    prepare_fbcsp_data,
    save_json,
    save_npz,
    standardize_selected_features,
    summary_from_metrics,
)


FloatArray = NDArray[np.float64]


@dataclass
class SmallNetworkLineResult:
    prepared: FBCSPPreparedData
    train_features: FloatArray
    replay_features: FloatArray
    train_scores: FloatArray
    replay_scores: FloatArray
    train_embeddings: FloatArray
    replay_embeddings: FloatArray
    classifier_weights: FloatArray
    classifier_bias: FloatArray
    history: dict[str, list[float]]
    metrics: dict[str, Any]
    reject_threshold: float
    summary: dict[str, Any]


def run_small_network_line(
    config: FBCSPDesignConfig | None = None,
    prepared: FBCSPPreparedData | None = None,
    save: bool = True,
) -> SmallNetworkLineResult:
    """Train a compact MLP on selected FBCSP features and save embeddings."""

    cfg = config or FBCSPDesignConfig()
    data = prepared or prepare_fbcsp_data(cfg)
    _standardizer, train_features, replay_features = standardize_selected_features(data)
    mlp_result = train_small_mlp(
        train_features=train_features,
        train_labels=data.train_labels,
        replay_features=replay_features,
        n_classes=len(data.dataset.class_names),
        config=SmallMLPConfig(
            hidden_dim=cfg.mlp_hidden_dim,
            embedding_dim=cfg.mlp_embedding_dim,
            dropout=cfg.mlp_dropout,
            epochs=cfg.mlp_epochs,
            seed=cfg.seed,
        ),
    )
    classifier_weights = (
        mlp_result.model.classifier.weight.detach().cpu().numpy().astype(np.float64)
    )
    classifier_bias = (
        mlp_result.model.classifier.bias.detach().cpu().numpy().astype(np.float64)
    )
    reject_threshold = calibrated_reject_threshold(mlp_result.train_scores, cfg)
    metrics = evaluate_scores(
        mlp_result.replay_scores,
        data.replay_labels,
        class_names=data.dataset.class_names,
        reject_threshold=reject_threshold,
        margin_threshold=cfg.margin_threshold,
    )
    summary = summary_from_metrics(
        "FBCSP + small MLP embedding",
        metrics,
        extra={
            "raw_fbcsp_dim": int(data.train_features_raw.shape[1]),
            "selected_features": int(len(data.selected_indices)),
            "embedding_dim": int(mlp_result.train_embeddings.shape[1]),
            "mlp_epochs": int(cfg.mlp_epochs),
            "reject_threshold": reject_threshold,
            "calibration_trials": 0,
            "tile_evaluations_per_window": 0,
        },
    )
    result = SmallNetworkLineResult(
        prepared=data,
        train_features=train_features,
        replay_features=replay_features,
        train_scores=mlp_result.train_scores,
        replay_scores=mlp_result.replay_scores,
        train_embeddings=mlp_result.train_embeddings,
        replay_embeddings=mlp_result.replay_embeddings,
        classifier_weights=classifier_weights,
        classifier_bias=classifier_bias,
        history=mlp_result.history,
        metrics=metrics,
        reject_threshold=reject_threshold,
        summary=summary,
    )
    if save:
        save_small_network_result(result, cfg.metrics_path / "small_network")
    return result


def save_small_network_result(result: SmallNetworkLineResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = result.prepared
    save_json(
        output_dir / "summary.json",
        {
            "line": "FBCSP + small MLP embedding",
            "summary": result.summary,
            "history_final": {
                "loss": result.history["loss"][-1],
                "accuracy": result.history["accuracy"][-1],
            },
            "config": data.config,
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
        train_embeddings=result.train_embeddings,
        replay_embeddings=result.replay_embeddings,
        classifier_weights=result.classifier_weights,
        classifier_bias=result.classifier_bias,
        train_labels=data.train_labels,
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
        history_loss=np.asarray(result.history["loss"], dtype=np.float64),
        history_accuracy=np.asarray(result.history["accuracy"], dtype=np.float64),
    )
