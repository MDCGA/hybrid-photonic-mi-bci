"""Evaluation helpers shared by the FBCSP design lines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


def softmax(scores: FloatArray) -> FloatArray:
    centered = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)


def decisions_from_scores(
    scores: FloatArray,
    class_names: tuple[str, ...],
    reject_threshold: float = 0.45,
    margin_threshold: float = 0.0,
) -> dict[str, object]:
    probabilities = softmax(np.asarray(scores, dtype=np.float64))
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    predicted = order[:, 0]
    confidence = probabilities[np.arange(len(probabilities)), predicted]
    margin = (
        probabilities[np.arange(len(probabilities)), order[:, 0]]
        - probabilities[np.arange(len(probabilities)), order[:, 1]]
    )
    rejected = (confidence < reject_threshold) | (margin < margin_threshold)
    labels = np.asarray([class_names[index] for index in predicted], dtype=object)
    labels[rejected] = "reject"
    return {
        "probabilities": probabilities,
        "predicted": predicted,
        "confidence": confidence,
        "margin": margin,
        "rejected": rejected,
        "labels": labels,
    }


def classification_metrics(
    y_true: IntArray,
    predicted: IntArray,
    rejected: NDArray[np.bool_] | None,
    n_classes: int,
) -> dict[str, object]:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(predicted, dtype=int)
    reject_mask = np.zeros_like(y, dtype=bool) if rejected is None else np.asarray(rejected, dtype=bool)
    confusion = np.zeros((n_classes, n_classes + 1), dtype=int)
    correct = 0
    rejected_count = int(reject_mask.sum())
    for true_index, pred_index, is_rejected in zip(y, pred, reject_mask):
        if is_rejected:
            confusion[int(true_index), n_classes] += 1
        else:
            confusion[int(true_index), int(pred_index)] += 1
            correct += int(true_index == pred_index)

    recalls = []
    for class_index in range(n_classes):
        total = confusion[class_index].sum()
        recalls.append(confusion[class_index, class_index] / total if total else 0.0)
    accepted = len(y) - rejected_count
    return {
        "total": int(len(y)),
        "correct": int(correct),
        "rejected": rejected_count,
        "accepted_accuracy": float(correct / accepted) if accepted else 0.0,
        "command_accuracy": float(correct / len(y)) if len(y) else 0.0,
        "balanced_command_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "reject_rate": float(rejected_count / len(y)) if len(y) else 0.0,
        "confusion": confusion,
        "per_class_recall": np.asarray(recalls, dtype=np.float64),
    }


def rolling_mean(values: FloatArray, window: int) -> FloatArray:
    values_arr = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values_arr, dtype=np.float64)
    for index in range(len(values_arr)):
        start = max(0, index + 1 - window)
        out[index] = values_arr[start : index + 1].mean()
    return out
