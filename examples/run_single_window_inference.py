"""Run one full online inference pass from a BCICIV_1_asc EEG trial."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable, TypeVar

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.fbcsp_design_args import add_design_arguments, config_from_args  # noqa: E402
from hybrid_photonic_mi_bci.backends import (  # noqa: E402
    BitSlicedPhotonicConfig,
    BitSlicedPhotonicMatrixOpsBackend,
    NumpyMatrixOpsBackend,
    ScipySignalOpsBackend,
    get_matrix_ops_backend,
    use_matrix_ops_backend,
    use_signal_ops_backend,
)
from hybrid_photonic_mi_bci.datasets import load_pooled_subject_trials  # noqa: E402
from hybrid_photonic_mi_bci.experience import (  # noqa: E402
    ExperienceEntry,
    build_bootstrap_experience_library,
)
from hybrid_photonic_mi_bci.fbcsp import FilterBankCSP  # noqa: E402
from hybrid_photonic_mi_bci.linear_models import (  # noqa: E402
    FeatureStandardizer,
    LinearHead,
    ShrinkageLDA,
    select_fisher_features,
)
from hybrid_photonic_mi_bci.small_networks import (  # noqa: E402
    SmallMLPConfig,
    _forward_numpy,
    train_small_mlp,
)
from hybrid_photonic_mi_bci.workflows.common import make_replay_split  # noqa: E402
from pure_runtime import PurePhotonicScanRuntime  # noqa: E402


T = TypeVar("T")


@dataclass(frozen=True)
class PreparedRuntime:
    dataset: object
    split: object
    fbcsp: FilterBankCSP
    selected_indices: np.ndarray
    standardizer: FeatureStandardizer
    mlp_model: object
    runtime: PurePhotonicScanRuntime
    reject_threshold: float


def main() -> None:
    parser = add_design_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--evaluation-index",
        type=int,
        default=0,
        help="Evaluation-window index to infer, relative to the held-out online split.",
    )
    parser.add_argument(
        "--online-repeats",
        type=int,
        default=1,
        help="Repeat the same online window after one setup and report timing statistics.",
    )
    parser.add_argument(
        "--precision-validation-windows",
        type=int,
        default=0,
        help="Compare adaptive precision with fixed 8-bit forward on N held-out windows.",
    )
    args = parser.parse_args()
    if args.online_repeats <= 0:
        raise ValueError("--online-repeats must be positive")
    if args.precision_validation_windows < 0:
        raise ValueError("--precision-validation-windows must be non-negative")
    cfg = config_from_args(args)

    timings: dict[str, float] = {}
    print("[single-inference] setup backend=numpy/scipy (offline preparation)")
    with use_matrix_ops_backend(NumpyMatrixOpsBackend()), use_signal_ops_backend(
        ScipySignalOpsBackend()
    ):
        prepared = timed("setup_total", timings, lambda: prepare_runtime(cfg, timings))

    online_backend = get_matrix_ops_backend()
    print(
        "[single-inference] online linear backend="
        f"{getattr(online_backend, 'execution_backend', type(online_backend).__name__)}"
    )

    evaluation_count = len(prepared.split.evaluation_abs)
    if not 0 <= args.evaluation_index < evaluation_count:
        raise ValueError(
            f"--evaluation-index must be in [0, {evaluation_count - 1}], "
            f"got {args.evaluation_index}"
        )
    absolute_index = int(prepared.split.evaluation_abs[args.evaluation_index])
    true_index = int(prepared.dataset.labels[absolute_index])
    true_label = prepared.dataset.class_names[true_index]
    trial = prepared.dataset.trials[absolute_index : absolute_index + 1]

    online_runs: list[dict[str, object]] = []
    output = None
    for _repeat_index in range(args.online_repeats):
        run_timings: dict[str, float] = {}
        run_tile_counts: dict[str, int] = {}
        output = run_one_online_forward(
            prepared,
            trial,
            run_timings,
            run_tile_counts,
        )
        online_runs.append(
            {
                "timings": run_timings,
                "tile_counts": run_tile_counts,
                "predicted_label": output.decisions[0].label,
                "rejected": output.decisions[0].rejected,
            }
        )
    assert output is not None
    online_timings = dict(online_runs[-1]["timings"])
    online_tile_counts = dict(online_runs[-1]["tile_counts"])
    decision = output.decisions[0]
    probabilities = output.probabilities[0]

    print("\nSingle-window full inference")
    print(f"dataset: BCICIV_1_asc / {prepared.dataset.subject}")
    print(f"classes: {prepared.dataset.class_names} + ('reject',)")
    print(
        "sample: "
        f"evaluation_index={args.evaluation_index}, absolute_trial_index={absolute_index}"
    )
    print(f"true_label: {true_label}")
    print(
        "prediction: "
        f"label={decision.label}, rejected={decision.rejected}, "
        f"confidence={decision.confidence:.4f}, margin={decision.margin:.4f}"
    )
    print(
        "probabilities: "
        + ", ".join(
            f"{label}={float(probabilities[index]):.4f}"
            for index, label in enumerate(prepared.dataset.class_names)
        )
    )
    print(
        "runtime: "
        f"selected_candidates={len(prepared.runtime.selected_entries or ())}, "
        f"tile_count_per_window={output.tile_count_per_window}, "
        f"reject_threshold={prepared.reject_threshold:.4f}"
    )

    selected_ids = [
        entry.entry_id for entry in (prepared.runtime.selected_entries or ())
    ]
    print(f"selected_entries: {', '.join(selected_ids)}")

    print("\nSetup timings")
    for name, elapsed in timings.items():
        print(f"{name}: {elapsed * 1000.0:.2f} ms")

    print("\nSingle online forward timings")
    total_online = sum(online_timings.values())
    for name, elapsed in online_timings.items():
        print(
            f"{name}: {elapsed * 1000.0:.3f} ms, "
            f"bit_sliced_photonic_tiles={online_tile_counts.get(name, 0)}"
        )
    print(f"online_forward_total: {total_online * 1000.0:.3f} ms")
    print(
        "online_bit_sliced_photonic_tiles: "
        f"{sum(online_tile_counts.values())}"
    )
    if len(online_runs) > 1:
        repeat_totals = np.array(
            [sum(dict(run["timings"]).values()) for run in online_runs],
            dtype=np.float64,
        )
        repeat_labels = [str(run["predicted_label"]) for run in online_runs]
        print("\nRepeated online timing")
        print(f"repeats: {len(online_runs)}")
        print(f"first: {repeat_totals[0] * 1000.0:.3f} ms")
        print(f"median: {np.median(repeat_totals) * 1000.0:.3f} ms")
        print(f"p90: {np.percentile(repeat_totals, 90) * 1000.0:.3f} ms")
        print(f"last: {repeat_totals[-1] * 1000.0:.3f} ms")
        print(
            "prediction_agreement: "
            f"{sum(label == repeat_labels[0] for label in repeat_labels) / len(repeat_labels):.3f}"
        )
    precision_validation = None
    if args.precision_validation_windows > 0:
        precision_validation = validate_precision_windows(
            prepared,
            count=min(args.precision_validation_windows, evaluation_count),
        )
        print("\nAdaptive vs fixed-8-bit validation")
        for key, value in precision_validation["summary"].items():
            print(f"{key}: {value}")
    precision_summary = getattr(online_backend, "precision_summary", lambda: [])()
    if precision_summary:
        print("\nAdaptive precision monitor")
        print(
            "policy | bits | operations | monitored/calls | "
            "max_8bit_shadow_error | escalations | tiles"
        )
        for row in precision_summary:
            print(
                f"{row['policy']} | {row['current_bits']} | {row['operations']} | "
                f"{row['monitored_calls']}/{row['calls']} | "
                f"{row['max_8bit_shadow_error']:.5f} | {row['escalations']} | "
                f"{row['tile_evaluations']}"
            )
        precision_report = getattr(online_backend, "precision_report", lambda: [])()
        report_path = cfg.metrics_dir / (
            f"adaptive_precision_eval_{args.evaluation_index:04d}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "dataset": "BCICIV_1_asc",
                    "evaluation_index": args.evaluation_index,
                    "absolute_trial_index": absolute_index,
                    "true_label": true_label,
                    "predicted_label": decision.label,
                    "rejected": decision.rejected,
                    "confidence": decision.confidence,
                    "online_backend": getattr(
                        online_backend,
                        "execution_backend",
                        type(online_backend).__name__,
                    ),
                    "online_timings_seconds": online_timings,
                    "online_tile_counts": online_tile_counts,
                    "online_repeats": online_runs,
                    "precision_validation": precision_validation,
                    "precision_summary": precision_summary,
                    "precision_operations": precision_report,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"precision_report: {report_path}")


def validate_precision_windows(prepared: PreparedRuntime, *, count: int) -> dict[str, object]:
    adaptive_rows = []
    fixed_backend = BitSlicedPhotonicMatrixOpsBackend(
        config=BitSlicedPhotonicConfig(
            logical_input_bits=8,
            logical_weight_bits=8,
        ),
        use_gazelle_model=False,
        announce=False,
    )
    for index in range(count):
        absolute_index = int(prepared.split.evaluation_abs[index])
        true_index = int(prepared.dataset.labels[absolute_index])
        true_label = prepared.dataset.class_names[true_index]
        trial = prepared.dataset.trials[absolute_index : absolute_index + 1]

        adaptive_timings: dict[str, float] = {}
        adaptive_tiles: dict[str, int] = {}
        adaptive_output = run_one_online_forward(
            prepared,
            trial,
            adaptive_timings,
            adaptive_tiles,
        )
        with use_matrix_ops_backend(fixed_backend):
            fixed_timings: dict[str, float] = {}
            fixed_tiles: dict[str, int] = {}
            fixed_output = run_one_online_forward(
                prepared,
                trial,
                fixed_timings,
                fixed_tiles,
            )

        adaptive_decision = adaptive_output.decisions[0]
        fixed_decision = fixed_output.decisions[0]
        probability_error = float(
            np.linalg.norm(adaptive_output.probabilities[0] - fixed_output.probabilities[0])
        )
        adaptive_rows.append(
            {
                "evaluation_index": index,
                "absolute_trial_index": absolute_index,
                "true_label": true_label,
                "adaptive_label": adaptive_decision.label,
                "fixed_8bit_label": fixed_decision.label,
                "adaptive_rejected": adaptive_decision.rejected,
                "fixed_8bit_rejected": fixed_decision.rejected,
                "adaptive_correct": adaptive_decision.label == true_label,
                "fixed_8bit_correct": fixed_decision.label == true_label,
                "prediction_agreement": adaptive_decision.label == fixed_decision.label,
                "probability_l2_error": probability_error,
                "adaptive_seconds": sum(adaptive_timings.values()),
                "fixed_8bit_seconds": sum(fixed_timings.values()),
                "adaptive_tiles": sum(adaptive_tiles.values()),
                "fixed_8bit_tiles": sum(fixed_tiles.values()),
            }
        )
        current = len(adaptive_rows)
        agreement = np.mean([row["prediction_agreement"] for row in adaptive_rows])
        adaptive_accuracy = np.mean([row["adaptive_correct"] for row in adaptive_rows])
        print(
            f"[precision-validation] {current}/{count} "
            f"adaptive_acc={adaptive_accuracy:.3f} agreement={agreement:.3f}"
        )

    adaptive_seconds = np.array([row["adaptive_seconds"] for row in adaptive_rows])
    fixed_seconds = np.array([row["fixed_8bit_seconds"] for row in adaptive_rows])
    adaptive_tiles = np.array([row["adaptive_tiles"] for row in adaptive_rows], dtype=np.int64)
    fixed_tiles = np.array([row["fixed_8bit_tiles"] for row in adaptive_rows], dtype=np.int64)
    accepted_adaptive = [row for row in adaptive_rows if not row["adaptive_rejected"]]
    accepted_fixed = [row for row in adaptive_rows if not row["fixed_8bit_rejected"]]
    summary = {
        "windows": count,
        "adaptive_command_accuracy": float(np.mean([row["adaptive_correct"] for row in adaptive_rows])),
        "fixed_8bit_command_accuracy": float(np.mean([row["fixed_8bit_correct"] for row in adaptive_rows])),
        "prediction_agreement": float(np.mean([row["prediction_agreement"] for row in adaptive_rows])),
        "adaptive_reject_rate": float(np.mean([row["adaptive_rejected"] for row in adaptive_rows])),
        "fixed_8bit_reject_rate": float(np.mean([row["fixed_8bit_rejected"] for row in adaptive_rows])),
        "adaptive_accepted_accuracy": _accepted_accuracy(
            accepted_adaptive,
            key="adaptive_correct",
        ),
        "fixed_8bit_accepted_accuracy": _accepted_accuracy(
            accepted_fixed,
            key="fixed_8bit_correct",
        ),
        "mean_probability_l2_error": float(np.mean([row["probability_l2_error"] for row in adaptive_rows])),
        "adaptive_median_ms": float(np.median(adaptive_seconds) * 1000.0),
        "fixed_8bit_median_ms": float(np.median(fixed_seconds) * 1000.0),
        "adaptive_mean_tiles": float(np.mean(adaptive_tiles)),
        "fixed_8bit_mean_tiles": float(np.mean(fixed_tiles)),
        "mean_tile_reduction": float(1.0 - np.mean(adaptive_tiles) / np.mean(fixed_tiles)),
    }
    return {"summary": summary, "windows": adaptive_rows}


def _accepted_accuracy(rows: list[dict[str, object]], *, key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([bool(row[key]) for row in rows]))


def prepare_runtime(cfg, timings: dict[str, float]) -> PreparedRuntime:
    print("[single-inference] loading BCICIV_1_asc trials")
    dataset = timed(
        "load_dataset",
        timings,
        lambda: load_pooled_subject_trials(
            data_dir=cfg.data_dir,
            subjects=cfg.subjects,
            n_train_per_subject=cfg.n_train_per_subject,
            channels=cfg.channels,
            window=cfg.window,
        ),
    )
    split = make_replay_split(
        total_trials=len(dataset.labels),
        n_subjects=len(cfg.subjects),
        n_train_per_subject=cfg.n_train_per_subject,
        calibration_trials_per_subject=cfg.calibration_trials_per_subject,
    )
    train_trials = dataset.trials[split.train]
    replay_trials = dataset.trials[split.replay]
    train_labels = dataset.labels[split.train]

    print("[single-inference] fitting FBCSP on the train split")
    fbcsp = FilterBankCSP(
        bands=cfg.bands,
        n_components=cfg.csp_components,
        filter_order=cfg.filter_order,
        covariance_shrinkage=cfg.csp_shrinkage,
    )
    train_set = timed(
        "fit_transform_fbcsp_train",
        timings,
        lambda: fbcsp.fit_transform(
            train_trials,
            train_labels,
            fs=dataset.fs,
            class_names=dataset.class_names,
        ),
    )
    replay_set = timed(
        "transform_fbcsp_replay",
        timings,
        lambda: fbcsp.transform(replay_trials),
    )
    selected_indices = timed(
        "select_fbcsp_features",
        timings,
        lambda: select_fisher_features(
            train_set.vector,
            train_labels,
            n_classes=len(dataset.class_names),
            n_features=cfg.selected_features,
        ),
    )
    standardizer = FeatureStandardizer()
    train_features = timed(
        "standardize_train_features",
        timings,
        lambda: standardizer.fit_transform(train_set.vector[:, selected_indices]),
    )
    replay_features = timed(
        "standardize_replay_features",
        timings,
        lambda: standardizer.transform(replay_set.vector[:, selected_indices]),
    )

    print("[single-inference] training compact MLP embedding model")
    mlp = timed(
        "train_small_mlp",
        timings,
        lambda: train_small_mlp(
            train_features=train_features,
            train_labels=train_labels,
            replay_features=replay_features,
            n_classes=len(dataset.class_names),
            config=SmallMLPConfig(
                hidden_dim=cfg.mlp_hidden_dim,
                embedding_dim=cfg.mlp_embedding_dim,
                dropout=cfg.mlp_dropout,
                epochs=cfg.mlp_epochs,
                seed=cfg.seed,
            ),
        ),
    )

    print("[single-inference] building and calibrating experience runtime")
    entries = timed(
        "build_experience_library",
        timings,
        lambda: build_experience_entries(
            embeddings=mlp.train_embeddings,
            labels=train_labels,
            class_names=dataset.class_names,
            mlp_model=mlp.model,
            n_entries=cfg.experience_entries,
            sample_fraction=cfg.experience_sample_fraction,
            seed=cfg.seed,
        ),
    )
    runtime = PurePhotonicScanRuntime(
        entries=entries,
        class_names=dataset.class_names,
        top_k=cfg.experience_top_k,
        tile_shape=cfg.tile_shape,
        reject_threshold=0.0,
        margin_threshold=cfg.margin_threshold,
    )
    calibration_embeddings = mlp.replay_embeddings[split.calibration_replay]
    timed(
        "calibrate_experience_runtime",
        timings,
        lambda: runtime.calibrate(calibration_embeddings),
    )
    reject_threshold = timed(
        "calibrate_reject_threshold",
        timings,
        lambda: calibrate_reject_threshold(
            runtime,
            train_embeddings=mlp.train_embeddings,
            target_rate=cfg.reject_target_rate,
            fixed_threshold=cfg.fixed_reject_threshold,
        ),
    )
    runtime.reject_threshold = reject_threshold

    return PreparedRuntime(
        dataset=dataset,
        split=split,
        fbcsp=fbcsp,
        selected_indices=selected_indices,
        standardizer=standardizer,
        mlp_model=mlp.model,
        runtime=runtime,
        reject_threshold=reject_threshold,
    )


def run_one_online_forward(
    prepared: PreparedRuntime,
    trial: np.ndarray,
    timings: dict[str, float],
    tile_counts: dict[str, int] | None = None,
):
    counts = tile_counts if tile_counts is not None else {}
    raw_features = timed_photonic(
        "fbcsp_transform_one_window",
        timings,
        counts,
        lambda: prepared.fbcsp.transform(trial).vector[:, prepared.selected_indices],
    )
    features = timed_photonic(
        "standardize_one_window",
        timings,
        counts,
        lambda: prepared.standardizer.transform(raw_features),
    )
    _scores, embedding = timed_photonic(
        "small_mlp_forward_one_window",
        timings,
        counts,
        lambda: _forward_numpy(prepared.mlp_model, features),
    )
    output = timed_photonic(
        "pure_runtime_photonic_scan_one_window",
        timings,
        counts,
        lambda: prepared.runtime.predict(embedding),
    )
    counts["pure_runtime_photonic_scan_one_window"] = output.tile_count_per_window
    return output


def build_experience_entries(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: tuple[str, ...],
    mlp_model,
    n_entries: int,
    sample_fraction: float,
    seed: int,
) -> tuple[ExperienceEntry, ...]:
    classifier = mlp_model.classifier
    classifier_weights = classifier.weight.detach().cpu().numpy().astype(np.float64)
    classifier_bias = classifier.bias.detach().cpu().numpy().astype(np.float64)
    mlp_head = LinearHead(
        weights=classifier_weights,
        bias=classifier_bias,
        class_names=class_names,
    )
    mlp_scores = mlp_head.scores(embeddings)
    indices = np.arange(len(labels), dtype=int)
    mlp_entry = ExperienceEntry(
        entry_id="anchor_mlp_classifier",
        centroid=embeddings.mean(axis=0),
        head=mlp_head,
        source="small_network_classifier",
        train_indices=indices,
        train_accuracy=float((mlp_scores.argmax(axis=1) == labels).mean()),
    )

    lda = ShrinkageLDA(shrinkage=0.15).fit(
        embeddings,
        labels,
        class_names=class_names,
    )
    lda_scores = lda.scores(embeddings)
    lda_entry = ExperienceEntry(
        entry_id="anchor_embedding_lda",
        centroid=embeddings.mean(axis=0),
        head=lda.head,
        source="global_embedding_lda",
        train_indices=indices,
        train_accuracy=float((lda_scores.argmax(axis=1) == labels).mean()),
    )

    bootstrap = build_bootstrap_experience_library(
        embeddings=embeddings,
        labels=labels,
        class_names=class_names,
        n_entries=n_entries,
        sample_fraction=sample_fraction,
        seed=seed,
    )
    return (mlp_entry, lda_entry, *bootstrap)


def calibrate_reject_threshold(
    runtime: PurePhotonicScanRuntime,
    *,
    train_embeddings: np.ndarray,
    target_rate: float,
    fixed_threshold: float | None,
) -> float:
    if fixed_threshold is not None:
        return float(fixed_threshold)
    if target_rate <= 0.0:
        return 0.0
    outputs = runtime.predict(train_embeddings)
    confidence = outputs.probabilities.max(axis=1)
    clipped_target = min(max(float(target_rate), 0.0), 0.95)
    return float(np.quantile(confidence, clipped_target))


def timed(name: str, timings: dict[str, float], func: Callable[[], T]) -> T:
    start = perf_counter()
    result = func()
    timings[name] = perf_counter() - start
    return result


def timed_photonic(
    name: str,
    timings: dict[str, float],
    tile_counts: dict[str, int],
    func: Callable[[], T],
) -> T:
    backend = get_matrix_ops_backend()
    before = int(getattr(backend, "total_tile_count", 0))
    result = timed(name, timings, func)
    after = int(getattr(backend, "total_tile_count", before))
    tile_counts[name] = max(0, after - before)
    return result


if __name__ == "__main__":
    main()
