"""Run hybrid photonic replay on BCICIV_1_asc calibration recordings."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_photonic_mi_bci.datasets import (  # noqa: E402
    DEFAULT_BAND,
    DEFAULT_MOTOR_CHANNELS,
    DEFAULT_WINDOW,
    BCICIVFeatures,
    load_pooled_subject_features,
    load_subject_features,
)
from hybrid_photonic_mi_bci.experiment import (  # noqa: E402
    PipelineBuildConfig,
    build_pipeline_from_features,
    run_replay,
    warmup_selector,
)


SUBJECTS = tuple("abcdefg")


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.subject == "pooled":
        pooled_result = run_pooled(
            data_dir=Path(args.data_dir),
            config=config,
            band=(args.band_low, args.band_high),
            window=(args.window_start, args.window_end),
            channels=tuple(args.channels.split(",")),
            n_train_per_subject=args.n_train,
            warmup_trials_per_subject=args.warmup_trials,
        )
        print_subject_result(pooled_result, verbose=True)
        return

    subjects = SUBJECTS if args.subject == "all" else (args.subject,)
    rows = []
    for subject in subjects:
        result = run_subject(
            data_dir=Path(args.data_dir),
            subject=subject,
            config=config,
            band=(args.band_low, args.band_high),
            window=(args.window_start, args.window_end),
            channels=tuple(args.channels.split(",")),
            warmup_trials=args.warmup_trials,
        )
        rows.append(result)
        print_subject_result(result, verbose=args.subject != "all")

    if len(rows) > 1:
        command = np.array([row["metrics"]["command_accuracy"] for row in rows], dtype=float)
        balanced = np.array(
            [row["metrics"]["balanced_command_accuracy"] for row in rows],
            dtype=float,
        )
        reject = np.array([row["metrics"]["reject_rate"] for row in rows], dtype=float)
        print("\nBCICIV_1_asc summary")
        print(f"subjects: {''.join(subjects)}")
        print(f"selector: {config.selector_kind}")
        print(f"mean command accuracy: {command.mean():.3f} +/- {command.std():.3f}")
        print(f"mean balanced accuracy: {balanced.mean():.3f} +/- {balanced.std():.3f}")
        print(f"mean reject rate: {reject.mean():.3f} +/- {reject.std():.3f}")


def run_pooled(
    data_dir: Path,
    config: PipelineBuildConfig,
    band: tuple[float, float] = DEFAULT_BAND,
    window: tuple[float, float] = DEFAULT_WINDOW,
    channels: tuple[str, ...] = DEFAULT_MOTOR_CHANNELS,
    n_train_per_subject: int = 120,
    warmup_trials_per_subject: int = 120,
) -> dict[str, object]:
    dataset = load_pooled_subject_features(
        data_dir=data_dir,
        subjects=SUBJECTS,
        n_train_per_subject=n_train_per_subject,
        channels=channels,
        band=band,
        window=window,
    )
    pooled_config = replace(config, n_train=n_train_per_subject * len(SUBJECTS))
    pipeline = build_pipeline_from_features(
        features=dataset.features,
        labels=dataset.labels,
        class_names=dataset.class_names,
        config=pooled_config,
    )
    warmup_count = warmup_selector(
        pipeline=pipeline,
        features=dataset.features,
        labels=dataset.labels,
        n_trials=warmup_trials_per_subject * len(SUBJECTS),
    )
    metrics = run_replay(
        pipeline=pipeline,
        features=dataset.features,
        labels=dataset.labels,
        start_index=pooled_config.n_train,
    )
    return {
        "subject": dataset.subject,
        "dataset": dataset,
        "pipeline": pipeline,
        "metrics": metrics,
        "warmup_count": warmup_count,
        "config": pooled_config,
    }


def run_subject(
    data_dir: Path,
    subject: str,
    config: PipelineBuildConfig,
    band: tuple[float, float] = DEFAULT_BAND,
    window: tuple[float, float] = DEFAULT_WINDOW,
    channels: tuple[str, ...] = DEFAULT_MOTOR_CHANNELS,
    warmup_trials: int = 120,
) -> dict[str, object]:
    dataset = load_subject_features(
        data_dir=data_dir,
        subject=subject,
        channels=channels,
        band=band,
        window=window,
    )
    pipeline = build_pipeline_from_features(
        features=dataset.features,
        labels=dataset.labels,
        class_names=dataset.class_names,
        config=config,
    )
    warmup_count = warmup_selector(
        pipeline=pipeline,
        features=dataset.features,
        labels=dataset.labels,
        n_trials=warmup_trials,
    )
    metrics = run_replay(
        pipeline=pipeline,
        features=dataset.features,
        labels=dataset.labels,
        start_index=config.n_train,
    )
    return {
        "subject": subject,
        "dataset": dataset,
        "pipeline": pipeline,
        "metrics": metrics,
        "warmup_count": warmup_count,
        "config": config,
    }


def print_subject_result(result: dict[str, object], verbose: bool) -> None:
    dataset = result["dataset"]
    metrics = result["metrics"]
    pipeline = result["pipeline"]
    config = result["config"]
    if not isinstance(dataset, BCICIVFeatures):
        raise TypeError("dataset result is not BCICIVFeatures")
    if not isinstance(config, PipelineBuildConfig):
        raise TypeError("config result is not PipelineBuildConfig")

    print(f"\nBCICIV_1_asc {format_dataset_label(str(result['subject']))}")
    print(f"classes: {dataset.class_names}")
    print(f"system outputs: {(*dataset.class_names, 'reject')}")
    print(f"features: {dataset.features.shape[0]} trials x {dataset.features.shape[1]} dims")
    print(f"feature names: {dataset.feature_names}")
    print(f"band/window: {dataset.band} Hz, {dataset.window} s")
    print(f"selector: {config.selector_kind}")
    print(f"candidate library: {config.library_kind}")
    print(f"candidates scanned per trial: {len(pipeline.projection_library)}")
    print(f"warmup trials: {result['warmup_count']}")
    print(f"replay trials: {metrics['total']}")
    print(f"accepted accuracy: {metrics['accepted_accuracy']:.3f}")
    print(f"command accuracy with rejects counted as misses: {metrics['command_accuracy']:.3f}")
    print(f"balanced command accuracy: {metrics['balanced_command_accuracy']:.3f}")
    print(f"reject rate: {metrics['reject_rate']:.3f}")
    if verbose:
        print(f"prediction counts: {dict(Counter(metrics['predictions']))}")
        cols = "/".join((*dataset.class_names, "reject"))
        print(f"confusion rows=true, cols={cols}:")
        print(metrics["confusion"])
        if hasattr(pipeline.selector, "state"):
            print(
                "top learned candidates:",
                np.argsort(pipeline.selector.state.values)[-5:][::-1].tolist(),
            )


def format_dataset_label(subject: str) -> str:
    if subject.startswith("pooled_"):
        return f"pooled subjects {subject.removeprefix('pooled_')}"
    return f"subject ds1{subject}"


def build_config(args: argparse.Namespace) -> PipelineBuildConfig:
    config = PipelineBuildConfig(
        n_train=args.n_train,
        n_candidates=args.n_candidates,
        candidate_noise=args.candidate_noise,
        library_kind=args.library_kind,
        epsilon=args.epsilon,
        temperature=args.temperature,
        reject_threshold=args.reject_threshold,
        margin_threshold=args.margin_threshold,
        selector_kind=args.selector,
        confidence_weight=args.confidence_weight,
        selector_margin_weight=args.selector_margin_weight,
        rejected_penalty=args.rejected_penalty,
        fusion_value_weight=args.fusion_value_weight,
        prototype_kind=args.prototype_kind,
        seed=args.seed,
    )
    return replace(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="Dataset/BCICIV_1_asc")
    parser.add_argument("--subject", choices=SUBJECTS + ("all", "pooled"), default="pooled")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--warmup-trials", type=int, default=120)
    parser.add_argument("--channels", default=",".join(DEFAULT_MOTOR_CHANNELS))
    parser.add_argument("--band-low", type=float, default=DEFAULT_BAND[0])
    parser.add_argument("--band-high", type=float, default=DEFAULT_BAND[1])
    parser.add_argument("--window-start", type=float, default=DEFAULT_WINDOW[0])
    parser.add_argument("--window-end", type=float, default=DEFAULT_WINDOW[1])
    parser.add_argument("--n-candidates", type=int, default=32)
    parser.add_argument("--candidate-noise", type=float, default=0.02)
    parser.add_argument(
        "--library-kind",
        choices=("perturb", "bootstrap", "mixed"),
        default="perturb",
    )
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--reject-threshold", type=float, default=0.34)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--selector", choices=("bandit", "confidence", "fusion"), default="fusion")
    parser.add_argument("--confidence-weight", type=float, default=0.25)
    parser.add_argument("--selector-margin-weight", type=float, default=0.10)
    parser.add_argument("--rejected-penalty", type=float, default=0.50)
    parser.add_argument("--fusion-value-weight", type=float, default=0.75)
    parser.add_argument("--prototype-kind", choices=("shared", "candidate"), default="candidate")
    return parser.parse_args()


if __name__ == "__main__":
    main()
