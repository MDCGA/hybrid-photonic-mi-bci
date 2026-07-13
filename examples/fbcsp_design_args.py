"""Shared CLI helpers for FBCSP design-line scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_photonic_mi_bci.datasets import DEFAULT_MOTOR_CHANNELS, DEFAULT_SUBJECTS, DEFAULT_WINDOW
from hybrid_photonic_mi_bci.fbcsp import DEFAULT_FILTER_BANK
from hybrid_photonic_mi_bci.workflows import FBCSPDesignConfig


def add_design_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-dir", default="Dataset/BCICIV_1_asc")
    parser.add_argument("--metrics-dir", default="artifacts/metrics/fbcsp_design")
    parser.add_argument("--subjects", default="".join(DEFAULT_SUBJECTS))
    parser.add_argument("--channels", default=",".join(DEFAULT_MOTOR_CHANNELS))
    parser.add_argument("--window-start", type=float, default=DEFAULT_WINDOW[0])
    parser.add_argument("--window-end", type=float, default=DEFAULT_WINDOW[1])
    parser.add_argument("--n-train-per-subject", type=int, default=120)
    parser.add_argument("--calibration-trials-per-subject", type=int, default=6)
    parser.add_argument("--filter-order", type=int, default=3)
    parser.add_argument("--csp-components", type=int, default=2)
    parser.add_argument("--csp-shrinkage", type=float, default=0.10)
    parser.add_argument("--selected-features", type=int, default=32)
    parser.add_argument("--reject-target-rate", type=float, default=0.02)
    parser.add_argument("--fixed-reject-threshold", type=float, default=None)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--mlp-epochs", type=int, default=220)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-embedding-dim", type=int, default=32)
    parser.add_argument("--mlp-dropout", type=float, default=0.20)
    parser.add_argument("--experience-entries", type=int, default=64)
    parser.add_argument("--experience-top-k", type=int, default=8)
    parser.add_argument("--experience-sample-fraction", type=float, default=0.75)
    parser.add_argument("--experience-anchor-prior", type=float, default=5.0)
    parser.add_argument("--tile-rows", type=int, default=2)
    parser.add_argument("--tile-cols", type=int, default=8)
    return parser


def config_from_args(args: argparse.Namespace) -> FBCSPDesignConfig:
    return FBCSPDesignConfig(
        data_dir=Path(args.data_dir),
        metrics_dir=Path(args.metrics_dir),
        subjects=tuple(args.subjects),
        channels=tuple(channel.strip() for channel in args.channels.split(",") if channel.strip()),
        window=(args.window_start, args.window_end),
        bands=DEFAULT_FILTER_BANK,
        n_train_per_subject=args.n_train_per_subject,
        calibration_trials_per_subject=args.calibration_trials_per_subject,
        filter_order=args.filter_order,
        csp_components=args.csp_components,
        csp_shrinkage=args.csp_shrinkage,
        selected_features=args.selected_features,
        reject_target_rate=args.reject_target_rate,
        fixed_reject_threshold=args.fixed_reject_threshold,
        margin_threshold=args.margin_threshold,
        seed=args.seed,
        mlp_epochs=args.mlp_epochs,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_embedding_dim=args.mlp_embedding_dim,
        mlp_dropout=args.mlp_dropout,
        experience_entries=args.experience_entries,
        experience_top_k=args.experience_top_k,
        experience_sample_fraction=args.experience_sample_fraction,
        experience_anchor_prior=args.experience_anchor_prior,
        tile_shape=(args.tile_rows, args.tile_cols),
    )


def print_summary_rows(rows: list[dict[str, object]]) -> None:
    print("\nFBCSP design comparison")
    print(
        "line | total | command | balanced | accepted | reject | tiles/window | "
        "photonic share | inference share"
    )
    for row in rows:
        print(
            f"{row['line']} | {row['total']} | "
            f"{row['command_accuracy']:.3f} | "
            f"{row['balanced_command_accuracy']:.3f} | "
            f"{row['accepted_accuracy']:.3f} | "
            f"{row['reject_rate']:.3f} | "
            f"{row.get('tile_evaluations_per_window', 0)} | "
            f"{row.get('photonic_linear_share', 0.0):.3f} | "
            f"{row.get('photonic_linear_share_inference', 0.0):.3f}"
        )
