"""Run the three FBCSP design lines on BNCI2014_004 / BCI IV 2b."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_photonic_mi_bci.workflows.bnci2014_004_personalization import (  # noqa: E402
    BNCI004PersonalizationConfig,
    run_bnci2014_004_personalization,
)


def main() -> None:
    args = parse_args()
    config = BNCI004PersonalizationConfig(
        data_dir=Path(args.data_dir),
        metrics_dir=Path(args.metrics_dir),
        subjects=tuple(int(item) for item in args.subjects.split(",") if item),
        calibration_trials_per_class=args.calibration_trials_per_class,
        filter_order=args.filter_order,
        experience_entries=args.experience_entries,
        top_k=args.top_k,
        seed=args.seed,
    )
    result = run_bnci2014_004_personalization(config, save=True)
    print("BNCI2014_004 three-line design comparison")
    print("line | subjects | windows | command | balanced | reject | photonic | inference")
    for row in result["summary"]:
        print(
            f"{row['line']} | "
            f"{row['subjects']} | "
            f"{row['total']} | "
            f"{row['command_accuracy']:.3f} | "
            f"{row['balanced_command_accuracy']:.3f} | "
            f"{row['reject_rate']:.3f} | "
            f"{row['photonic_linear_share']:.3f} | "
            f"{row['photonic_linear_share_inference']:.3f}"
        )
    print(f"\nSaved metrics to {Path(args.metrics_dir)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="Dataset/BNCI2014_004")
    parser.add_argument("--metrics-dir", default="artifacts/metrics/bnci2014_004_personalization")
    parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--calibration-trials-per-class", type=int, default=8)
    parser.add_argument("--filter-order", type=int, default=3)
    parser.add_argument("--experience-entries", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    main()
