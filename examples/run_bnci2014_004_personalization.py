"""Run single-subject personalization tests on BNCI2014_004 / BCI IV 2b."""

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
        calibration_trials_per_class=tuple(int(item) for item in args.calibration.split(",") if item),
        experience_entries=args.experience_entries,
        top_k=args.top_k,
        seed=args.seed,
    )
    result = run_bnci2014_004_personalization(config, save=True)
    print("BNCI2014_004 personalization summary")
    print("k/class | before | cal-only | exp-after | gain-before | improved")
    for row in result["summary"]:
        print(
            f"{row['calibration_trials_per_class']:>7} | "
            f"{row['before_mean']:.3f} | "
            f"{row['calibration_only_mean']:.3f} | "
            f"{row['experience_after_mean']:.3f} | "
            f"{row['gain_vs_before_mean']:+.3f} | "
            f"{row['subjects_improved_vs_before']}/{row['subjects']}"
        )
    print(f"\nSaved metrics to {Path(args.metrics_dir)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="Dataset/BNCI2014_004")
    parser.add_argument("--metrics-dir", default="artifacts/metrics/bnci2014_004_personalization")
    parser.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--calibration", default="2,4,8,12,16")
    parser.add_argument("--experience-entries", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    main()
