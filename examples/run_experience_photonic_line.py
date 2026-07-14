"""Run the FBCSP + MLP embedding + experience-library photonic scan line."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fbcsp_design_args import add_design_arguments, config_from_args, print_summary_rows  # noqa: E402
from hybrid_photonic_mi_bci.workflows.experience_photonic_line import run_experience_photonic_line  # noqa: E402


def main() -> None:
    parser = add_design_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the online evaluation progress bar.",
    )
    args = parser.parse_args()
    result = run_experience_photonic_line(
        config_from_args(args),
        save=True,
        show_progress=not args.no_progress,
    )
    print_summary_rows([result.summary])
    print(f"\nSaved metrics to {Path(args.metrics_dir) / 'experience_photonic'}")


if __name__ == "__main__":
    main()
