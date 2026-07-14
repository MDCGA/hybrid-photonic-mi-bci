"""Run all FBCSP design lines and save a comparison summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fbcsp_design_args import add_design_arguments, config_from_args, print_summary_rows  # noqa: E402
from hybrid_photonic_mi_bci.workflows.full_design import run_full_design_comparison  # noqa: E402


def main() -> None:
    parser = add_design_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the online evaluation progress bar for the photonic-scan line.",
    )
    args = parser.parse_args()
    result = run_full_design_comparison(
        config_from_args(args),
        save=True,
        show_progress=not args.no_progress,
    )
    print_summary_rows(result.summary_rows)
    print(f"\nSaved comparison to {Path(args.metrics_dir)}")


if __name__ == "__main__":
    main()
