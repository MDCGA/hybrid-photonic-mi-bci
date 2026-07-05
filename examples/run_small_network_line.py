"""Run the FBCSP + compact MLP embedding line."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fbcsp_design_args import add_design_arguments, config_from_args, print_summary_rows  # noqa: E402
from hybrid_photonic_mi_bci.workflows.small_network_line import run_small_network_line  # noqa: E402


def main() -> None:
    parser = add_design_arguments(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    result = run_small_network_line(config_from_args(args), save=True)
    print_summary_rows([result.summary])
    print(f"\nSaved metrics to {Path(args.metrics_dir) / 'small_network'}")


if __name__ == "__main__":
    main()
