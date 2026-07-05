"""Generate all figures for the FBCSP-based design implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.fbcsp_design import plot_experience_photonic  # noqa: E402
from visualization.fbcsp_design import plot_reference  # noqa: E402
from visualization.fbcsp_design import plot_small_network  # noqa: E402
from visualization.fbcsp_design import plot_summary  # noqa: E402
from visualization.fbcsp_design import plot_system_diagram  # noqa: E402


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    output_root = Path(args.output_dir)
    formats = tuple(fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip())
    generated: list[Path] = []
    generated.extend(plot_system_diagram.plot(output_root / "system", formats))
    generated.extend(plot_reference.plot(metrics_dir, output_root / "reference", formats))
    generated.extend(plot_small_network.plot(metrics_dir, output_root / "small_network", formats))
    generated.extend(plot_experience_photonic.plot(metrics_dir, output_root / "experience_photonic", formats))
    generated.extend(plot_summary.plot(metrics_dir, output_root / "summary", formats))
    print("Generated FBCSP design figures:")
    for path in generated:
        print(f"- {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", default="artifacts/metrics/fbcsp_design")
    parser.add_argument("--output-dir", default="artifacts/figures/fbcsp_design")
    parser.add_argument("--formats", default="png,pdf")
    return parser.parse_args()


if __name__ == "__main__":
    main()
