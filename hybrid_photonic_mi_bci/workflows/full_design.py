"""Run all FBCSP design lines and save a summary comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..compute_accounting import summarize_lines
from ..progress import ProgressLogger
from .common import FBCSPDesignConfig, prepare_fbcsp_data, save_json
from .experience_photonic_line import ExperiencePhotonicLineResult, run_experience_photonic_line
from .fbcsp_reference import FBCSPReferenceResult, run_fbcsp_reference
from .small_network_line import SmallNetworkLineResult, run_small_network_line


@dataclass
class FullDesignComparisonResult:
    reference: FBCSPReferenceResult
    small_network: SmallNetworkLineResult
    experience_photonic: ExperiencePhotonicLineResult
    summary_rows: list[dict[str, Any]]


def run_full_design_comparison(
    config: FBCSPDesignConfig | None = None,
    save: bool = True,
    show_progress: bool = False,
) -> FullDesignComparisonResult:
    """Run reference, embedding, and photonic-candidate-scan lines."""

    cfg = config or FBCSPDesignConfig()
    progress = ProgressLogger(
        "fbcsp_design_comparison",
        cfg.metrics_path / "run_progress.json" if save else None,
    )
    with progress.step("prepare pooled BCICIV FBCSP data", index=1, total=5):
        prepared = prepare_fbcsp_data(cfg)
    with progress.step("run FBCSP + shrinkage LDA", index=2, total=5):
        reference = run_fbcsp_reference(cfg, prepared=prepared, save=save)
    with progress.step("run FBCSP + small MLP embedding", index=3, total=5):
        small_network = run_small_network_line(cfg, prepared=prepared, save=save)
    with progress.step("run experience-library photonic scan", index=4, total=5):
        experience_photonic = run_experience_photonic_line(
            cfg,
            prepared=prepared,
            small_network=small_network,
            save=save,
            show_progress=show_progress,
        )
    summary_rows = [reference.summary, small_network.summary, experience_photonic.summary]
    if save:
        with progress.step("save comparison summaries", index=5, total=5):
            compute_accounting = summarize_lines(
                [
                    {
                        "line": reference.summary["line"],
                        "summary": reference.compute_summary,
                        "events": reference.compute_events,
                    },
                    {
                        "line": small_network.summary["line"],
                        "summary": small_network.compute_summary,
                        "events": small_network.compute_events,
                    },
                    {
                        "line": experience_photonic.summary["line"],
                        "summary": experience_photonic.compute_summary,
                        "events": experience_photonic.compute_events,
                    },
                ]
            )
            save_json(
                cfg.metrics_path / "summary.json",
                {
                    "dataset": "BCICIV_1_asc",
                    "design": "FBCSP mainline with compact embedding and experience-library candidate scan",
                    "rows": summary_rows,
                    "compute_accounting_file": "compute_accounting.json",
                    "progress_file": "run_progress.json",
                },
            )
            save_json(cfg.metrics_path / "compute_accounting.json", compute_accounting)
    progress.write()
    return FullDesignComparisonResult(
        reference=reference,
        small_network=small_network,
        experience_photonic=experience_photonic,
        summary_rows=summary_rows,
    )
