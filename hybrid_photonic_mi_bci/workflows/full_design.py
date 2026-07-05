"""Run all FBCSP design lines and save a summary comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
) -> FullDesignComparisonResult:
    """Run reference, embedding, and photonic-candidate-scan lines."""

    cfg = config or FBCSPDesignConfig()
    prepared = prepare_fbcsp_data(cfg)
    reference = run_fbcsp_reference(cfg, prepared=prepared, save=save)
    small_network = run_small_network_line(cfg, prepared=prepared, save=save)
    experience_photonic = run_experience_photonic_line(
        cfg,
        prepared=prepared,
        small_network=small_network,
        save=save,
    )
    summary_rows = [reference.summary, small_network.summary, experience_photonic.summary]
    if save:
        save_json(
            cfg.metrics_path / "summary.json",
            {
                "dataset": "BCICIV_1_asc",
                "design": "FBCSP mainline with compact embedding and experience-library candidate scan",
                "rows": summary_rows,
            },
        )
    return FullDesignComparisonResult(
        reference=reference,
        small_network=small_network,
        experience_photonic=experience_photonic,
        summary_rows=summary_rows,
    )
