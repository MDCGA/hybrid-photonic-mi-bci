"""Forward linear-compute accounting for hybrid photonic MI-BCI workflows.

The main accounting ratio is forward-only: preprocessing, calibration forward
passes, and online inference are included; model fitting and training/backprop
are excluded from the denominator. Algorithm-path matrix products routed
through ``MatrixOpsBackend`` and forward signal-processing operations routed
   through ``SignalOpsBackend`` are treated as photonic compute because the active
   forward defaults use uint4/int4 bit-sliced photonic tiles and photonic signal
   state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


STAGES = ("preprocessing", "fit", "training", "calibration", "inference")
FORWARD_STAGES = ("preprocessing", "calibration", "inference")


@dataclass(frozen=True)
class LinearComputeEvent:
    """A single linear-compute event measured in MAC-equivalent operations."""

    name: str
    macs: int
    photonic: bool
    stage: str
    category: str
    implementation: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "macs": int(self.macs),
            "photonic": bool(self.photonic),
            "stage": self.stage,
            "category": self.category,
            "implementation": self.implementation,
            "details": self.details,
        }


class LinearComputeLedger:
    """Collect and summarize linear-compute events."""

    def __init__(self, events: Iterable[LinearComputeEvent] | None = None) -> None:
        self.events: list[LinearComputeEvent] = list(events or ())

    def add(
        self,
        name: str,
        macs: int,
        *,
        photonic: bool,
        stage: str,
        category: str,
        implementation: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if macs < 0:
            raise ValueError("MAC count must be non-negative")
        if macs == 0:
            return
        self.events.append(
            LinearComputeEvent(
                name=name,
                macs=int(macs),
                photonic=bool(photonic),
                stage=stage,
                category=category,
                implementation=implementation,
                details=dict(details or {}),
            )
        )

    def extend(self, events: Iterable[LinearComputeEvent]) -> None:
        self.events.extend(events)

    def to_events(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]

    def summary(self) -> dict[str, Any]:
        all_stage_total = _sum_macs(self.events)
        all_stage_photonic = _sum_macs(event for event in self.events if event.photonic)
        forward_events = [
            event for event in self.events if event.stage in FORWARD_STAGES
        ]
        total = _sum_macs(forward_events)
        photonic = _sum_macs(event for event in forward_events if event.photonic)
        by_stage = {
            stage: _summary_for(event for event in self.events if event.stage == stage)
            for stage in STAGES
        }
        extra_stages = sorted({event.stage for event in self.events if event.stage not in STAGES})
        for stage in extra_stages:
            by_stage[stage] = _summary_for(event for event in self.events if event.stage == stage)
        by_category = {
            category: _summary_for(event for event in self.events if event.category == category)
            for category in sorted({event.category for event in self.events})
        }
        inference = by_stage.get("inference", _empty_summary())
        offline_events = [event for event in self.events if event.stage != "inference"]
        offline = _summary_for(offline_events)
        excluded_fit_training = _summary_for(
            event for event in self.events if event.stage in {"fit", "training"}
        )
        return {
            "scope": "forward linear MAC-equivalent operations",
            "forward_stages": FORWARD_STAGES,
            "photonic_rule": (
                "Algorithm-path MatrixOpsBackend matrix products are counted "
                "as photonic because the active forward backend is "
                "AdaptivePrecisionPhotonicMatrixOpsBackend. CAR uses monitored "
                "4-bit logical precision, SOS/FBCSP front-end operations start at "
                "6-bit and promote to 8-bit when their sampled 8-bit-shadow error "
                "exceeds policy limits. All logical precisions are reconstructed "
                "from physical uint4/int4 calls, and every matrix is decomposed "
                "into 2 x 8 tiles. Forward signal operations routed through "
                "SignalOpsBackend are also counted as photonic: CAR is an explicit "
                "channel-mixing matrix and each SOS section is a 3 x 3 state "
                "transition executed through the same MatrixOps backend. "
                "This includes CAR, SOS band-pass filtering, FBCSP/LDA matrix "
                "products, feature-standardization affine maps, exported MLP "
                "inference Linear and LayerNorm affine layers including bias "
                "via augmented constant inputs, 4-bit quantized candidate-head "
                "tile scans, retrieval fusion, and distance cross terms. Fit and PyTorch training/backprop MACs are "
                "reported separately but excluded from the main ratio."
            ),
            "excluded": [
                "model fitting and training/backprop stages for the main forward-only ratio",
                "activation, softmax, thresholding, and non-dot-product distance arithmetic",
                "variance, logarithm, and other nonlinear feature operations",
                "matrix decompositions/inversions such as eigh and pinv",
                "visualization-only PCA/projection matrix products",
            ],
            "linear_macs_total": int(total),
            "linear_macs_photonic": int(photonic),
            "linear_macs_digital": int(total - photonic),
            "photonic_linear_share": _share(photonic, total),
            "linear_macs_all_stages_total": int(all_stage_total),
            "linear_macs_all_stages_photonic": int(all_stage_photonic),
            "linear_macs_all_stages_digital": int(all_stage_total - all_stage_photonic),
            "photonic_linear_share_all_stages": _share(
                all_stage_photonic,
                all_stage_total,
            ),
            "linear_macs_inference": int(inference["linear_macs_total"]),
            "photonic_linear_macs_inference": int(inference["linear_macs_photonic"]),
            "digital_linear_macs_inference": int(inference["linear_macs_digital"]),
            "photonic_linear_share_inference": inference["photonic_linear_share"],
            "linear_macs_fit_or_training_excluded": int(
                excluded_fit_training["linear_macs_total"]
            ),
            "photonic_linear_macs_fit_or_training_excluded": int(
                excluded_fit_training["linear_macs_photonic"]
            ),
            "digital_linear_macs_fit_or_training_excluded": int(
                excluded_fit_training["linear_macs_digital"]
            ),
            "linear_macs_offline_or_calibration": int(offline["linear_macs_total"]),
            "photonic_linear_macs_offline_or_calibration": int(
                offline["linear_macs_photonic"]
            ),
            "photonic_linear_share_offline_or_calibration": offline[
                "photonic_linear_share"
            ],
            "by_stage": by_stage,
            "by_category": by_category,
        }


def events_from_dicts(events: Iterable[dict[str, Any]]) -> list[LinearComputeEvent]:
    return [
        LinearComputeEvent(
            name=str(event["name"]),
            macs=int(event["macs"]),
            photonic=bool(event["photonic"]),
            stage=str(event["stage"]),
            category=str(event["category"]),
            implementation=str(event["implementation"]),
            details=dict(event.get("details", {})),
        )
        for event in events
    ]


def add_fbcsp_events(
    ledger: LinearComputeLedger,
    *,
    prefix: str,
    n_train: int,
    n_replay: int,
    n_bands: int,
    n_classes: int,
    n_filters: int,
    n_channels: int,
    n_samples: int,
    filter_order: int = 3,
) -> None:
    """Account for linear operations inside FBCSP."""

    add_fbcsp_transform_events(
        ledger,
        prefix=f"{prefix}: train",
        n_trials=n_train,
        n_bands=n_bands,
        n_classes=n_classes,
        n_filters=n_filters,
        n_channels=n_channels,
        n_samples=n_samples,
        filter_order=filter_order,
        stage="fit",
    )
    add_sosfiltfilt_event(
        ledger,
        name=f"{prefix}: train band-pass for CSP covariance",
        n_trials=n_train,
        n_bands=n_bands,
        n_channels=n_channels,
        n_samples=n_samples,
        filter_order=filter_order,
        stage="fit",
    )
    ledger.add(
        f"{prefix}: trial covariance centered @ centered.T",
        n_train * n_bands * n_channels * n_channels * n_samples,
        photonic=True,
        stage="fit",
        category="fbcsp_covariance",
        implementation="bit_sliced_photonic_tiled_matmul_uint4_int4",
        details={
            "trials": int(n_train),
            "bands": int(n_bands),
            "channels": int(n_channels),
            "samples": int(n_samples),
        },
    )
    add_fbcsp_transform_events(
        ledger,
        prefix=f"{prefix}: replay",
        n_trials=n_replay,
        n_bands=n_bands,
        n_classes=n_classes,
        n_filters=n_filters,
        n_channels=n_channels,
        n_samples=n_samples,
        filter_order=filter_order,
        stage="inference",
    )


def add_fbcsp_transform_events(
    ledger: LinearComputeLedger,
    *,
    prefix: str,
    n_trials: int,
    n_bands: int,
    n_classes: int,
    n_filters: int,
    n_channels: int,
    n_samples: int,
    filter_order: int = 3,
    stage: str,
) -> None:
    """Account for FBCSP band-pass filtering and CSP spatial projection."""

    add_sosfiltfilt_event(
        ledger,
        name=f"{prefix} band-pass for feature projection",
        n_trials=n_trials,
        n_bands=n_bands,
        n_channels=n_channels,
        n_samples=n_samples,
        filter_order=filter_order,
        stage=stage,
    )
    ledger.add(
        f"{prefix} CSP spatial projection einsum",
        int(n_trials) * int(n_bands) * int(n_classes) * int(n_filters) * int(n_channels) * int(n_samples),
        photonic=True,
        stage=stage,
        category="fbcsp_projection",
        implementation="bit_sliced_photonic_tiled_einsum_uint4_int4",
        details={
            "trials": int(n_trials),
            "bands": int(n_bands),
            "classes": int(n_classes),
            "filters_per_class": int(n_filters),
            "channels": int(n_channels),
            "samples": int(n_samples),
        },
    )


def add_car_event(
    ledger: LinearComputeLedger,
    *,
    prefix: str,
    n_samples: int,
    n_channels: int,
    stage: str = "preprocessing",
) -> None:
    """Account for common-average reference through SignalOpsBackend.

    The estimate follows the implementation style: per sample, form the channel
    mean and subtract it from each channel. This is counted as roughly three
    MAC-equivalent linear scalar operations per channel.
    """

    ledger.add(
        f"{prefix}: common-average reference",
        int(n_samples) * int(n_channels) * 3,
        photonic=True,
        stage=stage,
        category="car_reference",
        implementation="bit_sliced_photonic_car_matrix_uint4_int4",
        details={
            "samples": int(n_samples),
            "channels": int(n_channels),
            "mac_equivalent_per_channel": 3,
        },
    )


def add_sosfiltfilt_event(
    ledger: LinearComputeLedger,
    *,
    name: str,
    n_trials: int,
    n_bands: int,
    n_channels: int,
    n_samples: int,
    filter_order: int,
    stage: str,
    macs_per_section_sample: int = 5,
    passes: int = 2,
) -> None:
    """Account for Butterworth SOS forward/backward filtering via SignalOps."""

    sos_sections = int(filter_order)
    ledger.add(
        name,
        (
            int(n_trials)
            * int(n_bands)
            * int(n_channels)
            * int(n_samples)
            * sos_sections
            * int(macs_per_section_sample)
            * int(passes)
        ),
        photonic=True,
        stage=stage,
        category="bandpass_filter",
        implementation="bit_sliced_photonic_sos_state_space_uint4_int4",
        details={
            "trials": int(n_trials),
            "bands": int(n_bands),
            "channels": int(n_channels),
            "samples": int(n_samples),
            "filter_order": int(filter_order),
            "sos_sections": sos_sections,
            "macs_per_section_sample": int(macs_per_section_sample),
            "passes": int(passes),
            "note": "Ignores small edge-padding/transient overhead.",
        },
    )


def add_lda_fit_events(
    ledger: LinearComputeLedger,
    *,
    prefix: str,
    n_samples: int,
    n_features: int,
    n_classes: int,
    stage: str = "fit",
) -> None:
    """Account for explicit NumPy matrix products in ``ShrinkageLDA.fit``."""

    ledger.add(
        f"{prefix}: pooled covariance centered.T @ centered",
        n_samples * n_features * n_features,
        photonic=True,
        stage=stage,
        category="lda_fit_covariance",
        implementation="bit_sliced_photonic_tiled_matmul_uint4_int4",
        details={
            "samples": int(n_samples),
            "features": int(n_features),
            "classes": int(n_classes),
        },
    )
    ledger.add(
        f"{prefix}: LDA weights/bias means @ inv_cov",
        2 * n_classes * n_features * n_features,
        photonic=True,
        stage=stage,
        category="lda_fit_parameters",
        implementation="bit_sliced_photonic_tiled_matmul_uint4_int4",
        details={
            "classes": int(n_classes),
            "features": int(n_features),
            "note": "Current code evaluates means @ inv_cov once for weights and once for bias.",
        },
    )


def add_linear_scores_event(
    ledger: LinearComputeLedger,
    *,
    name: str,
    n_samples: int,
    n_features: int,
    n_outputs: int,
    stage: str,
    photonic: bool = True,
    implementation: str = "bit_sliced_photonic_augmented_matmul_uint4_int4",
    category: str = "linear_head_scores",
) -> None:
    ledger.add(
        name,
        n_samples * (n_features + 1) * n_outputs,
        photonic=photonic,
        stage=stage,
        category=category,
        implementation=implementation,
        details={
            "samples": int(n_samples),
            "features": int(n_features),
            "augmented_features": int(n_features) + 1,
            "outputs": int(n_outputs),
            "note": "Bias is counted as an augmented constant-one input channel.",
        },
    )


def add_feature_standardization_event(
    ledger: LinearComputeLedger,
    *,
    name: str,
    n_samples: int,
    n_features: int,
    stage: str = "preprocessing",
) -> None:
    """Account for per-feature standardization via an augmented affine map."""

    ledger.add(
        name,
        int(n_samples) * (int(n_features) + 1) * int(n_features),
        photonic=True,
        stage=stage,
        category="feature_standardization_affine",
        implementation="bit_sliced_photonic_augmented_matmul_uint4_int4",
        details={
            "samples": int(n_samples),
            "features": int(n_features),
            "augmented_features": int(n_features) + 1,
            "note": "Standardization is represented as a diagonal affine matrix.",
        },
    )


def add_candidate_scan_events(
    ledger: LinearComputeLedger,
    *,
    prefix: str,
    n_windows: int,
    n_candidates: int,
    n_features: int,
    n_classes: int,
    stage: str,
) -> None:
    """Account for candidate head scan plus probability fusion."""

    ledger.add(
        f"{prefix}: candidate heads A_i h + b_i",
        n_windows * n_candidates * (n_features + 1) * n_classes,
        photonic=True,
        stage=stage,
        category="candidate_head_scan",
        implementation="quantized_photonic_tiled_mvm_uint4",
        details={
            "windows": int(n_windows),
            "candidates": int(n_candidates),
            "features": int(n_features),
            "augmented_features": int(n_features) + 1,
            "classes": int(n_classes),
            "note": "Bias is scanned with an appended constant-one input.",
        },
    )
    ledger.add(
        f"{prefix}: retrieval-weight fusion einsum",
        n_windows * n_candidates * n_classes,
        photonic=True,
        stage=stage,
        category="experience_fusion",
        implementation="bit_sliced_photonic_tiled_einsum_uint4_int4",
        details={
            "windows": int(n_windows),
            "candidates": int(n_candidates),
            "classes": int(n_classes),
        },
    )


def add_centroid_retrieval_event(
    ledger: LinearComputeLedger,
    *,
    name: str,
    n_queries: int,
    n_centroids: int,
    n_features: int,
    stage: str,
) -> None:
    """Account for backend-routed query-to-centroid distance cross terms."""

    ledger.add(
        name,
        int(n_queries) * int(n_centroids) * int(n_features),
        photonic=True,
        stage=stage,
        category="experience_retrieval_distance",
        implementation="bit_sliced_photonic_distance_cross_term_uint4_int4",
        details={
            "queries": int(n_queries),
            "centroids": int(n_centroids),
            "features": int(n_features),
            "note": "Counts the dot-product cross term exposed by pairwise distance expansion.",
        },
    )


def add_mlp_training_event(
    ledger: LinearComputeLedger,
    *,
    prefix: str,
    n_samples: int,
    input_dim: int,
    hidden_dim: int,
    embedding_dim: int,
    n_classes: int,
    epochs: int,
    training_multiplier: int = 3,
) -> None:
    """Account for PyTorch linear/affine MACs in MLP training.

    The default multiplier approximates forward, activation-gradient input
    propagation, and weight-gradient products for each affine layer.
    """

    per_forward = mlp_forward_macs(
        n_samples=n_samples,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_classes=n_classes,
    )
    ledger.add(
        f"{prefix}: PyTorch MLP Linear layers train/backprop",
        epochs * training_multiplier * per_forward,
        photonic=False,
        stage="training",
        category="mlp_linear_training",
        implementation="torch_nn_linear",
        details={
            "samples": int(n_samples),
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "embedding_dim": int(embedding_dim),
            "classes": int(n_classes),
            "epochs": int(epochs),
            "training_multiplier": int(training_multiplier),
            "note": "Includes Linear layers and LayerNorm affine parameters; nonlinear normalization is excluded.",
        },
    )


def add_mlp_forward_event(
    ledger: LinearComputeLedger,
    *,
    name: str,
    n_samples: int,
    input_dim: int,
    hidden_dim: int,
    embedding_dim: int,
    n_classes: int,
    stage: str,
) -> None:
    ledger.add(
        name,
        mlp_forward_macs(
            n_samples=n_samples,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            n_classes=n_classes,
        ),
        photonic=True,
        stage=stage,
        category="mlp_linear_forward",
        implementation="bit_sliced_photonic_linear_uint4_int4",
        details={
            "samples": int(n_samples),
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "embedding_dim": int(embedding_dim),
            "classes": int(n_classes),
            "note": "Training uses PyTorch, but exported inference Linear/LayerNorm affine ops route through MatrixOps.",
        },
    )


def mlp_forward_macs(
    *,
    n_samples: int,
    input_dim: int,
    hidden_dim: int,
    embedding_dim: int,
    n_classes: int,
) -> int:
    return int(n_samples) * (
        (int(input_dim) + 1) * int(input_dim)
        + (int(input_dim) + 1) * int(hidden_dim)
        + (int(hidden_dim) + 1) * int(embedding_dim)
        + (int(embedding_dim) + 1) * int(n_classes)
    )


def compact_summary_fields(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten the fields most useful in experiment summary rows."""

    return {
        "linear_macs_total": int(summary["linear_macs_total"]),
        "linear_macs_photonic": int(summary["linear_macs_photonic"]),
        "linear_macs_digital": int(summary["linear_macs_digital"]),
        "photonic_linear_share": float(summary["photonic_linear_share"]),
        "linear_macs_inference": int(summary["linear_macs_inference"]),
        "photonic_linear_macs_inference": int(
            summary["photonic_linear_macs_inference"]
        ),
        "digital_linear_macs_inference": int(
            summary["digital_linear_macs_inference"]
        ),
        "photonic_linear_share_inference": float(
            summary["photonic_linear_share_inference"]
        ),
    }


def summarize_lines(lines: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(lines)
    return {
        "scope": "per-line linear-compute accounting",
        "lines": rows,
        "note": (
            "Shares are computed per workflow line. They are not summed across "
            "lines because each line is an alternative implementation."
        ),
    }


def _summary_for(events: Iterable[LinearComputeEvent]) -> dict[str, Any]:
    event_list = list(events)
    total = _sum_macs(event_list)
    photonic = _sum_macs(event for event in event_list if event.photonic)
    return {
        "linear_macs_total": int(total),
        "linear_macs_photonic": int(photonic),
        "linear_macs_digital": int(total - photonic),
        "photonic_linear_share": _share(photonic, total),
        "events": int(len(event_list)),
    }


def _empty_summary() -> dict[str, Any]:
    return _summary_for(())


def _sum_macs(events: Iterable[LinearComputeEvent]) -> int:
    return int(sum(int(event.macs) for event in events))


def _share(part: int, total: int) -> float:
    return float(part / total) if total else 0.0
