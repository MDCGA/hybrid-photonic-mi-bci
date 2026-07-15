"""Hybrid photonic MI-BCI simulation toolkit.

The package keeps matrix products behind a backend interface. The default
MatrixOps backend uses monitored adaptive logical precision: CAR starts at
4-bit, SOS/FBCSP front-end operators start at 6-bit, and numerically sensitive
operators remain at 8-bit. Reduced-precision results are periodically checked
against an 8-bit digital shadow and promoted when needed. Every logical value is
executed through physical uint4/int4 slices and ``2 x 8`` hardware tiles. The
candidate-scan backend uses a single 4-bit pass by default.
"""

from .backends import (
    AdaptivePrecisionPhotonicMatrixOpsBackend,
    AdaptivePrecisionRule,
    BitSlicedPhotonicConfig,
    BitSlicedPhotonicMatrixOpsBackend,
    MVMBackend,
    MatrixOpsBackend,
    NumpyMVMBackend,
    NumpyMatrixOpsBackend,
    PhotonicMVMBackendStub,
    PhotonicMatrixOpsBackendStub,
    PhotonicSignalOpsBackendStub,
    PhotonicQuantizationConfig,
    QuantizedPhotonicMatrixOpsBackend,
    ScipySignalOpsBackend,
    SimulatedPhotonicMatrixOpsBackend,
    SimulatedPhotonicSignalOpsBackend,
    SignalOpsBackend,
    TiledMVMBackend,
    affine_transform,
    candidate_probability_fusion,
    common_average_reference,
    covariance_gram,
    csp_spatial_project,
    featurewise_affine,
    get_matrix_ops_backend,
    get_signal_ops_backend,
    linear_scores,
    matrix_einsum,
    matrix_multiply,
    pairwise_squared_distances,
    prototype_distances,
    set_matrix_ops_backend,
    set_signal_ops_backend,
    signal_sosfiltfilt,
    use_matrix_ops_backend,
    use_signal_ops_backend,
)
from .calibration import (
    ConfidenceSelector,
    EpsilonGreedyBandit,
    OnlineAdaptationState,
    ProbabilityFusionSelector,
)
from .decision import DecisionConfig, PrototypeDecisionHead
from .experiment import PipelineBuildConfig, build_pipeline_from_features, run_replay, warmup_selector
from .fbcsp import DEFAULT_FILTER_BANK, FBCSPFeatureSet, FilterBankCSP
from .features import Standardizer
from .linear_models import FeatureStandardizer, LinearHead, ShrinkageLDA
from .pipeline import HybridBCIPipeline, PipelineOutput
from .projection_library import ProjectionCandidate, ProjectionLibrary

__all__ = [
    "AdaptivePrecisionPhotonicMatrixOpsBackend",
    "AdaptivePrecisionRule",
    "BitSlicedPhotonicConfig",
    "BitSlicedPhotonicMatrixOpsBackend",
    "DecisionConfig",
    "DEFAULT_FILTER_BANK",
    "ConfidenceSelector",
    "EpsilonGreedyBandit",
    "FBCSPFeatureSet",
    "FeatureStandardizer",
    "FilterBankCSP",
    "HybridBCIPipeline",
    "LinearHead",
    "MatrixOpsBackend",
    "MVMBackend",
    "NumpyMVMBackend",
    "NumpyMatrixOpsBackend",
    "OnlineAdaptationState",
    "PhotonicMVMBackendStub",
    "PhotonicMatrixOpsBackendStub",
    "PhotonicSignalOpsBackendStub",
    "PhotonicQuantizationConfig",
    "QuantizedPhotonicMatrixOpsBackend",
    "ScipySignalOpsBackend",
    "SimulatedPhotonicMatrixOpsBackend",
    "SimulatedPhotonicSignalOpsBackend",
    "SignalOpsBackend",
    "PipelineBuildConfig",
    "PipelineOutput",
    "ProbabilityFusionSelector",
    "ProjectionCandidate",
    "ProjectionLibrary",
    "PrototypeDecisionHead",
    "ShrinkageLDA",
    "Standardizer",
    "TiledMVMBackend",
    "affine_transform",
    "build_pipeline_from_features",
    "candidate_probability_fusion",
    "common_average_reference",
    "covariance_gram",
    "csp_spatial_project",
    "featurewise_affine",
    "get_matrix_ops_backend",
    "get_signal_ops_backend",
    "linear_scores",
    "matrix_einsum",
    "matrix_multiply",
    "pairwise_squared_distances",
    "prototype_distances",
    "run_replay",
    "set_matrix_ops_backend",
    "set_signal_ops_backend",
    "signal_sosfiltfilt",
    "use_matrix_ops_backend",
    "use_signal_ops_backend",
    "warmup_selector",
]
