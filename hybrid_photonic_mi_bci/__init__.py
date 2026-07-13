"""Hybrid photonic MI-BCI simulation toolkit.

The package keeps matrix products behind a backend interface. The default
MatrixOps backend is a simulated photonic handoff that uses NumPy internally for
deterministic local execution while preserving the future hardware contract.
"""

from .backends import (
    MVMBackend,
    MatrixOpsBackend,
    NumpyMVMBackend,
    NumpyMatrixOpsBackend,
    PhotonicMVMBackendStub,
    PhotonicMatrixOpsBackendStub,
    PhotonicSignalOpsBackendStub,
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
