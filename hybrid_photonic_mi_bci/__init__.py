"""Hybrid photonic MI-BCI simulation toolkit.

The package keeps the matrix-vector multiplication path behind a small backend
interface so the first software baseline can use NumPy while a future photonic
driver can plug into the same pipeline.
"""

from .backends import MVMBackend, NumpyMVMBackend, PhotonicMVMBackendStub, TiledMVMBackend
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
    "MVMBackend",
    "NumpyMVMBackend",
    "OnlineAdaptationState",
    "PhotonicMVMBackendStub",
    "PipelineBuildConfig",
    "PipelineOutput",
    "ProbabilityFusionSelector",
    "ProjectionCandidate",
    "ProjectionLibrary",
    "PrototypeDecisionHead",
    "ShrinkageLDA",
    "Standardizer",
    "TiledMVMBackend",
    "build_pipeline_from_features",
    "run_replay",
    "warmup_selector",
]
