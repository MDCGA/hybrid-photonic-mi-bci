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
from .features import Standardizer
from .pipeline import HybridBCIPipeline, PipelineOutput
from .projection_library import ProjectionCandidate, ProjectionLibrary

__all__ = [
    "DecisionConfig",
    "ConfidenceSelector",
    "EpsilonGreedyBandit",
    "HybridBCIPipeline",
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
    "Standardizer",
    "TiledMVMBackend",
    "build_pipeline_from_features",
    "run_replay",
    "warmup_selector",
]
