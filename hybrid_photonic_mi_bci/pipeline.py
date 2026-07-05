"""End-to-end hybrid photonic MI-BCI decision pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backends import MVMBackend
from .calibration import ConfidenceSelector, EpsilonGreedyBandit, ProbabilityFusionSelector
from .decision import CandidateDecision, PrototypeDecisionHead
from .features import Standardizer
from .projection_library import ProjectionLibrary


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PipelineOutput:
    """One EEG decision-window output."""

    features: FloatArray
    projections: FloatArray
    candidate_decisions: list[CandidateDecision]
    selected_candidate: int
    selected_label: str
    predicted_class: str | None
    predicted_index: int | None
    confidence: float
    margin: float
    rejected: bool


class HybridBCIPipeline:
    """Scan candidate calibrators and produce a digital MI command."""

    def __init__(
        self,
        projection_library: ProjectionLibrary,
        backend: MVMBackend,
        decision_head: PrototypeDecisionHead,
        selector: EpsilonGreedyBandit | ConfidenceSelector | ProbabilityFusionSelector | None = None,
        standardizer: Standardizer | None = None,
    ):
        self.projection_library = projection_library
        self.backend = backend
        self.decision_head = decision_head
        self.selector = selector or EpsilonGreedyBandit(
            n_candidates=len(projection_library),
            epsilon=0.0,
        )
        self.standardizer = standardizer

    def predict_window(self, features: ArrayLike) -> PipelineOutput:
        x = np.asarray(features, dtype=np.float64)
        expected_dim = self.projection_library.weights.shape[2]
        if x.shape != (expected_dim,):
            raise ValueError(f"features must have shape ({expected_dim},), got {x.shape}")
        if self.standardizer is not None:
            x = self.standardizer.transform(x)

        projections = self.backend.scan(self.projection_library.weights, x)
        decisions = self.decision_head.decide_all(projections)
        if hasattr(self.selector, "fuse"):
            selected_decision = self.selector.fuse(decisions)
            selected = selected_decision.candidate_index
        else:
            selected = self.selector.choose(decisions)
            selected_decision = decisions[selected]
        predicted_index = None if selected_decision.rejected else selected_decision.predicted_index
        predicted_class = (
            None
            if predicted_index is None
            else self.decision_head.config.class_names[predicted_index]
        )
        return PipelineOutput(
            features=x,
            projections=projections,
            candidate_decisions=decisions,
            selected_candidate=selected,
            selected_label=self.projection_library[selected].label,
            predicted_class=predicted_class,
            predicted_index=predicted_index,
            confidence=selected_decision.confidence,
            margin=selected_decision.margin,
            rejected=selected_decision.rejected,
        )

    def update_from_label(
        self,
        output: PipelineOutput,
        true_index: int,
        update_all_candidates: bool = True,
    ) -> float:
        """Update online selector from replay/feedback labels.

        Returns the scalar reward used for the update. Rejected decisions receive
        a small penalty; wrong high-confidence decisions receive a larger one.
        """

        if not 0 <= true_index < len(self.decision_head.config.class_names):
            raise ValueError("true_index out of range")
        rewards = np.array(
            [_decision_reward(decision, true_index) for decision in output.candidate_decisions],
            dtype=np.float64,
        )
        if update_all_candidates and hasattr(self.selector, "update_many"):
            self.selector.update_many(rewards)
        else:
            self.selector.update(output.selected_candidate, float(rewards[output.selected_candidate]))
        return float(rewards[output.selected_candidate])


def _decision_reward(decision: CandidateDecision, true_index: int) -> float:
    if decision.rejected:
        return -0.05
    if decision.predicted_index == true_index:
        return 1.0
    return -1.0 - decision.confidence
