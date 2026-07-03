"""Online candidate selection/adaptation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .decision import CandidateDecision


FloatArray = NDArray[np.float64]


@dataclass
class OnlineAdaptationState:
    """Mutable state used by simple online calibration policies."""

    counts: FloatArray
    values: FloatArray

    @classmethod
    def zeros(cls, n_candidates: int) -> "OnlineAdaptationState":
        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        return cls(
            counts=np.zeros(n_candidates, dtype=np.float64),
            values=np.zeros(n_candidates, dtype=np.float64),
        )


class EpsilonGreedyBandit:
    """Small candidate selector for replay-style online calibration.

    The selector exploits the candidate with the highest running reward most of
    the time, while still occasionally exploring another candidate.
    """

    def __init__(
        self,
        n_candidates: int,
        epsilon: float = 0.05,
        seed: int | None = None,
        confidence_weight: float = 0.25,
        margin_weight: float = 0.10,
        rejected_penalty: float = 0.50,
    ):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if confidence_weight < 0 or margin_weight < 0 or rejected_penalty < 0:
            raise ValueError("selector weights and penalties must be non-negative")
        self.epsilon = epsilon
        self.confidence_weight = confidence_weight
        self.margin_weight = margin_weight
        self.rejected_penalty = rejected_penalty
        self.state = OnlineAdaptationState.zeros(n_candidates)
        self._rng = np.random.default_rng(seed)

    def choose(self, decisions: list[CandidateDecision]) -> int:
        if len(decisions) != len(self.state.values):
            raise ValueError("decisions length must match n_candidates")
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(len(decisions)))

        scores = self.state.values.copy()
        for i, decision in enumerate(decisions):
            scores[i] += self.confidence_weight * decision.confidence
            scores[i] += self.margin_weight * decision.margin
            if decision.rejected:
                scores[i] -= self.rejected_penalty
        return int(np.argmax(scores))

    def update(self, candidate_index: int, reward: float) -> None:
        if not 0 <= candidate_index < len(self.state.values):
            raise IndexError("candidate_index out of range")
        self.state.counts[candidate_index] += 1
        n = self.state.counts[candidate_index]
        old = self.state.values[candidate_index]
        self.state.values[candidate_index] = old + (reward - old) / n

    def update_many(self, rewards: FloatArray) -> None:
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        if rewards_arr.shape != self.state.values.shape:
            raise ValueError(
                f"rewards must have shape {self.state.values.shape}, got {rewards_arr.shape}"
            )
        for candidate_index, reward in enumerate(rewards_arr):
            self.update(candidate_index, float(reward))


class ConfidenceSelector:
    """Select the most confident currently valid candidate.

    This selector is a useful baseline for showing the value of scanning all
    candidate matrices inside one EEG decision window.
    """

    def choose(self, decisions: list[CandidateDecision]) -> int:
        if not decisions:
            raise ValueError("decisions must not be empty")
        non_rejected = [decision for decision in decisions if not decision.rejected]
        pool = non_rejected or decisions
        best = max(pool, key=lambda decision: (decision.confidence, decision.margin))
        return best.candidate_index

    def update(self, candidate_index: int, reward: float) -> None:
        _ = candidate_index, reward

    def update_many(self, rewards: FloatArray) -> None:
        _ = rewards


class ProbabilityFusionSelector:
    """Fuse candidate class probabilities into one decision.

    This selector uses the current window's candidate confidence/margin and the
    running online reward estimate to weight each candidate. It keeps the
    photonic value proposition visible: all candidate MVM outputs can contribute
    to the final command, instead of only selecting one winner.
    """

    def __init__(
        self,
        n_candidates: int,
        reject_threshold: float,
        margin_threshold: float,
        value_weight: float = 0.75,
        confidence_weight: float = 1.0,
        margin_weight: float = 0.50,
        rejected_penalty: float = 1.0,
    ):
        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        if reject_threshold < 0 or margin_threshold < 0:
            raise ValueError("reject and margin thresholds must be non-negative")
        if (
            value_weight < 0
            or confidence_weight < 0
            or margin_weight < 0
            or rejected_penalty < 0
        ):
            raise ValueError("fusion weights and penalties must be non-negative")
        self.reject_threshold = reject_threshold
        self.margin_threshold = margin_threshold
        self.value_weight = value_weight
        self.confidence_weight = confidence_weight
        self.margin_weight = margin_weight
        self.rejected_penalty = rejected_penalty
        self.state = OnlineAdaptationState.zeros(n_candidates)

    def choose(self, decisions: list[CandidateDecision]) -> int:
        return self.fuse(decisions).candidate_index

    def fuse(self, decisions: list[CandidateDecision]) -> CandidateDecision:
        if len(decisions) != len(self.state.values):
            raise ValueError("decisions length must match n_candidates")

        scores = self.state.values * self.value_weight
        for i, decision in enumerate(decisions):
            scores[i] += self.confidence_weight * decision.confidence
            scores[i] += self.margin_weight * decision.margin
            if decision.rejected:
                scores[i] -= self.rejected_penalty

        weights = _softmax_1d(scores)
        probabilities = np.sum(
            np.stack([decision.probabilities for decision in decisions], axis=0)
            * weights[:, None],
            axis=0,
        )
        order = np.argsort(probabilities)[::-1]
        predicted_index = int(order[0])
        confidence = float(probabilities[predicted_index])
        margin = float(probabilities[order[0]] - probabilities[order[1]])
        rejected = confidence < self.reject_threshold or margin < self.margin_threshold
        support = np.array(
            [weights[i] * decisions[i].probabilities[predicted_index] for i in range(len(decisions))],
            dtype=np.float64,
        )
        candidate_index = int(np.argmax(support))
        return CandidateDecision(
            candidate_index=candidate_index,
            probabilities=probabilities,
            predicted_index=predicted_index,
            confidence=confidence,
            margin=margin,
            rejected=rejected,
        )

    def update(self, candidate_index: int, reward: float) -> None:
        if not 0 <= candidate_index < len(self.state.values):
            raise IndexError("candidate_index out of range")
        self.state.counts[candidate_index] += 1
        n = self.state.counts[candidate_index]
        old = self.state.values[candidate_index]
        self.state.values[candidate_index] = old + (reward - old) / n

    def update_many(self, rewards: FloatArray) -> None:
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        if rewards_arr.shape != self.state.values.shape:
            raise ValueError(
                f"rewards must have shape {self.state.values.shape}, got {rewards_arr.shape}"
            )
        for candidate_index, reward in enumerate(rewards_arr):
            self.update(candidate_index, float(reward))


def _softmax_1d(scores: FloatArray) -> FloatArray:
    centered = scores - np.max(scores)
    exp = np.exp(centered)
    return exp / np.sum(exp)
