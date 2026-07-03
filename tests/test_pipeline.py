import unittest

import numpy as np

from hybrid_photonic_mi_bci import (
    DecisionConfig,
    EpsilonGreedyBandit,
    HybridBCIPipeline,
    NumpyMVMBackend,
    PipelineBuildConfig,
    ProbabilityFusionSelector,
    ProjectionLibrary,
    PrototypeDecisionHead,
    build_pipeline_from_features,
    run_replay,
    warmup_selector,
)


class PipelineTest(unittest.TestCase):
    def test_numpy_backend_scans_candidate_bank(self) -> None:
        weights = np.arange(32, dtype=np.float64).reshape(2, 2, 8)
        features = np.ones(8, dtype=np.float64)

        result = NumpyMVMBackend().scan(weights, features)

        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result, weights @ features)

    def test_pipeline_predicts_and_updates_selector(self) -> None:
        base = np.zeros((2, 8), dtype=np.float64)
        base[0, 0] = 1.0
        base[1, 1] = 1.0
        library = ProjectionLibrary.random_around(base, n_candidates=4, noise_scale=0.01, seed=1)
        decision_head = PrototypeDecisionHead(
            prototypes=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
            config=DecisionConfig(reject_threshold=0.34, margin_threshold=0.0),
        )
        pipeline = HybridBCIPipeline(
            projection_library=library,
            backend=NumpyMVMBackend(),
            decision_head=decision_head,
            selector=EpsilonGreedyBandit(n_candidates=4, epsilon=0.0, seed=1),
        )

        output = pipeline.predict_window(np.array([1.2, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float64))
        reward = pipeline.update_from_label(output, true_index=0)

        self.assertEqual(output.projections.shape, (4, 2))
        self.assertEqual(output.predicted_class, "left_hand")
        self.assertEqual(reward, 1.0)
        self.assertEqual(pipeline.selector.state.counts[output.selected_candidate], 1)
        self.assertEqual(pipeline.selector.state.counts.sum(), len(library))

    def test_probability_fusion_selector_predicts_from_all_candidates(self) -> None:
        base = np.zeros((2, 8), dtype=np.float64)
        base[0, 0] = 1.0
        base[1, 1] = 1.0
        library = ProjectionLibrary.random_around(base, n_candidates=4, noise_scale=0.02, seed=2)
        decision_head = PrototypeDecisionHead(
            prototypes=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
            config=DecisionConfig(reject_threshold=0.34, margin_threshold=0.0),
        )
        pipeline = HybridBCIPipeline(
            projection_library=library,
            backend=NumpyMVMBackend(),
            decision_head=decision_head,
            selector=ProbabilityFusionSelector(
                n_candidates=4,
                reject_threshold=0.34,
                margin_threshold=0.0,
            ),
        )

        output = pipeline.predict_window(np.array([1.2, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float64))
        reward = pipeline.update_from_label(output, true_index=0)

        self.assertEqual(output.predicted_class, "left_hand")
        self.assertEqual(reward, 1.0)
        self.assertEqual(pipeline.selector.state.counts.sum(), len(library))

    def test_generic_binary_feature_pipeline(self) -> None:
        rng = np.random.default_rng(3)
        labels = np.repeat(np.arange(2), 40)
        centers = np.array([[1.0, -0.4, 0.2, 0.1, 0, 0, 0, 0], [-1.0, 0.4, -0.2, -0.1, 0, 0, 0, 0]])
        features = centers[labels] + rng.normal(scale=0.2, size=(len(labels), 8))
        order = rng.permutation(len(labels))
        features = features[order]
        labels = labels[order]
        config = PipelineBuildConfig(
            n_train=40,
            n_candidates=4,
            reject_threshold=0.50,
            margin_threshold=0.0,
        )
        pipeline = build_pipeline_from_features(
            features=features,
            labels=labels,
            class_names=("left", "foot"),
            config=config,
        )

        warmup_selector(pipeline, features, labels, n_trials=40)
        metrics = run_replay(pipeline, features, labels, start_index=40)

        self.assertEqual(metrics["confusion"].shape, (2, 3))
        self.assertGreater(metrics["command_accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
