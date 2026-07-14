import unittest

import numpy as np

from hybrid_photonic_mi_bci import (
    DecisionConfig,
    EpsilonGreedyBandit,
    HybridBCIPipeline,
    NumpyMVMBackend,
    NumpyMatrixOpsBackend,
    PipelineBuildConfig,
    ProbabilityFusionSelector,
    ProjectionLibrary,
    PrototypeDecisionHead,
    TiledMVMBackend,
    build_pipeline_from_features,
    run_replay,
    use_matrix_ops_backend,
    warmup_selector,
)


class PipelineTest(unittest.TestCase):
    def test_numpy_backend_scans_candidate_bank(self) -> None:
        weights = np.arange(32, dtype=np.float64).reshape(2, 2, 8)
        features = np.ones(8, dtype=np.float64)

        with use_matrix_ops_backend(NumpyMatrixOpsBackend()):
            result = NumpyMVMBackend().scan(weights, features)

        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result, weights @ features)

    def test_numpy_backend_supports_generic_candidate_matrices(self) -> None:
        weights = np.arange(3 * 5 * 17, dtype=np.float64).reshape(3, 5, 17) / 10.0
        features = np.linspace(-1.0, 1.0, 17, dtype=np.float64)

        with use_matrix_ops_backend(NumpyMatrixOpsBackend()):
            result = NumpyMVMBackend().scan(weights, features)

        self.assertEqual(result.shape, (3, 5))
        np.testing.assert_allclose(result, weights @ features)

    def test_tiled_backend_matches_numpy_for_larger_matrices(self) -> None:
        rng = np.random.default_rng(11)
        weights = rng.normal(size=(4, 5, 17))
        features = rng.normal(size=17)
        backend = TiledMVMBackend(tile_shape=(2, 8), matrix_backend=NumpyMatrixOpsBackend())

        with use_matrix_ops_backend(NumpyMatrixOpsBackend()):
            result = backend.scan(weights, features)

        self.assertEqual(result.shape, (4, 5))
        self.assertEqual(backend.last_tile_count, 4 * 3 * 3)
        self.assertEqual(backend.count_tiles(weights), 4 * 3 * 3)
        np.testing.assert_allclose(result, weights @ features)

    def test_prototype_decision_head_accepts_generic_projection_dimension(self) -> None:
        decision_head = PrototypeDecisionHead(
            prototypes=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            config=DecisionConfig(reject_threshold=0.34, margin_threshold=0.0),
        )

        decisions = decision_head.decide_all(np.array([[0.9, 0.1, 0.0]], dtype=np.float64))

        self.assertEqual(decisions[0].predicted_index, 0)
        self.assertFalse(decisions[0].rejected)

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

    def test_pipeline_can_use_tiled_backend_with_non_2x8_projection(self) -> None:
        base = np.zeros((3, 5), dtype=np.float64)
        base[0, 0] = 1.0
        base[1, 1] = 1.0
        base[2, 2] = 1.0
        library = ProjectionLibrary.random_around(base, n_candidates=2, noise_scale=0.0, seed=1)
        decision_head = PrototypeDecisionHead(
            prototypes=np.eye(3, dtype=np.float64),
            config=DecisionConfig(reject_threshold=0.34, margin_threshold=0.0),
        )
        pipeline = HybridBCIPipeline(
            projection_library=library,
            backend=TiledMVMBackend(tile_shape=(2, 8)),
            decision_head=decision_head,
            selector=EpsilonGreedyBandit(n_candidates=2, epsilon=0.0, seed=1),
        )

        output = pipeline.predict_window(np.array([1.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))

        self.assertEqual(output.projections.shape, (2, 3))
        self.assertEqual(output.predicted_index, 0)

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
