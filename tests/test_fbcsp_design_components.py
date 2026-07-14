import unittest

import numpy as np

from hybrid_photonic_mi_bci.backends import (
    BitSlicedPhotonicMatrixOpsBackend,
    MatrixOpsBackend,
    QuantizedPhotonicMatrixOpsBackend,
    SignalOpsBackend,
    SimulatedPhotonicMatrixOpsBackend,
    SimulatedPhotonicSignalOpsBackend,
    get_matrix_ops_backend,
    get_signal_ops_backend,
    signal_sosfiltfilt,
    use_matrix_ops_backend,
    use_signal_ops_backend,
)
from hybrid_photonic_mi_bci.backends import TiledMVMBackend
from hybrid_photonic_mi_bci.datasets.bnci2014_004 import calibration_eval_split
from hybrid_photonic_mi_bci.experience import (
    build_bootstrap_experience_library,
    retrieve_top_k,
    scan_experience_heads,
)
from hybrid_photonic_mi_bci.fbcsp import FilterBankCSP
from hybrid_photonic_mi_bci.linear_models import FeatureStandardizer, ShrinkageLDA
from hybrid_photonic_mi_bci.decision import DecisionConfig, PrototypeDecisionHead
from hybrid_photonic_mi_bci.workflows.common import make_replay_split


class FBCSPDesignComponentsTest(unittest.TestCase):
    def test_default_matrix_ops_backend_is_photonic_handoff(self) -> None:
        self.assertIsInstance(get_matrix_ops_backend(), BitSlicedPhotonicMatrixOpsBackend)
        self.assertIsInstance(get_signal_ops_backend(), SimulatedPhotonicSignalOpsBackend)

    def test_simulated_photonic_backend_tiles_and_reconstructs_large_matmul(self) -> None:
        backend = SimulatedPhotonicMatrixOpsBackend(tile_shape=(2, 8))
        left = np.arange(33, dtype=np.float64).reshape(3, 11) / 10.0
        right = np.arange(55, dtype=np.float64).reshape(11, 5) / 20.0

        result = backend.matmul(left, right, name="test_tiled_matmul")

        np.testing.assert_allclose(result, left @ right, atol=1e-12)
        self.assertEqual(backend.last_tile_count, 3 * 3 * 2)

    def test_bit_sliced_backend_reconstructs_eight_bit_logical_matmul(self) -> None:
        rng = np.random.default_rng(41)
        left = rng.normal(size=(3, 11))
        right = rng.normal(size=(11, 5))
        backend = BitSlicedPhotonicMatrixOpsBackend(
            use_gazelle_model=False,
            announce=False,
        )

        result = backend.matmul(left, right, name="test_bit_sliced_matmul")
        reference = left @ right

        np.testing.assert_allclose(result, reference, atol=0.12, rtol=0.08)
        base_tile_count = 3 * 3 * 2
        self.assertGreaterEqual(backend.last_slice_pair_count, 2)
        self.assertEqual(backend.last_tile_count, base_tile_count * backend.last_slice_pair_count)

    def test_tiled_mvm_backend_defaults_to_4bit_quantized_tiles(self) -> None:
        backend = TiledMVMBackend(tile_shape=(2, 8))

        self.assertIsInstance(backend.matrix_backend, QuantizedPhotonicMatrixOpsBackend)
        self.assertEqual(backend.matrix_backend.config.bit_width, 4)
        self.assertEqual(backend.matrix_backend.config.qinmin, 0)
        self.assertEqual(backend.matrix_backend.config.qinmax, 15)
        self.assertEqual(backend.matrix_backend.config.qwtmin, -8)
        self.assertEqual(backend.matrix_backend.config.qwtmax, 7)

    def test_quantized_photonic_backend_dequantizes_simple_identity(self) -> None:
        backend = QuantizedPhotonicMatrixOpsBackend(use_gazelle_model=False)
        features = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64)
        weights = np.eye(4, dtype=np.float64)

        result = backend.matmul(features, weights, name="test_identity")

        np.testing.assert_allclose(result, features, atol=0.25)

    def test_tiled_quantized_scan_keeps_features_unsigned_and_weights_signed(self) -> None:
        backend = TiledMVMBackend(
            tile_shape=(2, 8),
            matrix_backend=QuantizedPhotonicMatrixOpsBackend(
                use_gazelle_model=False,
                announce=False,
            ),
        )
        features = np.array([0.25, 0.5, 1.0, 0.75], dtype=np.float64)
        weights = np.array(
            [[[-1.0, 0.5, -0.25, 0.75], [0.5, -1.0, 0.75, -0.25]]],
            dtype=np.float64,
        )

        result = backend.scan(weights, features)

        np.testing.assert_allclose(result, weights @ features, atol=0.20, rtol=0.15)

    def test_photonic_sos_state_space_matches_scipy_with_float_tiles(self) -> None:
        from scipy.signal import butter, sosfiltfilt

        rng = np.random.default_rng(43)
        samples = rng.normal(size=(2, 128))
        sos = butter(3, (8.0, 20.0), btype="bandpass", fs=100.0, output="sos")
        expected = sosfiltfilt(sos, samples, axis=1)

        with use_matrix_ops_backend(SimulatedPhotonicMatrixOpsBackend()):
            result = signal_sosfiltfilt(
                sos,
                samples,
                axis=1,
                name="test_photonic_sos",
            )

        np.testing.assert_allclose(result, expected, atol=1e-11, rtol=1e-11)

    def test_fbcsp_returns_expected_multiclass_filter_bank_shape(self) -> None:
        rng = np.random.default_rng(5)
        fs = 100.0
        n_samples = 180
        t = np.arange(n_samples) / fs
        labels = np.repeat(np.arange(3), 12)
        trials = []
        for label in labels:
            trial = rng.normal(scale=0.15, size=(4, n_samples))
            trial[label] += np.sin(2 * np.pi * (10 + 4 * label) * t)
            trials.append(trial)
        trials = np.asarray(trials, dtype=np.float64)

        fbcsp = FilterBankCSP(
            bands=((8.0, 12.0), (12.0, 16.0), (16.0, 20.0)),
            n_components=1,
            covariance_shrinkage=0.10,
        )
        features = fbcsp.fit_transform(trials, labels, fs, ("left", "right", "foot"))

        self.assertEqual(features.tensor.shape, (36, 3, 3, 2))
        self.assertEqual(features.vector.shape, (36, 18))
        self.assertEqual(len(features.feature_names), 18)

    def test_shrinkage_lda_classifies_simple_standardized_features(self) -> None:
        rng = np.random.default_rng(7)
        labels = np.repeat(np.arange(3), 25)
        centers = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, -1.5]])
        features = centers[labels] + rng.normal(scale=0.25, size=(len(labels), 2))
        standardizer = FeatureStandardizer()
        standardized = standardizer.fit_transform(features)

        lda = ShrinkageLDA(shrinkage=0.20).fit(
            standardized,
            labels,
            class_names=("left", "right", "foot"),
        )

        accuracy = (lda.predict(standardized) == labels).mean()
        self.assertGreater(accuracy, 0.95)

    def test_experience_scan_uses_tiled_backend_shape_and_tile_count(self) -> None:
        rng = np.random.default_rng(11)
        labels = np.repeat(np.arange(3), 20)
        centers = np.eye(3, 6)
        embeddings = centers[labels] + rng.normal(scale=0.05, size=(len(labels), 6))
        entries = build_bootstrap_experience_library(
            embeddings,
            labels,
            class_names=("left", "right", "foot"),
            n_entries=4,
            seed=2,
        )
        selected, weights, _distances = retrieve_top_k(entries, embeddings[:9], k=2)
        backend = TiledMVMBackend(tile_shape=(2, 8))

        scan = scan_experience_heads(selected, weights, embeddings[:5], backend=backend)

        self.assertEqual(scan.candidate_scores.shape, (5, 2, 3))
        self.assertEqual(scan.fused_scores.shape, (5, 3))
        self.assertEqual(scan.tile_count_per_window, 2 * 2 * 1)

    def test_pooled_replay_split_reserves_calibration_per_subject(self) -> None:
        split = make_replay_split(
            total_trials=1400,
            n_subjects=7,
            n_train_per_subject=120,
            calibration_trials_per_subject=6,
        )

        self.assertEqual(len(split.train), 840)
        self.assertEqual(len(split.replay), 560)
        self.assertEqual(len(split.calibration_replay), 42)
        self.assertEqual(len(split.evaluation_replay), 518)
        self.assertEqual(split.replay_per_subject, 80)

    def test_bnci004_calibration_split_is_class_balanced(self) -> None:
        labels = np.array([0, 1] * 10, dtype=int)

        calibration, evaluation = calibration_eval_split(labels, trials_per_class=3)

        self.assertEqual(len(calibration), 6)
        self.assertEqual(len(evaluation), 14)
        self.assertEqual(int((labels[calibration] == 0).sum()), 3)
        self.assertEqual(int((labels[calibration] == 1).sum()), 3)

    def test_matrix_ops_backend_receives_algorithm_matrix_products(self) -> None:
        class RecordingBackend(MatrixOpsBackend):
            def __init__(self) -> None:
                self.names: list[str] = []

            def matmul(self, left, right, *, name: str = "matmul"):
                self.names.append(name)
                return np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64)

            def einsum(self, subscripts: str, *operands, name: str = "einsum"):
                self.names.append(name)
                arrays = [np.asarray(operand, dtype=np.float64) for operand in operands]
                return np.einsum(subscripts, *arrays)

        rng = np.random.default_rng(23)
        labels = np.repeat(np.arange(2), 8)
        trials = rng.normal(size=(len(labels), 3, 96))
        backend = RecordingBackend()

        with use_matrix_ops_backend(backend):
            fbcsp = FilterBankCSP(
                bands=((8.0, 12.0),),
                n_components=1,
                covariance_shrinkage=0.10,
            )
            features = fbcsp.fit_transform(trials, labels, 100.0, ("left", "right"))
            standardizer = FeatureStandardizer()
            standardized = standardizer.fit_transform(features.vector)
            lda = ShrinkageLDA(shrinkage=0.20).fit(
                standardized,
                labels,
                class_names=("left", "right"),
            )
            _scores = lda.scores(standardized[:3])
            head = PrototypeDecisionHead(
                prototypes=np.eye(2, dtype=np.float64),
                config=DecisionConfig(class_names=("left", "right")),
            )
            _decisions = head.decide_all(np.array([[0.8, 0.2]], dtype=np.float64))

        self.assertIn("fbcsp_trial_covariance", backend.names)
        self.assertIn("fbcsp_spatial_projection", backend.names)
        self.assertIn("feature_standardizer_affine", backend.names)
        self.assertIn("lda_pooled_covariance", backend.names)
        self.assertIn("linear_head_scores", backend.names)
        self.assertIn("prototype_decision_distances_shared_cross", backend.names)

    def test_signal_ops_backend_receives_fbcsp_filtering(self) -> None:
        class RecordingSignalBackend(SignalOpsBackend):
            def __init__(self) -> None:
                self.names: list[str] = []

            def common_average_reference(
                self,
                samples,
                *,
                channel_axis: int,
                name: str = "common_average_reference",
            ):
                self.names.append(name)
                x = np.asarray(samples, dtype=np.float64)
                return x - x.mean(axis=channel_axis, keepdims=True)

            def sosfiltfilt(self, sos, samples, *, axis: int, name: str = "sosfiltfilt"):
                self.names.append(name)
                from scipy.signal import sosfiltfilt

                return sosfiltfilt(
                    np.asarray(sos, dtype=np.float64),
                    np.asarray(samples, dtype=np.float64),
                    axis=axis,
                )

        rng = np.random.default_rng(31)
        labels = np.repeat(np.arange(2), 8)
        trials = rng.normal(size=(len(labels), 3, 128))
        backend = RecordingSignalBackend()

        with use_signal_ops_backend(backend):
            fbcsp = FilterBankCSP(
                bands=((8.0, 12.0),),
                n_components=1,
                covariance_shrinkage=0.10,
            )
            _features = fbcsp.fit_transform(trials, labels, 100.0, ("left", "right"))

        self.assertIn("fbcsp_filter_bank_sosfiltfilt", backend.names)


if __name__ == "__main__":
    unittest.main()
