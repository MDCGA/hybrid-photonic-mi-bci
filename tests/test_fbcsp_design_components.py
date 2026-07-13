import unittest

import numpy as np

from hybrid_photonic_mi_bci.backends import TiledMVMBackend
from hybrid_photonic_mi_bci.datasets.bnci2014_004 import calibration_eval_split
from hybrid_photonic_mi_bci.experience import (
    build_bootstrap_experience_library,
    retrieve_top_k,
    scan_experience_heads,
)
from hybrid_photonic_mi_bci.fbcsp import FilterBankCSP
from hybrid_photonic_mi_bci.linear_models import FeatureStandardizer, ShrinkageLDA
from hybrid_photonic_mi_bci.workflows.common import make_replay_split


class FBCSPDesignComponentsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
