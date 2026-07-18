import unittest
from types import SimpleNamespace

import numpy as np

from demo_api.engine import BCICIV_DATASET_KEY, LiveInferenceEngine


class DemoProtocolTest(unittest.TestCase):
    def test_bciciv_split_matches_single_window_interface(self) -> None:
        engine = LiveInferenceEngine()
        target = SimpleNamespace(subject="a", trials=np.zeros((200, 1, 2)))

        calibration, evaluation = engine._bciciv_pooled_indices(
            target,
            np.zeros(200, dtype=int),
        )

        np.testing.assert_array_equal(calibration, np.arange(120, 126))
        np.testing.assert_array_equal(evaluation, np.arange(126, 200))

    def test_bciciv_runtime_uses_all_subjects_and_pooled_calibration(self) -> None:
        engine = LiveInferenceEngine()

        self.assertEqual(
            engine._training_subjects(BCICIV_DATASET_KEY, "a"),
            tuple("abcdefg"),
        )
        self.assertEqual(engine._runtime_calibration_windows(BCICIV_DATASET_KEY, 6), 42)


if __name__ == "__main__":
    unittest.main()
