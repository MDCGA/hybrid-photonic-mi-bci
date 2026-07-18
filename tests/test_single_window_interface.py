import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from examples.run_single_window_inference import run_one_online_forward
from hybrid_photonic_mi_bci.backends import (
    ScipySignalOpsBackend,
    SimulatedPhotonicSignalOpsBackend,
    get_signal_ops_backend,
    use_signal_ops_backend,
)


class _FBCSPProbe:
    def __init__(self) -> None:
        self.signal_backend = None

    def transform(self, trial: np.ndarray) -> SimpleNamespace:
        self.signal_backend = get_signal_ops_backend()
        return SimpleNamespace(vector=np.asarray(trial, dtype=np.float64))


class _IdentityStandardizer:
    def transform(self, features: np.ndarray) -> np.ndarray:
        return features


class _RuntimeProbe:
    def predict(self, embedding: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(tile_count_per_window=7, embedding=embedding)


class SingleWindowInterfaceTest(unittest.TestCase):
    def test_online_forward_uses_digital_recursive_filter_and_restores_backend(self) -> None:
        fbcsp = _FBCSPProbe()
        prepared = SimpleNamespace(
            fbcsp=fbcsp,
            selected_indices=np.array([0, 1]),
            standardizer=_IdentityStandardizer(),
            mlp_model=object(),
            runtime=_RuntimeProbe(),
        )
        installed_backend = SimulatedPhotonicSignalOpsBackend()

        with patch(
            "examples.run_single_window_inference._forward_numpy",
            return_value=(np.zeros((1, 2)), np.ones((1, 2))),
        ), use_signal_ops_backend(installed_backend):
            output = run_one_online_forward(
                prepared,
                np.array([[1.0, 2.0]], dtype=np.float64),
                {},
                {},
            )
            self.assertIs(get_signal_ops_backend(), installed_backend)

        self.assertIsInstance(fbcsp.signal_backend, ScipySignalOpsBackend)
        self.assertEqual(output.tile_count_per_window, 7)


if __name__ == "__main__":
    unittest.main()
