import unittest

from hybrid_photonic_mi_bci.compute_accounting import (
    LinearComputeLedger,
    add_car_event,
    add_candidate_scan_events,
    add_linear_scores_event,
    add_sosfiltfilt_event,
    add_mlp_forward_event,
    add_mlp_training_event,
)


class ComputeAccountingTest(unittest.TestCase):
    def test_photonic_and_digital_shares_are_summarized_by_stage(self) -> None:
        ledger = LinearComputeLedger()
        add_linear_scores_event(
            ledger,
            name="numpy score",
            n_samples=5,
            n_features=4,
            n_outputs=3,
            stage="inference",
        )
        add_mlp_forward_event(
            ledger,
            name="torch forward",
            n_samples=5,
            input_dim=4,
            hidden_dim=6,
            embedding_dim=2,
            n_classes=3,
            stage="inference",
        )

        summary = ledger.summary()

        self.assertEqual(summary["linear_macs_photonic"], 440)
        self.assertEqual(summary["linear_macs_digital"], 0)
        self.assertAlmostEqual(summary["photonic_linear_share_inference"], 1.0)

    def test_candidate_scan_counts_head_scan_and_fusion(self) -> None:
        ledger = LinearComputeLedger()
        add_candidate_scan_events(
            ledger,
            prefix="scan",
            n_windows=7,
            n_candidates=3,
            n_features=5,
            n_classes=2,
            stage="inference",
        )

        summary = ledger.summary()

        self.assertEqual(summary["linear_macs_photonic"], 7 * 3 * 6 * 2 + 7 * 3 * 2)
        self.assertEqual(summary["photonic_linear_share"], 1.0)

    def test_car_and_sos_filtering_are_photonic_signal_ops(self) -> None:
        ledger = LinearComputeLedger()
        add_car_event(
            ledger,
            prefix="loader",
            n_samples=100,
            n_channels=8,
        )
        add_sosfiltfilt_event(
            ledger,
            name="filter",
            n_trials=2,
            n_bands=3,
            n_channels=4,
            n_samples=50,
            filter_order=4,
            stage="inference",
        )

        summary = ledger.summary()

        expected = 100 * 8 * 3 + 2 * 3 * 4 * 50 * 4 * 5 * 2
        self.assertEqual(summary["linear_macs_photonic"], expected)
        self.assertEqual(summary["linear_macs_digital"], 0)
        self.assertEqual(summary["by_stage"]["preprocessing"]["linear_macs_photonic"], 2400)
        self.assertEqual(summary["photonic_linear_share"], 1.0)

    def test_training_backprop_is_excluded_from_forward_share(self) -> None:
        ledger = LinearComputeLedger()
        add_mlp_training_event(
            ledger,
            prefix="train",
            n_samples=10,
            input_dim=4,
            hidden_dim=6,
            embedding_dim=2,
            n_classes=3,
            epochs=5,
        )
        add_mlp_forward_event(
            ledger,
            name="eval forward",
            n_samples=2,
            input_dim=4,
            hidden_dim=6,
            embedding_dim=2,
            n_classes=3,
            stage="inference",
        )

        summary = ledger.summary()

        self.assertEqual(summary["linear_macs_total"], 146)
        self.assertEqual(summary["linear_macs_photonic"], 146)
        self.assertAlmostEqual(summary["photonic_linear_share"], 1.0)
        self.assertGreater(summary["linear_macs_all_stages_total"], summary["linear_macs_total"])
        self.assertEqual(summary["linear_macs_fit_or_training_excluded"], 10950)


if __name__ == "__main__":
    unittest.main()
