import unittest

import numpy as np

from demo_api.engine import _normalized_fft_spectrum


class DemoSpectrumTest(unittest.TestCase):
    def test_normalized_fft_keeps_all_channels_and_frequency_peaks(self) -> None:
        sample_rate_hz = 250.0
        time = np.arange(750, dtype=np.float64) / sample_rate_hz
        trial = np.stack(
            [
                np.sin(2.0 * np.pi * 10.0 * time),
                0.4 * np.sin(2.0 * np.pi * 20.0 * time),
                np.zeros_like(time),
            ]
        )

        frequencies, spectrum_db = _normalized_fft_spectrum(trial, sample_rate_hz)

        self.assertEqual(spectrum_db.shape, (3, len(frequencies)))
        self.assertAlmostEqual(float(frequencies[-1]), 40.0)
        self.assertAlmostEqual(float(frequencies[np.argmax(spectrum_db[0])]), 10.0)
        self.assertAlmostEqual(float(frequencies[np.argmax(spectrum_db[1])]), 20.0)
        np.testing.assert_allclose(spectrum_db[:2].max(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(spectrum_db[2], -60.0, atol=1e-12)
        self.assertTrue(np.isfinite(spectrum_db).all())


if __name__ == "__main__":
    unittest.main()
