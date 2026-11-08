import unittest
import numpy as np
from research.segmentation_experiment import baseline, hysteresis, frames, spectral_boundaries, split_at


class SegmentationTests(unittest.TestCase):
    def test_silence(self):
        for method in (baseline, hysteresis):
            self.assertEqual(method(np.zeros(300)), [])

    def test_separated_events_and_short_gap(self):
        x = np.full(300, .01)
        x[30:60] = 1
        x[150:180] = 1
        self.assertEqual(hysteresis(x), [(30, 60), (150, 180)])
        # Two peaks separated by 50 ms should be one event under the declared
        # 80 ms hangover; this is a known test property, not a gesture claim.
        x[60:65] = .01
        x[65:80] = 1
        self.assertEqual(hysteresis(x), [(30, 80), (150, 180)])

    def test_relative_gain_invariance_above_absolute_floor(self):
        x = np.full(300, .01)
        x[30:60] = 1
        x[150:180] = 1
        for gain in (.1, 1, 10):
            self.assertEqual(hysteresis(x*gain), [(30, 60), (150, 180)])

    def test_edge_events_preserved_by_hysteresis(self):
        x = np.full(200, .01)
        x[:10] = 1
        x[-10:] = 1
        self.assertEqual(hysteresis(x), [(0, 10), (190, 200)])

    def test_spectral_change_inside_sustained_sound(self):
        rate = 48000
        t = np.arange(rate*2)/rate
        x = np.sin(2*np.pi*np.where(t < 1, 1000, 6000)*t)
        _, shape = frames(x, rate)
        cuts, _ = spectral_boundaries(shape, [(0, len(shape))])
        self.assertTrue(any(abs(c-100) <= 3 for c in cuts), cuts)
        self.assertEqual(len(split_at([(0, 200)], [100])), 2)


if __name__ == '__main__':
    unittest.main()
