import unittest
import numpy as np
from swipe.preview_model import features
from research.robust_feature_probe import FORMULAS, transform, reference_indices


class RobustFeaturesTests(unittest.TestCase):
    def test_silence_and_dimensions(self):
        x = features(np.zeros(88200), 44100, 'raw')[0]
        for name in FORMULAS:
            result = transform(x, name)
            self.assertEqual(result.shape, (32, 16 if name == 'shape_delta' else 8))
            self.assertTrue(np.isfinite(result).all())

    def test_gain_invariance_of_features_and_fixed_transforms(self):
        rng = np.random.default_rng(17)
        x = rng.normal(0, .001, 88200)
        x[30000:36000] += rng.normal(0, .2, 6000)
        reference = features(x, 44100, 'raw')[0]
        for gain in (.1, 10):
            changed = features(x*gain, 44100, 'raw')[0]
            for name in FORMULAS:
                np.testing.assert_allclose(transform(reference, name), transform(changed, name), atol=1e-9)

    def test_reference_exclusion(self):
        records = [{'session':'a', 'group':'a1'}, {'session':'a', 'group':'a2'}, {'session':'b', 'group':'b1'}]
        self.assertEqual(reference_indices(0, records, 'cross_session').tolist(), [2])
        self.assertEqual(reference_indices(0, records, 'leave_group_out').tolist(), [1,2])


if __name__ == '__main__':
    unittest.main()
