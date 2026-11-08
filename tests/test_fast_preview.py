import unittest
from pathlib import Path

import numpy as np

from swipe.audio_format import decode_wav
from swipe.gesture_model import dtw, direction
from swipe.fast_preview import batch_dtw, predict
from swipe.preview_model import predict as reference_predict

ROOT = Path(__file__).resolve().parents[1]


class FastPreviewTests(unittest.TestCase):
    def test_random_shapes_radii_and_unreachable_paths_are_exact(self):
        rng = np.random.default_rng(391)
        for n, m, dimensions, count in [(32,32,8,40), (7,11,3,6), (1,1,1,2), (4,8,2,1)]:
            query = rng.normal(size=(n, dimensions))
            templates = rng.normal(size=(count, m, dimensions))
            for radius in (0, 1, 4, 20):
                expected = np.array([dtw(query, template, radius) for template in templates])
                np.testing.assert_array_equal(batch_dtw(query, templates, radius), expected)

    def test_ties_and_near_ties_keep_reference_order(self):
        query = np.zeros((32, 8))
        templates = np.stack([query, query, query+1e-12, query+1e-11])
        labels = np.array(['down', 'up', 'up', 'down'])
        expected = np.array([dtw(query, template, 4) for template in templates])
        actual = batch_dtw(query, templates, 4)
        np.testing.assert_array_equal(actual, expected)
        for k in (1, 2, 3):
            self.assertEqual(direction(actual, labels, k), direction(expected, labels, k))

    def test_real_decision_and_numeric_fields_identical(self):
        with np.load(ROOT/'models/preview/model.npz', allow_pickle=False) as bank:
            model = {key: bank[key] for key in bank.files}
        paths = [ROOT/'data/20261103T165139_cf7201/001_b1_down_take1.wav',
                 ROOT/'data/20261103T200050_87dcb0/002_b1_up_take1.wav']
        for path in paths:
            samples, rate, _ = decode_wav(path.read_bytes())
            self.assertEqual(predict(samples, rate, model), reference_predict(samples, rate, model))
        self.assertEqual(predict(np.zeros(96000), 48000, model), reference_predict(np.zeros(96000), 48000, model))

    def test_input_contract_and_empty_reference_bank(self):
        self.assertEqual(batch_dtw(np.zeros((2,3)), np.zeros((0,2,3))).shape, (0,))
        with self.assertRaises(ValueError):
            batch_dtw(np.zeros((2,3)), np.zeros((4,2,2)))
        with self.assertRaises(ValueError):
            batch_dtw(np.zeros((2,3)), np.zeros((4,2,3)), -1)


if __name__ == '__main__':
    unittest.main()
