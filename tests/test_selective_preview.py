from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from swipe.aligned_preview import load_model, predict as aligned_predict
from swipe.audio_format import decode_wav
from swipe.fast_preview import predict as raw_predict
from swipe.selective_preview import classify_with_raw_result, predict

ROOT = Path(__file__).resolve().parents[1]


class SelectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aligned, _ = load_model()
        with np.load(ROOT/'models/preview/model.npz', allow_pickle=False) as archive:
            cls.raw = {key: archive[key] for key in archive.files}
        cls.samples, cls.rate, _ = decode_wav(
            (ROOT/'data/20261103T200050_87dcb0/001_b1_down_take1.wav').read_bytes(), array=True)

    def test_preserves_every_other_reason_without_alignment(self):
        for reason in ('accepted', 'background_gate', 'silence', 'ambiguous_direction'):
            raw = {'reason': reason, 'prediction': 'up', 'extra_future_field': [1, 2]}
            before = deepcopy(raw)
            with patch('swipe.selective_preview.direction_features', side_effect=AssertionError('Unnecessary alignment')):
                self.assertIs(classify_with_raw_result(self.samples, self.rate, raw, self.aligned), raw)
                with patch('swipe.selective_preview.raw_predict', return_value=raw) as first:
                    self.assertIs(predict(self.samples, self.rate, self.raw, self.aligned), raw)
                    self.assertEqual(first.call_count, 1)
            self.assertEqual(raw, before)

    def test_rescue_matches_standalone_alignment_and_reuses_raw_quality(self):
        raw = raw_predict(self.samples, self.rate, self.raw)
        raw['reason'] = 'unfamiliar_sound'
        raw['prediction'] = 'uncertain'
        before = deepcopy(raw)
        samples_before = self.samples.copy()
        expected = aligned_predict(self.samples, self.rate, self.aligned)
        self.assertIn(expected['prediction'], ('up', 'down'))
        result = classify_with_raw_result(self.samples, self.rate, raw, self.aligned)
        self.assertEqual({k: v for k, v in result.items() if k != 'recognition_path'}, expected)
        self.assertEqual(result['recognition_path'], 'encoding_fallback')
        self.assertIs(result['quality'], raw['quality'])
        self.assertEqual(raw, before)
        np.testing.assert_array_equal(self.samples, samples_before)

    def test_unsuccessful_alignment_keeps_raw_result_exactly(self):
        raw = raw_predict(self.samples, self.rate, self.raw)
        raw.update(reason='unfamiliar_sound', prediction='uncertain')
        aligned = dict(self.aligned, distance_cutoff=np.array(-1.0))
        self.assertIs(classify_with_raw_result(self.samples, self.rate, raw, aligned), raw)

    def test_raw_accepted_result_and_overrange_input_preserved(self):
        before = self.samples.copy()
        self.assertGreater(float(np.max(np.abs(self.samples))), 1.0)
        raw = raw_predict(self.samples, self.rate, self.raw)
        self.assertEqual(raw['reason'], 'accepted')
        self.assertEqual(predict(self.samples, self.rate, self.raw, self.aligned), raw)
        np.testing.assert_array_equal(self.samples, before)


if __name__ == '__main__':
    unittest.main()
