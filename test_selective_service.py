"""Service routing, rollback and integrity checks without fitting any model."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from audio_format import decode_wav
from fast_preview import predict as raw_predict
from preview_service import PreviewService

ROOT = Path(__file__).resolve().parent


class SelectiveServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blob = (ROOT/'data/20261103T200050_87dcb0/001_b1_down_take1.wav').read_bytes()

    def test_legacy_selective_engine_remains_available_and_preserves_accepted_clip(self):
        original = Path.read_text
        def read(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if path == ROOT/'diagnostics/preview/engine.json':
                config = json.loads(value); config['engine'] = 'selective_encoding_fallback'
                return json.dumps(config)
            return value
        with patch.object(Path, 'read_text', read):
            service = PreviewService()
        self.assertEqual(service.info()['recognizer'], 'selective_encoding_fallback')
        self.assertIsNotNone(service.aligned_model)
        samples, rate, _ = decode_wav(self.blob, array=True)
        expected = raw_predict(samples, rate, service.model)
        self.assertEqual(expected['reason'], 'accepted')
        actual = service.classify(self.blob)
        self.assertFalse(actual.pop('audio_saved'))
        actual.pop('processing_ms')
        self.assertEqual(actual, expected)

    def test_manifest_can_restore_raw_engine_without_refitting(self):
        original = Path.read_text
        def read(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if path == ROOT/'diagnostics/preview/engine.json':
                config = json.loads(value)
                config['engine'] = 'exact_batched_dtw'
                return json.dumps(config)
            return value
        with patch.object(Path, 'read_text', read):
            service = PreviewService()
        self.assertIsNone(service.aligned_model)
        self.assertEqual(service.info()['recognizer'], 'exact_batched_dtw')
        with patch('selective_preview.predict', side_effect=AssertionError('Raw rollback must not invoke fallback')):
            self.assertEqual(service.classify(self.blob)['prediction'], 'down')

    def test_changed_fallback_bank_is_refused(self):
        original = Path.read_bytes
        def read(path):
            if path == ROOT/'diagnostics/aligned_preview/model.npz':
                return b'changed bank'
            return original(path)
        with patch.object(Path, 'read_bytes', read):
            with self.assertRaisesRegex(ValueError, 'Aligned reference bank checksum changed'):
                PreviewService()

    def test_unknown_engine_is_refused(self):
        original = Path.read_text
        def read(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if path == ROOT/'diagnostics/preview/engine.json':
                config = json.loads(value)
                config['engine'] = 'unknown'
                return json.dumps(config)
            return value
        with patch.object(Path, 'read_text', read):
            with self.assertRaisesRegex(ValueError, 'Unknown preview inference engine'):
                PreviewService()


if __name__ == '__main__':
    unittest.main()
