"""Public-checkout integration tests: bundled banks and synthetic audio only."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import server
from audio_format import decode_wav
from fast_preview import predict as raw_predict
from gesture_model import canonical_audio
from mfcc_fallback import predict as hybrid_predict
from mfcc_preview import gate_quality
from preview_model import features, predict as reference_predict
from preview_service import PreviewService
from test_recorder import RecorderTest, float_wav

ROOT = Path(__file__).resolve().parent


def synthetic_event(rate=48000):
    t = np.arange(2 * rate) / rate
    return 2 * np.exp(-((t - .8) / .05) ** 2) * np.sin(2 * np.pi * (800 * t + 600 * t * t))


class PublicRuntimeTests(unittest.TestCase):
    def test_bundled_banks_load_and_active_service_routes_without_recordings(self):
        service = PreviewService()
        info = service.info()
        self.assertEqual(info['recognizer'], 'mfcc8_fallback')
        self.assertEqual(info['reference_swipes'], 60)
        self.assertEqual(info['training_sessions'], 3)
        self.assertFalse(info['system_actions'])
        blob = float_wav(synthetic_event())
        samples, rate, _ = decode_wav(blob, array=True)
        expected = hybrid_predict(samples, rate, service.mfcc_model, service.model, service.aligned_model)
        result = service.classify(blob)
        self.assertGreaterEqual(result.pop('processing_ms'), 0)
        self.assertFalse(result.pop('audio_saved'))
        self.assertEqual(result, expected)

    def test_gate_and_fast_matcher_preserve_reference_math_at_both_rates(self):
        model = PreviewService().model
        for rate in (44100, 48000):
            with self.subTest(rate=rate):
                samples = synthetic_event(rate)
                before = samples.copy()
                _, gate, quality = features(samples, rate, 'raw')
                actual_gate, actual_quality = gate_quality(canonical_audio(samples, rate))
                np.testing.assert_array_equal(actual_gate, gate)
                self.assertEqual(actual_quality, quality)
                self.assertEqual(raw_predict(samples, rate, model), reference_predict(samples, rate, model))
                np.testing.assert_array_equal(samples, before)

    def test_silence_duration_and_busy_contracts(self):
        service = PreviewService()
        silent = float_wav(np.zeros(96000))
        self.assertEqual(service.classify(silent)['prediction'], 'background')
        with service.lock, self.assertRaises(BlockingIOError):
            service.classify(silent)
        with self.assertRaisesRegex(ValueError, 'two-second'):
            service.classify(float_wav(np.zeros(24000)))

    def test_both_rollback_engines_load_and_unknown_engine_is_refused(self):
        original = Path.read_text
        for engine in ('selective_encoding_fallback', 'exact_batched_dtw', 'unknown'):
            def read(path, *args, **kwargs):
                value = original(path, *args, **kwargs)
                if path == ROOT / 'diagnostics/preview/engine.json':
                    config = json.loads(value)
                    config['engine'] = engine
                    return json.dumps(config)
                return value
            with self.subTest(engine=engine), patch.object(Path, 'read_text', read):
                if engine == 'unknown':
                    with self.assertRaisesRegex(ValueError, 'Unknown preview inference engine'):
                        PreviewService()
                else:
                    service = PreviewService()
                    self.assertEqual(service.info()['recognizer'], engine)
                    self.assertIsNone(service.mfcc_model)
                    self.assertEqual(service.classify(float_wav(np.zeros(96000)))['prediction'], 'background')

    def test_changed_model_banks_are_refused(self):
        original = Path.read_bytes
        for directory in ('preview', 'aligned_preview', 'mfcc_preview'):
            target = ROOT / 'diagnostics' / directory / 'model.npz'
            def read(path):
                return b'invalid bank' if path == target else original(path)
            with self.subTest(bank=directory), patch.object(Path, 'read_bytes', read):
                with self.assertRaises(ValueError):
                    PreviewService()


class PublicEndpointTests(RecorderTest):
    def test_preview_is_authenticated_and_does_not_save_audio(self):
        blob = float_wav(np.zeros(96000))
        self.assertEqual(self.request('/api/predict', blob, token='wrong')[0], 403)
        with patch.object(server, 'PREVIEW', PreviewService()):
            status, result = self.request('/api/predict', blob)
            self.assertEqual(status, 200)
            self.assertEqual(result['prediction'], 'background')
            self.assertFalse(result['audio_saved'])
            self.assertEqual(self.request('/api/predict', b'not wav')[0], 400)
            self.assertEqual(self.request('/api/preview')[0], 200)
        self.assertEqual(list(server.DATA.iterdir()), [])

    def test_public_assets_load_and_private_paths_are_not_served(self):
        for route in ('/', '/preview', '/continuous', '/app.js', '/preview.js',
                      '/continuous.js', '/style.css', '/preview.css', '/audio-worklet.js'):
            with self.subTest(route=route):
                status, body = self.request(route)
                self.assertEqual(status, 200)
                self.assertTrue(body)
        status, worklet = self.request('/preview-worklet.js')
        self.assertEqual(status, 200)
        self.assertEqual(worklet, (ROOT / 'cascade-preview-worklet.js').read_bytes())
        for route in ('/data/private.wav', '/diagnostics/mfcc_preview/model.npz',
                      '/checkpoints/', '/.git/config', '/../server.py'):
            self.assertEqual(self.request(route)[0], 404)


if __name__ == '__main__':
    unittest.main()
