import io
import json
import struct
import unittest
import numpy as np
from swipe.native_worker import read_exact, serve
from swipe.preview_service import PreviewService

class NativeWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = PreviewService()
    def test_bridge_uses_exact_service_and_never_saves(self):
        samples = np.zeros(96000, dtype='<f4').tobytes()
        wav = b'RIFF' + struct.pack('<I', 36+len(samples)) + b'WAVEfmt ' + struct.pack('<IHHIIHH',16,3,1,48000,192000,4,32) + b'data' + struct.pack('<I',len(samples)) + samples
        output = io.StringIO()
        serve(io.BytesIO(struct.pack('<I',len(wav))+wav), output, self.service)
        ready,result = map(json.loads, output.getvalue().splitlines())
        expected=self.service.classify(wav)
        self.assertTrue(ready['ready'])
        self.assertEqual(result['prediction'],expected['prediction'])
        self.assertEqual(result['reason'],expected['reason'])
        self.assertFalse(result['audio_saved'])
    def test_malformed_and_oversized_frames_fail_closed(self):
        for blob in [b'x', struct.pack('<I',2_000_001),struct.pack('<I',100)]:
            with self.assertRaises(ValueError): serve(io.BytesIO(blob),io.StringIO(),self.service)
        self.assertIsNone(read_exact(io.BytesIO(),4))
    def test_invalid_wav_returns_error_not_prediction(self):
        output=io.StringIO()
        serve(io.BytesIO(struct.pack('<I',44)+b'x'*44),output,self.service)
        self.assertIn('error',json.loads(output.getvalue().splitlines()[1]))
