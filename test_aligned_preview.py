from pathlib import Path
import unittest

import numpy as np

from aligned_preview import align_signal, direction_features, predict
from audio_format import decode_wav
from fast_preview import predict as original
from gesture_model import RATE, canonical_audio

ROOT=Path(__file__).resolve().parent


class AlignedTests(unittest.TestCase):
    def test_transform_order_and_source_immutability(self):
        x=np.sin(np.arange(48000)*.37)*8;before=x.copy()
        canonical=np.clip(canonical_audio(x,48000),-1,1)
        expected=np.where(canonical<0,canonical*32768,canonical*32767).astype(np.int16).astype(float)/32768
        np.testing.assert_array_equal(align_signal(x,48000),expected)
        np.testing.assert_array_equal(x,before)

    def test_pcm16_preserved_and_live_float_path_explicit(self):
        x=np.full(RATE,32767/32768)
        np.testing.assert_array_equal(align_signal(x,RATE,'pcm16'),x)
        self.assertLess(align_signal(x,RATE,'float32')[0],x[0])

    def test_live_float_feature_equals_reference_feature(self):
        samples,rate,encoding=decode_wav((ROOT/'data/20261103T200050_87dcb0/001_b1_down_take1.wav').read_bytes(),array=True)
        np.testing.assert_array_equal(direction_features(samples,rate),direction_features(samples,rate,encoding))

    def test_raw_gate_quality_and_timestamps_unchanged(self):
        with np.load(ROOT/'diagnostics/preview/model.npz',allow_pickle=False) as archive:
            model={key:archive[key] for key in archive.files}
        x=np.sin(np.arange(96000)*.3)*20
        result=predict(x,48000,model)
        old=original(x,48000,model)
        self.assertEqual(result['swipe_gate_score'],old['swipe_gate_score'])
        self.assertEqual(result['quality'],old['quality'])


if __name__=='__main__':unittest.main()
