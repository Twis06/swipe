import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import server
from audio_format import decode_wav
from preview_model import features, predict
from preview_service import PreviewService
from test_recorder import RecorderTest, float_wav
from train_preview import Experiment, ROOT


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service=PreviewService()

    def test_export_matches_training_pipeline_for_real_audio(self):
        e=Experiment()
        ids=np.arange(len(e.y));config=self.service.config
        fitted=e.fit(ids,(config['feature_input'],config['neighbors']))
        expected=e.evaluate(fitted,np.array([0,28,29,30]))
        for index,row in zip([0,28,29,30],expected):
            clip=e.clips[index];blob=(ROOT/'data'/clip['session']/clip['filename']).read_bytes()
            actual=self.service.classify(blob)
            self.assertEqual(actual['prediction'],row['prediction'])
            self.assertAlmostEqual(actual['distance_margin'],row['distance_margin'],places=9)

    def test_silence_rejected_and_busy_service_does_not_queue(self):
        blob=float_wav([0]*96000)
        self.assertEqual(self.service.classify(blob)['prediction'],'background')
        with self.service.lock:
            with self.assertRaises(BlockingIOError):self.service.classify(blob)

    def test_gain_invariance_for_each_feature_copy(self):
        clip=ROOT/'data/20261103T200050_87dcb0/001_b1_down_take1.wav'
        samples,rate,_=decode_wav(clip.read_bytes());x=np.array(samples);original=x.copy()
        for kind in ('raw','soft_polarity','polarity'):
            a=features(x,rate,kind)[0];b=features(x*.25,rate,kind)[0]
            np.testing.assert_allclose(a,b,atol=1e-10)
        np.testing.assert_array_equal(x,original)


class PreviewEndpointTests(RecorderTest):
    def test_predict_is_authenticated_and_does_not_save(self):
        self.assertEqual(self.request('/api/predict',float_wav([0]*96000),token='wrong')[0],403)
        with patch.object(server,'PREVIEW',PreviewService()):
            status,result=self.request('/api/predict',float_wav([0]*96000))
            self.assertEqual(status,200);self.assertEqual(result['prediction'],'background');self.assertFalse(result['audio_saved'])
            self.assertEqual(list(server.DATA.iterdir()),[])
            self.assertEqual(self.request('/api/predict',b'not wav')[0],400)
            self.assertEqual(self.request('/api/preview')[0],200)


if __name__=='__main__':unittest.main()
