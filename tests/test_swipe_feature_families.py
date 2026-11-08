import unittest
import numpy as np
from swipe.swipe_feature_families import extract_all, METHODS
from swipe.preview_model import features

class FeatureTests(unittest.TestCase):
    def test_gain_and_input_preservation(self):
        rate=48000;t=np.arange(rate*2)/rate
        x=(np.exp(-((t-.7)/.04)**2)*np.sin(2*np.pi*(700*t+1200*t*t))).astype(float)
        before=x.copy();a=extract_all(x,rate);b=extract_all(x*7,rate)
        np.testing.assert_array_equal(x,before)
        for key in METHODS:
            self.assertEqual(a[key].shape[0],32)
            self.assertTrue(np.isfinite(a[key]).all())
            np.testing.assert_allclose(a[key],b[key],atol=1e-9)
        np.testing.assert_array_equal(a['raw'],features(x,rate,'raw')[0])

    def test_silence_and_boundary_impulse(self):
        for rate in (16000,44100,48000):
            x=np.zeros(rate//2)
            for index in (None,0,len(x)-1):
                y=x.copy()
                if index is not None:y[index]=2
                result=extract_all(y,rate)
                for value in result.values():self.assertTrue(np.isfinite(value).all())
                np.testing.assert_array_equal(result['raw'],features(y,rate,'raw')[0])

    def test_temporal_order_is_retained(self):
        rate=44100;t=np.arange(rate*2)/rate
        x=np.exp(-((t-.7)/.01)**2)*np.sin(2*np.pi*1000*t)+.5*np.exp(-((t-.76)/.02)**2)*np.sin(2*np.pi*3000*t)
        a=extract_all(x,rate);b=extract_all(x[::-1],rate)
        self.assertGreater(np.max(abs(a['envelope']-b['envelope'])),.1)
        self.assertGreater(np.max(abs(a['onset_offset']-b['onset_offset'])),.1)

if __name__=='__main__':unittest.main()
