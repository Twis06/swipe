import unittest
import numpy as np
from motif_probe import features, subsequences, join, disjoint_matches


class MotifProbeTests(unittest.TestCase):
    def test_shifted_repeat_and_disjoint(self):
        rng=np.random.default_rng(3)
        a=rng.normal(size=(80,8));b=rng.normal(size=(100,8))
        b[53:63]=a[19:29]
        rms_a=np.ones(80);rms_b=np.ones(100)
        rms_a[23]=20;rms_b[57]=20
        ia,va=subsequences(a,rms_a,10);ib,vb=subsequences(b,rms_b,10)
        d=join(va,vb)
        best=disjoint_matches(d,ia,ib,10,3)
        self.assertEqual((best[0]['start_frame_a'],best[0]['start_frame_b']),(19,53))
        self.assertLess(best[0]['distance'],1e-12)
        for i,x in enumerate(best):
            for y in best[i+1:]:
                self.assertGreaterEqual(abs(x['start_frame_a']-y['start_frame_a']),10)
                self.assertGreaterEqual(abs(x['start_frame_b']-y['start_frame_b']),10)

    def test_audio_gain_invariance(self):
        rng=np.random.default_rng(7)
        x=rng.normal(0,.001,48000)
        x[10000:11000]+=rng.normal(0,1,1000)
        _,f,rms=features(x,48000);_,g,grms=features(x*10,48000)
        ia,a=subsequences(f,rms,10);ib,b=subsequences(g,grms,10)
        np.testing.assert_array_equal(ia,ib)
        np.testing.assert_allclose(a,b,atol=1e-12)


if __name__=='__main__':unittest.main()
