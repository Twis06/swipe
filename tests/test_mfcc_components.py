import unittest
import numpy as np
from swipe.mfcc_components import METHODS,extract_variant
from research.mfcc_direction import extract_mfcc
from research.probe_mfcc_rejection import cutoff_for_references

class ComponentTests(unittest.TestCase):
    def test_all_variants_gain_invariance_and_preservation(self):
        rate=48000;t=np.arange(2*rate)/rate
        x=np.exp(-((t-.8)/.06)**2)*np.sin(2*np.pi*(800*t+500*t*t));before=x.copy()
        for name in METHODS:
            a=extract_variant(x,rate,name)
            self.assertEqual(a.shape[0],32)
            self.assertTrue(np.isfinite(a).all())
            np.testing.assert_allclose(a,extract_variant(x*.25,rate,name),atol=1e-10)
        np.testing.assert_array_equal(x,before)
        np.testing.assert_allclose(extract_variant(x,rate),extract_mfcc(x,rate),atol=1e-10)

    def test_silence_and_boundary_impulses(self):
        for location in (None,0,22049):
            x=np.zeros(22050)
            if location is not None:x[location]=2
            for name in METHODS:self.assertTrue(np.isfinite(extract_variant(x,44100,name)).all())

    def test_cutoff_independent_of_heldout_session(self):
        rng=np.random.default_rng(7);d=rng.random((6,6));np.fill_diagonal(d,0)
        labels=np.array(['up','down']*3);sessions=np.array(['a','a','b','b','heldout','heldout']);refs=np.arange(4)
        cutoff=cutoff_for_references(d,labels,sessions,refs)
        d[4:,:]=1e9;d[:,4:]=1e9;labels[4:]='down'
        self.assertEqual(cutoff_for_references(d,labels,sessions,refs),cutoff)
        with self.assertRaises(ValueError):cutoff_for_references(d,labels,sessions,np.arange(2))

if __name__=='__main__':unittest.main()
