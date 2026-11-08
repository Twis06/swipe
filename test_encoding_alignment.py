import unittest
import numpy as np
from encoding_alignment import align_signal,reference_indices,summary


class EncodingAlignmentTests(unittest.TestCase):
    def test_encoding_copy_values_and_legacy_preservation(self):
        x=np.tile(np.array([-4.,-1.,-.5,0.,.5,1.,4.]),2000)
        original=x.copy();actual=align_signal(x,44100,'float32')
        expected=np.clip(x,-1,1);expected=np.where(expected<0,expected*32768,expected*32767).astype(np.int16).astype(float)/32768
        np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(x,original)
        np.testing.assert_array_equal(align_signal(actual,44100,'pcm16'),actual)

    def test_queries_excluded_and_negatives_not_references(self):
        records=[{'session':'a','group':'a1','label':'up'},{'session':'a','group':'a2','label':'down'},
                 {'session':'b','group':'b1','label':'up'},{'session':'b','group':'b1','label':'idle'}]
        self.assertEqual(reference_indices(0,records,'cross_session').tolist(),[2])
        self.assertEqual(reference_indices(0,records,'leave_group_out').tolist(),[1,2])

    def test_accepted_wrong_and_background_counted(self):
        rows=[{'truth':'up','direction_candidate':'down','prediction':'down'},
              {'truth':'typing','direction_candidate':'up','prediction':'up'}]
        s=summary(rows,True)
        self.assertEqual(s['accepted_wrong'],1);self.assertEqual(s['negative_activations'],1)


if __name__=='__main__':unittest.main()
