import struct
import unittest
from pathlib import Path
import numpy as np
from swipe.audio_format import decode_wav
from tests.test_recorder import float_wav

ROOT=Path(__file__).resolve().parents[1]


class FastAudioTests(unittest.TestCase):
    def test_real_encodings_have_identical_samples(self):
        for sid in ('20261103T165139_cf7201','20261103T200050_87dcb0'):
            for path in sorted((ROOT/'data'/sid).glob('*.wav')):
                blob=path.read_bytes()
                expected,rate,encoding=decode_wav(blob)
                actual,actual_rate,actual_encoding=decode_wav(blob,array=True)
                self.assertEqual((rate,encoding),(actual_rate,actual_encoding))
                np.testing.assert_array_equal(actual,expected)

    def test_invalid_layout_and_nonfinite_are_rejected_both_ways(self):
        valid=float_wav([0,1,2])
        bad_align=bytearray(valid)
        struct.pack_into('<H',bad_align,32,2)
        blobs=[b'bad',valid[:-1],bytes(bad_align),float_wav([float('nan')]),float_wav([float('inf')]),float_wav([])]
        for blob in blobs:
            for array in (False,True):
                with self.assertRaises(ValueError):decode_wav(blob,array=array)

    def test_headroom_and_signed_zero_are_preserved(self):
        actual,_,_=decode_wav(float_wav([30.,-28.,-0.,0.]),array=True)
        np.testing.assert_array_equal(actual,[30.,-28.,-0.,0.])
        self.assertTrue(np.signbit(actual[2]))


if __name__=='__main__':unittest.main()
