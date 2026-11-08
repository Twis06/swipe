from pathlib import Path
import json,unittest
from unittest.mock import patch
import numpy as np
from swipe.mfcc_preview import gate_quality,load_model,predict
from swipe.mfcc_fallback import predict as hybrid
from swipe.audio_format import decode_wav
from swipe.gesture_model import canonical_audio
from swipe.preview_model import features

ROOT=Path(__file__).resolve().parents[1]

class MFCCPreviewTests(unittest.TestCase):
    def test_active_service_routes_to_candidate_with_correct_counts(self):
        from swipe.preview_service import PreviewService
        service=PreviewService();info=service.info()
        self.assertEqual(info['recognizer'],'mfcc8_fallback')
        self.assertEqual(info['reference_swipes'],60)
        self.assertEqual(info['training_sessions'],3)
        blob=(ROOT/'data/20261107T185348_9cbd41/013_b1_up_take1.wav').read_bytes()
        x,rate,_=decode_wav(blob,array=True)
        expected=hybrid(x,rate,service.mfcc_model,service.model,service.aligned_model)
        actual=service.classify(blob);actual.pop('processing_ms');self.assertFalse(actual.pop('audio_saved'))
        self.assertEqual(actual,expected)

    def test_gate_quality_and_timestamps_match_all_recordings(self):
        for r in json.loads((ROOT/'diagnostics/mfcc_components_v1/records.json').read_text()):
            x,rate,_=decode_wav((ROOT/'data'/r['session']/r['file']).read_bytes(),array=True)
            _,g,q=features(x,rate,'raw');actual_g,actual_q=gate_quality(canonical_audio(x,rate))
            np.testing.assert_array_equal(actual_g,g);self.assertEqual(actual_q,q)

    def test_fallback_routes_only_uncertain_and_only_accepts_direction(self):
        for prediction in ('up','down','background'):
            result={'prediction':prediction}
            with patch('swipe.mfcc_fallback.primary_predict',return_value=result),patch('swipe.mfcc_fallback.legacy_predict',side_effect=AssertionError('Unnecessary fallback')):
                self.assertIs(hybrid([],48000,{}, {},{}),result)
        primary={'prediction':'uncertain','quality':{'event_start_seconds':.5}}
        for prediction in ('uncertain','background','up','down'):
            old={'prediction':prediction,'quality':primary['quality']}
            with patch('swipe.mfcc_fallback.primary_predict',return_value=primary),patch('swipe.mfcc_fallback.legacy_predict',return_value=old):
                result=hybrid([],48000,{},{},{})
                self.assertEqual(result['prediction'],prediction if prediction in ('up','down') else 'uncertain')

    def test_candidate_silence_and_input_preservation(self):
        model,_=load_model();x=np.zeros(96000);before=x.copy()
        self.assertEqual(predict(x,48000,model)['prediction'],'background')
        np.testing.assert_array_equal(x,before)

    def test_changed_model_is_refused(self):
        original=Path.read_bytes
        def read(path):return b'changed' if path==ROOT/'models/mfcc_preview/model.npz' else original(path)
        with patch.object(Path,'read_bytes',read),self.assertRaisesRegex(ValueError,'MFCC bank checksum'):
            load_model()

if __name__=='__main__':unittest.main()
