from pathlib import Path
import unittest
from unittest.mock import patch

import server
from preview_service import PreviewService


class EngineVerificationTests(unittest.TestCase):
    def test_changed_engine_source_is_refused(self):
        read_bytes=Path.read_bytes
        def changed(path):
            return b'changed' if path.name=='fast_preview.py' else read_bytes(path)
        with patch.object(Path,'read_bytes',changed):
            with self.assertRaisesRegex(ValueError,'inference engine changed'):
                PreviewService()

    def test_engine_error_does_not_recommend_refitting(self):
        with patch.object(server,'PREVIEW',None), patch('preview_service.PreviewService',side_effect=ValueError('engine checksum failed')):
            with self.assertRaisesRegex(RuntimeError,'engine checksum failed') as error:
                server.preview()
            self.assertNotIn('train_preview.py',str(error.exception))
            self.assertIn('retraining is not required',str(error.exception))


if __name__=='__main__':unittest.main()
