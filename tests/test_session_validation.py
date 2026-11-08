"""Check evaluation bookkeeping and the new-session model export."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from swipe.audio_format import decode_wav
from swipe.gesture_model import classify
from research.validate_session import summarize

ROOT = Path(__file__).resolve().parents[1]
SESSION = "20261103T200050_87dcb0"


class SessionValidationTests(unittest.TestCase):
    def test_abstention_does_not_count_as_correct_or_perfect_accuracy(self):
        rows = [{"truth": "up", "prediction": "uncertain", "direction_candidate": "down", "reason": "ambiguous_direction"},
                {"truth": "typing", "prediction": "up", "direction_candidate": "up", "reason": "accepted"}]
        result = summarize(rows)
        self.assertEqual(result["forced_direction_correct"], 0)
        self.assertEqual(result["accepted_swipes"], 0)
        self.assertIsNone(result["accepted_accuracy"])
        self.assertEqual(result["background_false_activations"], 1)

    def test_frozen_evaluation_and_original_models_preserved(self):
        folder = ROOT / "diagnostics/validation" / SESSION
        manifest = json.loads((folder / "manifest.json").read_text())
        plan = json.loads((ROOT / "diagnostics/calibration" / SESSION / "plan.json").read_text())
        self.assertEqual(hashlib.sha256((folder / "report.json").read_bytes()).hexdigest(), plan["validation_sha256"])
        self.assertEqual(hashlib.sha256((ROOT / "data" / SESSION / "session.json").read_bytes()).hexdigest(), manifest["session_json_sha256"])
        for info in manifest["models"].values():
            self.assertEqual(hashlib.sha256((ROOT / info["path"]).read_bytes()).hexdigest(), info["sha256"])

    def test_new_session_model_matches_block_holdout_and_excludes_test_hashes(self):
        folder = ROOT / "diagnostics/calibration" / SESSION
        report = json.loads((folder / "report.json").read_text())
        with np.load(folder / "block1_model.npz", allow_pickle=False) as model:
            self.assertEqual(len(model["training_sha256"]), 14)
            for row in report["block_evaluation"]["1_to_2"]["predictions"]:
                blob = (ROOT / "data" / SESSION / row["file"]).read_bytes()
                self.assertNotIn(hashlib.sha256(blob).hexdigest(), model["training_sha256"])
                samples, rate, _ = decode_wav(blob)
                result = classify(samples, rate, model)
                self.assertEqual(result["prediction"], row["prediction"])
                self.assertAlmostEqual(result["distance_margin"], row["margin"], places=9)


if __name__ == "__main__":
    unittest.main()
