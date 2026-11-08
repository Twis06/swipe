"""Regression checks for invariance, rejection, serialization and fold isolation."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from scipy.signal import resample_poly

from swipe.audio_format import decode_wav
from swipe.gesture_model import RATE, classify, decision, direction, extract
from research.improve_model import Experiment, SOURCE, OUT


class FeatureTests(unittest.TestCase):
    def signal(self):
        x = np.zeros(RATE * 2)
        t = np.arange(int(RATE * .12)) / RATE
        burst = np.sin(2 * np.pi * (700 * t + 3000 * t * t)) * np.hanning(len(t))
        x[RATE:RATE + len(t)] = burst
        return x

    def test_level_and_position_invariance(self):
        x = self.signal()
        for crop in ("energy", "noise_aware"):
            original = extract(x, RATE, crop)[0]
            np.testing.assert_allclose(original, extract(x * .25, RATE, crop)[0], atol=1e-10)
            shifted = np.r_[np.zeros(RATE // 5), x[:-RATE // 5]]
            np.testing.assert_allclose(original, extract(shifted, RATE, crop)[0], atol=1e-10)

    def test_48k_input_matches_44k(self):
        x = self.signal()
        original = extract(x, RATE, "noise_aware")[0]
        converted = extract(resample_poly(x, 160, 147), 48000, "noise_aware")[0]
        self.assertLess(float(np.mean(np.abs(original - converted))), .03)

    def test_background_noise_does_not_expand_event_to_whole_clip(self):
        x = self.signal()
        x += np.random.default_rng(42).normal(0, .02, len(x))
        self.assertGreater(extract(x, RATE, "energy")[2]["energy_duration_ms"], 500)
        self.assertLess(extract(x, RATE, "noise_aware")[2]["energy_duration_ms"], 250)

    def test_invalid_input_and_silence(self):
        for x, rate in ((np.zeros(20), RATE), (np.full(RATE, np.nan), RATE), (np.zeros(RATE), 4000)):
            with self.assertRaises(ValueError):
                extract(x, rate)
        sequence, gate, info = extract(np.zeros(RATE), RATE)
        self.assertTrue(info["silent"])
        self.assertTrue(np.isfinite(sequence).all() and np.isfinite(gate).all())

    def test_conflicting_neighbors_abstain(self):
        pred, margin, distance = direction(np.array([.1, .2, .3]), np.array(["up", "down", "down"]), 3)
        self.assertEqual(pred, "down")
        self.assertLess(margin, 0)
        self.assertEqual(decision(pred, margin, distance, 1, .99)[0], "uncertain")
        self.assertEqual(decision("up", .9, 2, 1, .99), ("uncertain", "unfamiliar_sound"))


class PilotIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = Experiment("noise_aware")

    def test_test_labels_cannot_change_fitted_model(self):
        e = self.experiment
        train = np.where(e.groups != "1_natural")[0]
        held = np.where((e.groups == "1_natural") & e.swipe)[0]
        first, scores_before = e.select(train)
        fit_before = e.fit(train, first)
        original = e.y.copy()
        try:
            e.y[held] = np.where(e.y[held] == "up", "down", "up")
            second, scores_after = e.select(train)
            fit_after = e.fit(train, second)
        finally:
            e.y = original
        self.assertEqual(first, second)
        self.assertEqual(scores_before, scores_after)
        self.assertEqual(fit_before["cutoff"], fit_after["cutoff"])
        np.testing.assert_array_equal(fit_before["gate"].named_steps["logisticregression"].coef_,
                                      fit_after["gate"].named_steps["logisticregression"].coef_)
        self.assertFalse(set(fit_before["refs"]) & set(held))

    def test_serialized_model_matches_held_out_evaluation(self):
        for folder in (OUT, OUT / "noise_aware"):
            report = json.loads((folder / "report.json").read_text())
            with np.load(folder / "block1_model.npz", allow_pickle=False) as model:
                for expected in report["block_evaluation"]["1_to_2"]["predictions"]:
                    samples, rate, _ = decode_wav((SOURCE / expected["file"]).read_bytes())
                    result = classify(samples, rate, model)
                    self.assertEqual(result["prediction"], expected["prediction"])
                    self.assertEqual(result["direction_candidate"], expected["direction_candidate"])
                    self.assertAlmostEqual(result["distance_margin"], expected["margin"], places=9)
                self.assertEqual(classify(np.zeros(RATE), RATE, model)["prediction"], "background")

    def test_cli_refuses_training_file(self):
        first = self.experiment.clips[0]["filename"]
        completed = subprocess.run([sys.executable, '-m', 'research.recognize_clip', str(SOURCE / first)],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("used to fit", completed.stderr)
        self.assertEqual(completed.stdout, "")


if __name__ == "__main__":
    unittest.main()
