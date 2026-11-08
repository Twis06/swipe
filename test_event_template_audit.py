import unittest
import numpy as np
from event_template_audit import isolate, real_context, consistency, boundary_stress


class EventAuditTests(unittest.TestCase):
    def test_isolation_preserves_samples_and_excludes_neighbor(self):
        x = np.arange(4000, dtype=float)
        intervals = [(1, 1.1), (1.16, 1.3)]
        pcm, meta = isolate(x, 1000, intervals, 0, .1)
        self.assertAlmostEqual(meta['source_end'], 1.13)
        self.assertEqual(len(pcm), 2000)
        offset = round(meta['destination_start']*1000)
        np.testing.assert_array_equal(pcm[offset:offset+230], x[900:1130])
        self.assertTrue(np.all(pcm[:offset] == 0))

    def test_edge_real_context_is_not_padded(self):
        x = np.arange(3000)
        pcm, meta = real_context(x, 1000, [(.01, .1), (.5, .6)], 0)
        np.testing.assert_array_equal(pcm, x[:2000])
        self.assertEqual(meta['overlapping_event_ids'], [0, 1])

    def test_stable_candidate_is_not_stable_acceptance(self):
        rows = [{'direction_candidate': 'up', 'prediction': p} for p in ['up', 'uncertain', 'up']]
        result = consistency(rows)
        self.assertTrue(result['candidate_direction_stable'])
        self.assertFalse(result['strict_accepted_direction_stable'])
        rows[1] = {'direction_candidate': 'down', 'prediction': 'down'}
        self.assertTrue(consistency(rows)['accepted_direction_flip'])

    def test_long_interval_is_never_truncated_to_fit(self):
        with self.assertRaises(ValueError):
            isolate(np.zeros(4000), 1000, [(0, 3)], 0, .1)

    def test_boundary_stress_removes_actual_samples(self):
        x = np.arange(3000)
        variants = boundary_stress(x, 1000, [(1, 1.1)], 0)
        self.assertEqual([v[2]['source_duration'] for v in variants], [.1, .08, .08])
        for (_, pcm, meta), source in zip(variants, [x[1000:1100], x[1020:1100], x[1000:1080]]):
            offset = round(meta['destination_start']*1000)
            np.testing.assert_array_equal(pcm[offset:offset+len(source)], source)


if __name__ == '__main__':
    unittest.main()
