import unittest
from direction_observability import references_for, summarize


class DirectionDiagnosticTests(unittest.TestCase):
    def test_references_exclude_whole_session(self):
        rows = [{'session': s, 'label': l, 'key': s+'/'+l} for s in ('a', 'b') for l in ('up', 'down')]
        references = references_for(rows[0], rows)
        self.assertEqual([r['key'] for r in references], ['b/up', 'b/down'])
        with self.assertRaises(ValueError):
            references_for(rows[0], rows[:2])
        with self.assertRaises(ValueError):
            references_for(rows[0], rows[:3])

    def test_reversal_is_counted_as_change_not_correct_opposite(self):
        rows = [
            {'label': 'up', 'original': {'candidates': {'1': 'up'}},
             'time_reversed_waveform': {'candidates': {'1': 'down'}}},
            {'label': 'down', 'original': {'candidates': {'1': 'up'}},
             'time_reversed_waveform': {'candidates': {'1': 'up'}}},
        ]
        result = summarize(rows, 1)
        self.assertEqual(result['original_correct'], 1)
        self.assertEqual(result['reversed_matches_original_label'], 0)
        self.assertEqual(result['prediction_changes_after_reversal'], 1)
        self.assertEqual(result['original_per_class']['down'], {'total': 1, 'correct': 0})


if __name__ == '__main__':
    unittest.main()
