"""No-fit cross-session direction diagnostic on already-inspected isolated clips.

This is descriptive template retrieval, not an independent accuracy estimate.
Reverse playback is a sensitivity control, not the opposite physical swipe.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from swipe.audio_format import decode_wav
from swipe.gesture_model import direction, dtw
from swipe.preview_model import features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'diagnostics/continuous/direction_observability'
SESSIONS = ['20261103T165139_cf7201', '20261103T200050_87dcb0']
PARAMETERS = {'feature_input': 'raw', 'dtw_radius': 4, 'neighbors': [1, 3],
              'query_variants': ['original', 'time_reversed_waveform'],
              'rejection': 'None: all outputs are forced candidates, not accepted live predictions.'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def references_for(query, records):
    references = [r for r in records if r['session'] != query['session']]
    if not references or any(r['key'] == query['key'] for r in references):
        raise ValueError('References must be from a distinct session and exclude the query.')
    if set(r['label'] for r in references) != {'up', 'down'}:
        raise ValueError('Cross-session references must contain both directions.')
    return references


def summarize(rows, k):
    name = str(k)
    original = [r['original']['candidates'][name] for r in rows]
    reversed_ = [r['time_reversed_waveform']['candidates'][name] for r in rows]
    return {'queries': len(rows),
            'original_correct': sum(p == r['label'] for p, r in zip(original, rows)),
            'reversed_matches_original_label': sum(p == r['label'] for p, r in zip(reversed_, rows)),
            'prediction_changes_after_reversal': sum(a != b for a, b in zip(original, reversed_)),
            'original_per_class': {label: {'total': sum(r['label'] == label for r in rows),
                'correct': sum(r['label'] == label and p == label for p, r in zip(original, rows))}
                for label in ('up', 'down')}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    protected = [ROOT/'swipe/preview_model.py', ROOT/'swipe/gesture_model.py', ROOT/'research/worklets/preview-worklet.js',
                 ROOT/'models/preview/model.npz', ROOT/'models/preview/model.json']
    for sid in SESSIONS:
        folder = ROOT/'data'/sid
        protected.append(folder/'session.json')
        session = json.loads((folder/'session.json').read_text())
        for clip in session['clips']:
            if clip['accepted'] and clip['label'] in ('up', 'down'):
                path = folder/clip['filename']
                protected.append(path)
                records.append({'session': sid, 'file': clip['filename'], 'key': sid+'/'+clip['filename'],
                                'label': clip['label'], 'path': path})
    counts = {sid: {label: sum(r['session'] == sid and r['label'] == label for r in records)
                    for label in ('up', 'down')} for sid in SESSIONS}
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    plan = {'status': 'Parameters fixed before this run. No fitting, optimization, or method selection.',
            'parameters': PARAMETERS, 'source_sha256': sha(Path(__file__)), 'input_sha256': hashes,
            'class_counts': counts,
            'limits': ['Both sessions have already been inspected and used in prior model development.',
                       'The first session is clipped PCM16; the second is Float32. Session and encoding are confounded.',
                       'A reversed recording is not verified audio of the opposite physical direction.',
                       'Query references exclude its entire session. No full-bank self-match is possible.',
                       'All directions are forced nearest-template candidates; no calibrated probability or live acceptance is measured.']}
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    for record in records:
        samples, rate, _ = decode_wav(record['path'].read_bytes())
        record['original'] = features(samples, rate, 'raw')[0]
        record['reversed'] = features(np.asarray(samples)[::-1], rate, 'raw')[0]
    rows = []
    for query in records:
        references = references_for(query, records)
        labels = np.array([r['label'] for r in references])
        row = {key: query[key] for key in ('session', 'file', 'key', 'label')}
        row['reference_keys'] = [r['key'] for r in references]
        for variant, source in [('original', 'original'), ('time_reversed_waveform', 'reversed')]:
            distances = np.array([dtw(query[source], r['original'], 4) for r in references])
            same = distances[labels == query['label']]
            other = distances[labels != query['label']]
            nearest = np.argsort(distances, kind='stable')[:3]
            row[variant] = {'candidates': {str(k): direction(distances, labels, k)[0] for k in (1, 3)},
                            'nearest_same_class_distance': float(same.min()),
                            'nearest_other_class_distance': float(other.min()),
                            'median_same_class_distance': float(np.median(same)),
                            'median_other_class_distance': float(np.median(other)),
                            'nearest_three': [{'key': references[i]['key'], 'label': references[i]['label'],
                                               'distance': float(distances[i])} for i in nearest]}
        rows.append(row)
    # Equal class representation in every stratum, unique pairs, no self-pairs.
    pair_distances = {(session_relation, label_relation): []
                      for session_relation in ('same_session', 'cross_session')
                      for label_relation in ('same_label', 'opposite_label')}
    for i, a in enumerate(records):
        for b in records[:i]:
            key = ('same_session' if a['session'] == b['session'] else 'cross_session',
                   'same_label' if a['label'] == b['label'] else 'opposite_label')
            pair_distances[key].append(dtw(a['original'], b['original'], 4))
    distance_summary = {'/'.join(key): {'pairs': len(values), 'median_distance': float(np.median(values)),
                        'interquartile_distance': list(map(float, np.quantile(values, [.25, .75])))}
                        for key, values in pair_distances.items()}
    summary = {sid: {str(k): summarize([r for r in rows if r['session'] == sid], k) for k in (1, 3)}
               for sid in SESSIONS}
    unchanged = all(sha(ROOT/path) == expected for path, expected in hashes.items())
    assert unchanged, 'A protected file changed during analysis'
    report = {'status': 'Descriptive, already-inspected data. No classifier fitted; no independent validation.',
              'plan_sha256': sha(OUT/'plan.json'), 'parameters': PARAMETERS, 'class_counts': counts,
              'summary_by_query_session': summary, 'pair_distance_summary': distance_summary,
              'records': rows, 'protected_files_unchanged': unchanged, 'limits': plan['limits']}
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'summary': summary, 'pair_distances': distance_summary,
                      'protected_files_unchanged': unchanged}, indent=2))


if __name__ == '__main__':
    main()
