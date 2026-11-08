"""Bounded deterministic feature-transfer probe; no model fitting or deployment.

Both sessions were inspected during prior work. Scores describe reused-data
template retrieval, not fresh validation. Transforms change distance units, so
existing rejection cutoffs cannot simply be carried across to a winner.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from audio_format import decode_wav
from gesture_model import direction, dtw
from preview_model import features

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'diagnostics/robust_features'
SESSIONS = ['20261103T165139_cf7201', '20261103T200050_87dcb0']
FORMULAS = {
    'raw': 'Unchanged 32x8 raw preview log-band sequence.',
    'frame_shape': 'Subtract mean across 8 bands independently at each frame.',
    'band_trajectory': 'Subtract mean across 32 frames independently for each band.',
    'shape_delta': 'Concatenate frame_shape with twice its first temporal gradient; 32x16.',
    'temporal_rank': 'Average-tie temporal ranks independently per band, scaled (rank-1)/31 - 0.5.',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(sequence, name):
    x = np.asarray(sequence, dtype=np.float64)
    if x.shape != (32, 8) or not np.isfinite(x).all():
        raise ValueError('Expected finite 32x8 feature sequence.')
    if name == 'raw':
        return x.copy()
    if name == 'frame_shape':
        return x-x.mean(axis=1, keepdims=True)
    if name == 'band_trajectory':
        return x-x.mean(axis=0, keepdims=True)
    if name == 'shape_delta':
        shape = transform(x, 'frame_shape')
        return np.concatenate([shape, 2*np.gradient(shape, axis=0)], axis=1)
    if name == 'temporal_rank':
        return (rankdata(x, axis=0, method='average')-1)/(len(x)-1)-.5
    raise ValueError('Unknown fixed transform.')


def reference_indices(query_index, records, split):
    query = records[query_index]
    key = 'session' if split == 'cross_session' else 'group' if split == 'leave_group_out' else None
    if key is None:
        raise ValueError('Unknown split.')
    result = np.array([i for i, record in enumerate(records) if record[key] != query[key]], dtype=int)
    if query_index in result or not len(result):
        raise ValueError('Invalid reference exclusion.')
    return result


def summarize(rows):
    return {'correct': sum(r['prediction'] == r['label'] for r in rows), 'total': len(rows),
            'by_class': {label: {'correct': sum(r['prediction'] == label and r['label'] == label for r in rows),
                                  'total': sum(r['label'] == label for r in rows)} for label in ('up', 'down')}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'plan.json').exists():
        raise RuntimeError('Fixed run already exists; do not overwrite its plan/results.')
    records = []
    protected = [ROOT/'preview_model.py', ROOT/'gesture_model.py', ROOT/'preview-worklet.js',
                 ROOT/'diagnostics/preview/model.json', ROOT/'diagnostics/preview/model.npz']
    for sid in SESSIONS:
        session_path = ROOT/'data'/sid/'session.json'
        protected.append(session_path)
        session = json.loads(session_path.read_text())
        for clip in session['clips']:
            if clip['accepted'] and clip['label'] in ('up', 'down'):
                path = session_path.parent/clip['filename']
                protected.append(path)
                records.append({'session': sid, 'file': clip['filename'], 'label': clip['label'],
                                'group': f'{sid}_{clip["block"]}_{clip["pace"]}',
                                'path': path, 'sha256': clip['sha256']})
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    plan = {'status': 'Five transforms fixed before feature extraction. No fitting, optimization, or expanded search.',
            'formulas': FORMULAS, 'radius': 4, 'neighbors': 3,
            'splits': ['Entire query session excluded', 'Entire session/block/pace group excluded'],
            'rejection': 'None. Forced candidates only; transformed distances require separately justified rejection.',
            'eligibility': 'Cross-session correct >= raw in BOTH directions, strictly greater total. Also report whether pooled group score regresses.',
            'source_sha256': sha(Path(__file__)), 'input_sha256': hashes,
            'limits': ['All recordings previously inspected; choosing a best transform is exploratory selection, not independent validation.',
                       'The old session is clipped PCM16 and the new session Float32; encoding is confounded with session.',
                       'No continuous take or cue timing enters this comparison.',
                       'No threshold or live model changes; all transformations are fixed formulas.']}
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    for record in records:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Source audio differs from session checksum.')
        samples, rate, _ = decode_wav(record['path'].read_bytes())
        record['features'] = features(samples, rate, 'raw')[0]
    labels = np.array([r['label'] for r in records])
    report = {'status': 'Exploratory deterministic template retrieval on reused recordings; no trained or deployed model.',
              'plan_sha256': sha(OUT/'plan.json'), 'methods': {}, 'limits': plan['limits']}
    for name in FORMULAS:
        sequences = [transform(r['features'], name) for r in records]
        distances = np.full((len(records), len(records)), np.inf)
        for i in range(len(records)):
            for j in range(i):
                distances[i, j] = distances[j, i] = dtw(sequences[i], sequences[j], 4)
        splits = {}
        for split in ('cross_session', 'leave_group_out'):
            rows = []
            for i, record in enumerate(records):
                refs = reference_indices(i, records, split)
                prediction, margin, nearest_distance = direction(distances[i, refs], labels[refs], 3)
                nearest = refs[np.argsort(distances[i, refs], kind='stable')[:3]]
                rows.append({k: record[k] for k in ('session', 'file', 'label', 'group')}
                            | {'prediction': prediction, 'margin': margin, 'nearest_distance': nearest_distance,
                               'reference_indices': refs.tolist(), 'nearest_three_indices': nearest.tolist()})
            key = 'session' if split == 'cross_session' else 'group'
            folds = {value: summarize([r for r in rows if r[key] == value]) for value in sorted({r[key] for r in rows})}
            splits[split] = {'summary': summarize(rows), 'folds': folds, 'predictions': rows}
        report['methods'][name] = splits
    baseline = report['methods']['raw']
    eligible = []
    for name, result in report['methods'].items():
        nondegrade = all(result['cross_session']['folds'][sid]['correct'] >= baseline['cross_session']['folds'][sid]['correct'] for sid in SESSIONS)
        gain = result['cross_session']['summary']['correct'] > baseline['cross_session']['summary']['correct']
        result['eligible_cross_session_candidate'] = name != 'raw' and nondegrade and gain
        result['pooled_group_score_nondegrading'] = result['leave_group_out']['summary']['correct'] >= baseline['leave_group_out']['summary']['correct']
        if result['eligible_cross_session_candidate']:
            eligible.append(name)
    unchanged = all(sha(ROOT/path) == expected for path, expected in hashes.items())
    if not unchanged:
        raise RuntimeError('A protected source changed during the run.')
    report['protected_unchanged'] = unchanged
    report['eligible_candidates'] = eligible
    report['record_order'] = [{k:r[k] for k in ('session','file','label','group')} for r in records]
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    summary = {name: {'cross_session': result['cross_session']['summary'],
                      'session_correct': {sid: result['cross_session']['folds'][sid]['correct'] for sid in SESSIONS},
                      'grouped': result['leave_group_out']['summary'],
                      'eligible': result['eligible_cross_session_candidate'],
                      'grouped_nondegrading': result['pooled_group_score_nondegrading']}
               for name, result in report['methods'].items()}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
