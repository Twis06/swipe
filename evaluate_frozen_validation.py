"""Evaluate a complete prospective session using the pre-collection snapshot.

No fitting, threshold selection, or modification of recordings or model files.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
VALIDATION = ROOT/'diagnostics/prospective_validation/20261103_guided_validation_01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, engine):
    swipes = [r for r in rows if r['truth'] in ('up', 'down')]
    negatives = [r for r in rows if r['truth'] not in ('up', 'down')]
    accepted = [r for r in swipes if r[engine]['prediction'] in ('up', 'down')]
    return {
        'swipes': len(swipes),
        'correct_accepted': sum(r[engine]['prediction'] == r['truth'] for r in accepted),
        'wrong_accepted': sum(r[engine]['prediction'] != r['truth'] for r in accepted),
        'rejected_swipes': len(swipes)-len(accepted),
        'negative_clips': len(negatives),
        'negative_activations': sum(r[engine]['prediction'] in ('up', 'down') for r in negatives),
        'confusion': {truth: dict(Counter(r[engine]['prediction'] for r in rows if r['truth'] == truth))
                      for truth in ('up', 'down', 'idle', 'typing')},
        'rejection_reasons': dict(Counter(r[engine]['reason'] for r in swipes if r not in accepted)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('session')
    sid = parser.parse_args().session
    if not re.fullmatch(r'\d{8}T\d{6}_[a-f0-9]+', sid):
        raise ValueError('Expected isolated-session ID.')
    plan = json.loads((VALIDATION/'plan.json').read_text())
    repeat = json.loads((VALIDATION/'repeat_plan.json').read_text())
    folder = ROOT/'data'/sid
    session_path = folder/'session.json'
    session = json.loads(session_path.read_text())
    if sid in plan['existing_sessions_excluded'] or sid == repeat['excluded_session']:
        raise ValueError('This session is excluded from prospective validation.')
    if datetime.fromisoformat(session['created_at']) <= datetime.fromisoformat(plan['created_at_utc']):
        raise ValueError('Session predates the frozen plan.')
    metadata = session['metadata']
    if metadata['microphone_label'] != 'MacBook Pro Microphone (Built-in)':
        raise ValueError('Expected explicit MacBook built-in microphone.')
    if metadata['audio_settings'].get('deviceId') in (None, '', 'default', 'communications'):
        raise ValueError('Expected explicitly selected device.')
    clips = sorted((c for c in session['clips'] if c['accepted']), key=lambda c: c['trial_index'])
    if len(session['plan']) != 28 or [c['trial_index'] for c in clips] != list(range(28)):
        raise ValueError('Session must contain all 28 distinct accepted trials.')
    if Counter(c['label'] for c in clips) != {'up': 10, 'down': 10, 'idle': 4, 'typing': 4}:
        raise ValueError('Unexpected label counts.')
    protected = {str(session_path.relative_to(ROOT)): sha(session_path)}
    for clip in clips:
        if any(clip[k] != session['plan'][clip['trial_index']][k] for k in ('label', 'block', 'pace')):
            raise ValueError('Clip labels do not match the recording plan.')
        path = folder/clip['filename']
        if path.parent != folder or sha(path) != clip['sha256']:
            raise ValueError('Clip path/checksum mismatch.')
        protected[str(path.relative_to(ROOT))] = sha(path)
    for name, expected in plan['frozen_sha256'].items():
        path = VALIDATION/'frozen'/name
        if sha(path) != expected:
            raise ValueError(f'Frozen implementation/model changed: {name}')
        protected[str(path.relative_to(ROOT))] = expected
    out = VALIDATION/'evaluations'/sid
    out.mkdir(parents=True, exist_ok=False)
    execution = {'session': sid, 'original_plan_sha256': sha(VALIDATION/'plan.json'),
                 'repeat_plan_sha256': sha(VALIDATION/'repeat_plan.json'), 'script_sha256': sha(Path(__file__)),
                 'protected_sha256': protected, 'fitting': False}
    (out/'execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    sys.path.insert(0, str(VALIDATION/'frozen'))
    from preview_service import PreviewService
    from audio_format import decode_wav
    from fast_preview import predict
    service = PreviewService()
    rows = []
    for clip in clips:
        blob = (folder/clip['filename']).read_bytes()
        samples, rate, encoding = decode_wav(blob, array=True)
        if encoding != 'float32' or rate != metadata['capture_sample_rate']:
            raise ValueError('Encoding/sample rate differs from session metadata.')
        active = service.classify(blob)
        raw = predict(samples, rate, service.model)
        rows.append({'file': clip['filename'], 'truth': clip['label'], 'block': clip['block'],
                     'pace': clip['pace'], 'sha256': clip['sha256'], 'active': active, 'raw': raw,
                     'capture_quality': {k: clip.get(k) for k in ('peak', 'rms', 'possible_clipping', 'quality_warning')}})
    report = {'session': sid, 'status': 'Frozen prospective evaluation; no fitting or tuning on this session.',
              'microphone': metadata['microphone_label'], 'sample_rate': metadata['capture_sample_rate'],
              'accepted_clips': len(clips), 'excluded_takes': len(session['clips'])-len(clips),
              'summary': {e: summarize(rows, e) for e in ('active', 'raw')},
              'by_block': {str(b): summarize([r for r in rows if r['block'] == b], 'active') for b in (1, 2)},
              'by_pace': {p: summarize([r for r in rows if r['pace'] == p], 'active') for p in ('natural', 'slow', 'brisk')},
              'fallback_changes': [r for r in rows if r['active']['prediction'] != r['raw']['prediction']],
              'predictions': rows, 'execution_sha256': sha(out/'execution.json'),
              'limits': ['One user and one new session; small sample.', 'Labels follow cues and user completion, not independent motion verification.',
                         'Isolated two-second clip results do not measure continuous capture accuracy.',
                         'Further model changes using this session make subsequent evaluations on it reused-data diagnostics.']}
    report['protected_unchanged'] = all(sha(ROOT/p) == h for p, h in protected.items())
    assert report['protected_unchanged']
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('predictions', 'fallback_changes')}, indent=2))


if __name__ == '__main__':
    main()
