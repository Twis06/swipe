#!/usr/bin/env python3
"""Reproducible pilot check. Requires numpy; preserves all source recordings.

Primary direction model: standardized spectral/envelope features, ridge binary
classifier (fixed lambda=10). Fit block 1 only; evaluate block 2 once. These
features and regularization are not selected using held-out labels.
"""
import argparse
import hashlib
import json
import wave
from pathlib import Path
import numpy as np

BANDS = np.array([100, 250, 500, 1000, 2000, 4000, 8000, 16000, 22051])

def read_wav(path):
    with wave.open(str(path), 'rb') as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2')
    return pcm.astype(np.float64) / 32768, rate, pcm

def features(x, rate):
    # Whole-clip spectral shape plus normalized temporal envelope. Absolute
    # loudness is excluded from the primary direction model.
    frames = np.lib.stride_tricks.sliding_window_view(x, 1024)[::512]
    power = abs(np.fft.rfft(frames * np.hanning(1024), axis=1)) ** 2
    frequencies = np.fft.rfftfreq(1024, 1 / rate)
    bands = np.stack([power[:, (frequencies >= lo) & (frequencies < hi)].sum(axis=1)
                      for lo, hi in zip(BANDS[:-1], BANDS[1:])], axis=1)
    sums = bands.sum(axis=0)
    shape = np.log10((sums + 1e-12) / (sums.sum() + 1e-12))
    energy = np.array([np.mean(part ** 2) for part in np.array_split(x, 4)])
    envelope = energy / (energy.sum() + 1e-12)
    return np.concatenate([shape, envelope])

def fit_ridge(X, y):
    mean, scale = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-6)
    z = (X - mean) / scale
    ymean = y.mean()
    weights = np.linalg.solve(z.T @ z + 10 * np.eye(z.shape[1]), z.T @ (y - ymean))
    return mean, scale, weights, ymean

def predict(model, X):
    mean, scale, weights, intercept = model
    return np.where((X - mean) / scale @ weights + intercept >= 0, 'up', 'down')

def confusion(truth, predictions, labels):
    return [[sum(t == a and p == b for t, p in zip(truth, predictions)) for b in labels] for a in labels]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=Path)
    args = parser.parse_args()
    folder = args.session
    s = json.loads((folder / 'session.json').read_text())
    clips = [c for c in s['clips'] if c['accepted']]
    rows, X = [], []
    for c in clips:
        path = folder / c['filename']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == c['sha256'], f'Hash mismatch: {path}'
        x, rate, pcm = read_wav(path)
        assert len(x) == c['frames'] and rate == c['sample_rate']
        X.append(features(x, rate))
        rows.append({'filename': c['filename'], 'label': c['label'], 'block': c['block'],
                     'pace': c.get('pace'), 'duration_seconds': len(x) / rate,
                     'rms_dbfs': float(20*np.log10(np.sqrt(np.mean(x*x)) + 1e-12)),
                     'clipped_fraction': float(np.mean((pcm <= -32767) | (pcm >= 32766)))})
    X = np.array(X)
    blocks = np.array([r['block'] for r in rows])
    labels = np.array([r['label'] for r in rows])
    swipes = np.isin(labels, ['up', 'down'])
    train, test = (blocks == 1) & swipes, (blocks == 2) & swipes
    assert train.sum() >= 4 and test.sum() >= 4, 'Need gestures in both blocks.'
    model = fit_ridge(X[train], np.where(labels[train] == 'up', 1., -1.))
    pred = predict(model, X[test])
    directional = {'train_block':1,'test_block':2,'train_clips':int(train.sum()),'test_clips':int(test.sum()),
                   'correct':int(np.sum(pred == labels[test])), 'accuracy':float(np.mean(pred == labels[test])),
                   'labels':['up','down'],'confusion_rows_true_columns_predicted':confusion(labels[test],pred,['up','down']),
                   'predictions':[{'file':r['filename'],'true':r['label'],'predicted':str(p)} for r,p in zip([r for r,m in zip(rows,test) if m],pred)]}
    rms = np.array([r['rms_dbfs'] for r in rows])
    # One scalar baseline, kept separate: could the result just reflect pressure?
    rms_model = fit_ridge(rms[train,None], np.where(labels[train] == 'up',1.,-1.))
    amp_pred = predict(rms_model,rms[test,None])
    # Training-only threshold for swipe activity vs idle/typing.
    cutoff = float((rms[(blocks==1)&swipes].min() + rms[(blocks==1)&~swipes].max())/2)
    activity_pred = rms[blocks==2] > cutoff
    activity_true = swipes[blocks==2]
    report = {'session':s['id'],'source_integrity':'All WAV hashes and frame counts match metadata.',
              'accepted_clips':len(rows),'counts':{k:int(np.sum(labels==k)) for k in sorted(set(labels))},
              'audio':{'sample_rates':sorted(set(c['sample_rate'] for c in clips)), 'total_seconds':sum(r['duration_seconds'] for r in rows),
                       'swipe_clips_with_clipping':sum(r['clipped_fraction']>0 for r,m in zip(rows,swipes) if m),
                       'swipe_clipped_fraction_range':[float(min(r['clipped_fraction'] for r,m in zip(rows,swipes) if m)),float(max(r['clipped_fraction'] for r,m in zip(rows,swipes) if m))]},
              'activity_baseline':{'threshold_dbfs_fit_on_block1':cutoff,'test_block':2,'test_clips':int(np.sum(blocks==2)),
                                   'correct':int(np.sum(activity_pred==activity_true)),
                                   'false_positives':int(np.sum(activity_pred & ~activity_true)),
                                   'false_negatives':int(np.sum(~activity_pred & activity_true))},
              'direction_model':directional,
              'loudness_only_direction_baseline':{'correct':int(np.sum(amp_pred==labels[test])),'test_clips':int(test.sum())},
              'method':{'features':'8 log relative spectral bands (100 Hz–22.05 kHz) + 4 normalized quarter-clip energy values',
                        'model':'Standardized ridge binary classifier, lambda=10, training-set scaling only',
                        'split':'Block 1 training, block 2 held out. No parameter search.',
                        'limits':['Only one session and one user. Blocks are not independent sessions.',
                                  '10 direction test examples; results are preliminary.',
                                  'All swipe clips have digital saturation; spectral features may reflect clipping.',
                                  'Prompt labels are not independently verified gestures.',
                                  'Quiet/typing samples do not measure real-world false alarms.']},
              'clips':rows}
    output = folder / 'analysis'
    output.mkdir(exist_ok=True)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    np.savez(output/'block1_direction_model.npz',mean=model[0],scale=model[1],weights=model[2],intercept=model[3],bands=BANDS)
    print(json.dumps({k:v for k,v in report.items() if k!='clips'},indent=2))

if __name__ == '__main__':
    main()
