"""Bounded, training-free segmentation comparison; no live application changes.

Acoustic candidates are not physical swipes or direction labels. Whole-take
noise quantiles make this an offline experiment, not a causal streaming claim.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'diagnostics/continuous/segmentation'
STEP = .01
PARAMETERS = {
    'frame_seconds': STEP, 'noise_quantile': .2, 'absolute_rms_floor': .003,
    'baseline': {'close_frames': 9, 'open_frames': 4, 'multipliers': [4, 6, 8]},
    'hysteresis': {'on_multipliers': [4, 6, 8], 'off_fraction_of_on': .5,
                   'on_frames': 2, 'hangover_frames': 8, 'minimum_frames': 4},
    'spectral_boundary': {'band_edges_hz': [100, 250, 500, 1000, 2000, 4000, 8000, 16000, 24001],
                          'context_frames': 3, 'mad_multipliers': [3, 4, 5],
                          'minimum_distance_frames': 12, 'edge_guard_frames': 8},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(samples, rate):
    x = np.asarray(samples, dtype=np.float64)
    width = round(rate*STEP)
    x = x[:len(x)//width*width].reshape(-1, width)
    envelope = np.sqrt(np.mean(x*x, axis=1))
    spectrum = abs(np.fft.rfft(x*np.hanning(width), axis=1))**2
    frequency = np.fft.rfftfreq(width, 1/rate)
    edges = PARAMETERS['spectral_boundary']['band_edges_hz']
    bands = np.stack([spectrum[:, (frequency >= a) & (frequency < b)].sum(axis=1)
                      for a, b in zip(edges[:-1], edges[1:])], axis=1)
    # Spectral proportions suppress pure loudness differences, but amplify noise
    # when no sound is present. Activity gating below is therefore essential.
    proportions = bands/np.maximum(bands.sum(axis=1, keepdims=True), 1e-30)
    return envelope, np.sqrt(proportions)


def mask_intervals(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]


def baseline(envelope, multiplier=6):
    threshold = max(.003, float(np.quantile(envelope, .2))*multiplier)
    mask = binary_closing(envelope > threshold, structure=np.ones(9))
    mask = binary_opening(mask, structure=np.ones(4))
    return mask_intervals(mask)


def hysteresis(envelope, multiplier=6):
    high = max(.003, float(np.quantile(envelope, .2))*multiplier)
    low = high*.5
    active = False
    on_run = off_run = 0
    start = 0
    result = []
    for i, value in enumerate(envelope):
        if not active:
            on_run = on_run+1 if value > high else 0
            if on_run >= 2:
                start = i-on_run+1
                active = True
                off_run = 0
        else:
            off_run = off_run+1 if value < low else 0
            if off_run >= 8:
                end = i-off_run+1
                if end-start >= 4:
                    result.append((start, end))
                active = False
                on_run = off_run = 0
    if active:
        end = len(envelope)-off_run
        if end-start >= 4:
            result.append((start, end))
    return result


def spectral_boundaries(spectral_shape, intervals, mad_multiplier=4):
    # Two-sided adjacent context contrast: useful when sound continues through a
    # direction reversal. Boundaries remain hypotheses, not learned direction.
    n = len(spectral_shape)
    score = np.zeros(n)
    for i in range(3, n-3):
        left = spectral_shape[i-3:i].mean(axis=0)
        right = spectral_shape[i:i+3].mean(axis=0)
        score[i] = np.linalg.norm(left-right)
    eligible = np.zeros(n, dtype=bool)
    for a, b in intervals:
        eligible[a+8:max(a+8, b-8)] = True
    values = score[eligible]
    if not len(values):
        return [], score
    median = float(np.median(values))
    mad = float(np.median(abs(values-median)))
    threshold = median+mad_multiplier*max(mad, .01)
    peaks, _ = find_peaks(np.where(eligible, score, 0), height=threshold, distance=12)
    return peaks.astype(int).tolist(), score


def split_at(intervals, boundaries):
    result = []
    for a, b in intervals:
        cuts = [a]+[p for p in boundaries if a < p < b]+[b]
        result.extend(zip(cuts[:-1], cuts[1:]))
    return [(int(a), int(b)) for a, b in result]


def evaluate(samples, rate):
    envelope, shape = frames(samples, rate)
    result = {}
    for k in (4, 6, 8):
        for name, method in [('baseline', baseline), ('hysteresis', hysteresis)]:
            result[f'{name}_{k}'] = method(envelope, k)
    for k in (3, 4, 5):
        boundaries, _ = spectral_boundaries(shape, result['hysteresis_6'], k)
        result[f'spectral_split_{k}'] = split_at(result['hysteresis_6'], boundaries)
    return envelope, {name: {'count': len(intervals),
                            'intervals_seconds': [[round(a*STEP, 3), round(b*STEP, 3)] for a,b in intervals],
                            'durations_seconds': [round((b-a)*STEP, 3) for a,b in intervals]}
                      for name, intervals in result.items()}


def main():
    from swipe.audio_format import decode_wav
    OUT.mkdir(parents=True, exist_ok=True)
    reviewed = json.loads((ROOT/'diagnostics/continuous/label_review.json').read_text())
    continuous_ids = reviewed['take_ids']
    isolated_ids = ['20261103T165139_cf7201', '20261103T200050_87dcb0']
    inputs = []
    for sid in continuous_ids+isolated_ids:
        folder = ROOT/'data'/sid
        inputs.append(folder/'session.json')
        session = json.loads((folder/'session.json').read_text())
        inputs.extend(folder/c['filename'] for c in session['clips'] if c['accepted'])
    protected = inputs+[ROOT/'models/preview/model.json', ROOT/'models/preview/model.npz', ROOT/'research/worklets/preview-worklet.js']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    plan = {'status': 'Fixed methods and parameters, written before analyzing audio. No fitting or method selection.',
            'parameters': PARAMETERS, 'input_sha256': hashes,
            'source_sha256': sha(Path(__file__)),
            'interpretation': 'Count/sensitivity diagnostic only. Continuous boundaries/directions are unverified. Isolated clips have intended class labels but no verified boundaries. Old PCM16 session contains clipping.',
            'prohibited': ['training', 'parameter optimization', 'using cue times as labels', 'modifying raw audio or live model']}
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    rows = []
    envelopes = []
    for sid in continuous_ids+isolated_ids:
        folder = ROOT/'data'/sid
        session = json.loads((folder/'session.json').read_text())
        for clip in session['clips']:
            if not clip['accepted']:
                continue
            samples, rate, _ = decode_wav((folder/clip['filename']).read_bytes())
            envelope, methods = evaluate(samples, rate)
            row = {'session': sid, 'file': clip['filename'], 'kind': session.get('kind', 'isolated'),
                   'intended_label': clip['label'], 'methods': methods}
            rows.append(row)
            if sid in continuous_ids:
                envelopes.append(envelope)
    names = list(rows[0]['methods'])
    summary = {}
    for sid in isolated_ids:
        subset = [r for r in rows if r['session'] == sid]
        classes = {}
        for label in ('swipe', 'idle', 'typing'):
            group = [r for r in subset if (r['intended_label'] in ('up', 'down') if label == 'swipe' else r['intended_label'] == label)]
            classes[label] = {name: {'clips': len(group), 'zero_candidates': sum(r['methods'][name]['count'] == 0 for r in group),
                                      'one_candidate': sum(r['methods'][name]['count'] == 1 for r in group),
                                      'multiple_candidates': sum(r['methods'][name]['count'] > 1 for r in group)}
                              for name in names}
        summary[sid] = classes
    unchanged = all(sha(ROOT/p) == digest for p,digest in hashes.items())
    if not unchanged:
        raise RuntimeError('A protected source changed during analysis.')
    report = {'status': 'Exploratory training-free comparison; no ranking by accuracy is possible.',
              'protected_files_unchanged': unchanged, 'parameters': PARAMETERS,
              'continuous_counts': [{'session': r['session'], 'counts': {n:m['count'] for n,m in r['methods'].items()}}
                                    for r in rows if r['kind'] == 'continuous'],
              'isolated_sanity_checks': summary, 'records': rows,
              'limits': ['Segmentation can split one physical swipe into friction bursts or merge reversals.',
                         'A two-sided spectral change is not proof of a direction reversal.',
                         'Whole-recording noise quantiles are offline; no streaming latency result.',
                         'Idle/typing activity candidates do not imply classifier false activations.',
                         'No classifiers fitted or invoked; no direction accuracy measured.']}
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/swipe-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, layout='constrained')
    selected = [('baseline_6', 'Baseline activity', '#80936f'), ('hysteresis_6', 'Hysteresis', '#34756c'), ('spectral_split_4', '+ spectral changes', '#a26545')]
    continuous = [r for r in rows if r['kind'] == 'continuous']
    for number, (ax, envelope, row) in enumerate(zip(axes, envelopes, continuous), 1):
        ax.plot((np.arange(len(envelope))+.5)*STEP, 20*np.log10(np.maximum(envelope, 1e-8)), color='#34453b', linewidth=.7)
        for lane, (name, title, color) in enumerate(selected):
            y = -66-lane*10
            for a, b in row['methods'][name]['intervals_seconds']:
                ax.plot([a,b], [y,y], color=color, linewidth=4, solid_capstyle='butt')
            ax.text(15.05, y, f'{title}: {row["methods"][name]["count"]}', color=color, va='center', fontsize=9)
        ax.set_ylim(-93, 30)
        ax.set_xlim(0, 18)
        ax.set_xticks(np.arange(0, 16, 2))
        ax.set_ylabel('RMS dB re 1.0')
        ax.set_title(f'Take {number} — candidate intervals, not verified movements', loc='left', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xlabel('Seconds')
    fig.suptitle('Training-free segmentation diagnostics\nParameters fixed before this run; fewer or more candidates does not establish better accuracy.', fontsize=12)
    fig.savefig(OUT/'comparison.png', dpi=140)
    plt.close(fig)
    print(json.dumps({'continuous_counts': report['continuous_counts'], 'isolated_sanity_checks': summary,
                      'protected_files_unchanged': unchanged}, indent=2))


if __name__ == '__main__':
    main()
