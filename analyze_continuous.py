"""Unlabeled continuous-take audit; does not train or change the live model.

Run .venv/bin/python analyze_continuous.py. A burst is acoustic activity, not a
verified swipe. Worklet replay has real startup audio and no synthetic padding;
it excludes network latency and therefore cannot measure browser throughput.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/swipe-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, binary_opening

from audio_format import decode_wav, recording_info
from preview_model import predict
from preview_service import PreviewService

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'diagnostics/continuous'


def activity(envelope, multiplier):
    # Fixed 10 ms analysis frames. A lower quantile approximates quiet gaps;
    # this assumption can fail during uninterrupted friction. Report sensitivity.
    threshold = max(.003, float(np.quantile(envelope, .2)) * multiplier)
    mask = binary_closing(envelope > threshold, structure=np.ones(9))
    mask = binary_opening(mask, structure=np.ones(4))
    changes = np.diff(np.r_[False, mask, False].astype(int))
    intervals = [(float(a*.01), float(b*.01)) for a, b in
                 zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]
    return threshold, intervals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    correction = json.loads((OUT/'label_review.json').read_text())
    service = PreviewService()
    report = {'status': 'Unlabeled exploratory signal and detector audit; no direction accuracy estimate.',
              'label_review': correction,
              'model_sha256': service.config['model_sha256'],
              'worklet_sha256': hashlib.sha256((ROOT/'preview-worklet.js').read_bytes()).hexdigest(),
              'method': {'frame_seconds': .01, 'gap_close_seconds': .09,
                         'minimum_activity_seconds': .04, 'noise_quantile': .2,
                         'threshold_multipliers': [4, 6, 8],
                         'replay': 'Exact live worklet, no padding, initial one-second warmup uses real take audio. All emitted windows classified sequentially; HTTP latency and busy-drop behavior not simulated.'},
              'takes': []}
    fig, axes = plt.subplots(len(correction['take_ids']), 1, figsize=(13, 8), sharex=True, layout='constrained', squeeze=False)
    colors = {'up': '#237357', 'down': '#b85e36', 'uncertain': '#756b93', 'background': '#888888'}
    for number, (sid, ax) in enumerate(zip(correction['take_ids'], axes[:, 0]), 1):
        directory = ROOT/'data'/sid
        session = json.loads((directory/'session.json').read_text())
        blob = (directory/'continuous.wav').read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != session['clips'][0]['sha256']:
            raise ValueError(f'Audio checksum mismatch: {sid}')
        samples, rate, _ = decode_wav(blob)
        x = np.asarray(samples, dtype=np.float64)
        width = round(rate*.01)
        envelope = np.sqrt(np.mean(x[:len(x)//width*width].reshape(-1, width)**2, axis=1))
        sensitivity = {}
        for multiplier in (4, 6, 8):
            threshold, intervals = activity(envelope, multiplier)
            sensitivity[str(multiplier)] = {'threshold_rms': threshold, 'bursts': intervals, 'count': len(intervals)}
        bursts = sensitivity['6']['bursts']
        windows = []
        with tempfile.TemporaryDirectory(prefix='grille-continuous-replay-') as temp:
            subprocess.run(['node', str(ROOT/'replay_continuous.cjs'), str(directory/'continuous.wav'), temp], check=True)
            replay = json.loads((Path(temp)/'windows.json').read_text())
            last_end = float('-inf')
            for window in replay['windows']:
                pcm = np.fromfile(Path(temp)/window.pop('file'), dtype='<f4')
                result = predict(pcm, replay['rate'], service.model)
                suppressed = False
                if result['prediction'] != 'background':
                    start = window['start_seconds']+result['quality']['event_start_seconds']
                    end = window['start_seconds']+result['quality']['event_end_seconds']
                    suppressed = start <= last_end+.1
                    if not suppressed:
                        last_end = end
                overlap = [i for i, (a, b) in enumerate(bursts)
                           if min(b, window['end_seconds'])-max(a, window['start_seconds']) >= .04]
                windows.append({**window, **result, 'duplicate_suppressed': suppressed, 'overlapping_activity_bursts': overlap})
        shown = [w for w in windows if not w['duplicate_suppressed'] and w['prediction'] != 'background']
        row = {'id': sid, 'recorded_at': session['created_at'], 'sha256': digest,
               'original_mode': session['mode'], 'reviewed_mode': 'free',
               'quality': recording_info(blob, 15), 'activity_sensitivity': sensitivity,
               'live_replay': {'candidate_windows': len(windows), 'displayed_results': len(shown),
                               'accepted_directions': sum(w['prediction'] in ('up', 'down') for w in shown),
                               'suppressed_overlap': sum(w['duplicate_suppressed'] for w in windows),
                               'windows_overlapping_multiple_bursts': sum(len(w['overlapping_activity_bursts']) > 1 for w in windows),
                               'windows': windows}}
        report['takes'].append(row)
        t = (np.arange(len(envelope))+.5)*.01
        db = 20*np.log10(np.maximum(envelope, 1e-8))
        ax.plot(t, db, color='#243b35', linewidth=.8)
        ax.axhline(20*np.log10(sensitivity['6']['threshold_rms']), color='#92958e', linestyle='--', linewidth=.7)
        for a, b in bursts:
            ax.axvspan(a, b, color='#a4b5a6', alpha=.25)
        bottom = min(-70, float(db.min())-5)
        for i, window in enumerate(windows):
            y = bottom+4+(i%2)*7
            ax.plot([window['start_seconds'], window['end_seconds']], [y, y],
                    color=colors[window['prediction']], linewidth=3,
                    alpha=.3 if window['duplicate_suppressed'] else .9)
            ax.text((window['start_seconds']+window['end_seconds'])/2, y+1,
                    window['prediction']+(' (suppressed)' if window['duplicate_suppressed'] else ''),
                    ha='center', fontsize=7, color=colors[window['prediction']])
        ax.set_ylim(bottom, max(25, float(db.max())+3))
        ax.set_ylabel('10 ms RMS\n(dB re 1.0)')
        ax.set_title(f'Take {number}: 15 s | {len(bursts)} acoustic bursts at middle threshold | {len(windows)} live windows, {len(shown)} displayed results', loc='left', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1, 0].set_xlabel('Seconds from recording start')
    axes[-1, 0].set_xlim(0, 15)
    fig.suptitle('Three free-movement takes: signal activity and unchanged detector replay\nShading = estimated activity, not verified swipes. Colored bars = model windows, not true direction.', fontsize=12)
    fig.savefig(OUT/'overview.png', dpi=150)
    plt.close(fig)
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps([{'take': i+1, 'quality': r['quality'],
                      'burst_counts_4_6_8': [r['activity_sensitivity'][str(m)]['count'] for m in (4,6,8)],
                      'replay': {k:v for k,v in r['live_replay'].items() if k!='windows'}}
                     for i,r in enumerate(report['takes'])], indent=2))


if __name__ == '__main__':
    main()
