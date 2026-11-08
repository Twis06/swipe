"""Frozen-template, offline boundary-context sensitivity audit. No fitting.

Isolated zero-filled inputs change noise and gate statistics. Neither acceptance
nor agreement is accuracy. Event IDs are fixed to the earlier hysteresis6 output.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from audio_format import decode_wav
from gesture_model import dtw
from preview_model import features, predict
from preview_service import PreviewService

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'diagnostics/continuous/event_templates'
PADS = (.03, .06, .10)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def isolate(samples, rate, intervals, index, padding):
    """Copy actual context into a centered 2 s array; never rescale/stretch."""
    x = np.asarray(samples, dtype=np.float64)
    a, b = intervals[index]
    duration = len(x)/rate
    lower = (intervals[index-1][1]+a)/2 if index else 0
    upper = (b+intervals[index+1][0])/2 if index+1 < len(intervals) else duration
    left, right = max(lower, a-padding, 0), min(upper, b+padding, duration)
    start, end = max(0, round(left*rate)), min(len(x), round(right*rate))
    source = x[start:end]
    n = round(rate*2)
    if not 0 < len(source) <= n:
        raise ValueError('An isolated interval must fit unchanged into two seconds.')
    pcm = np.zeros(n)
    offset = (n-len(source))//2
    pcm[offset:offset+len(source)] = source
    return pcm, {'source_start': start/rate, 'source_end': end/rate,
                 'source_duration': len(source)/rate, 'destination_start': offset/rate,
                 'zero_fill_fraction': 1-len(source)/n,
                 'context_was_limited': left > a-padding+1e-9 or right < b+padding-1e-9,
                 'source_edge_absolute_values': [float(abs(source[0])), float(abs(source[-1]))]}


def real_context(samples, rate, intervals, index):
    n = round(rate*2)
    if len(samples) < n:
        raise ValueError('Real context requires at least two seconds.')
    a, b = intervals[index]
    start = min(max(round(((a+b)/2-1)*rate), 0), len(samples)-n)
    end = start+n
    overlaps = [i for i, (left, right) in enumerate(intervals)
                if min(right, end/rate) > max(left, start/rate)]
    return np.asarray(samples[start:end]), {'source_start': start/rate, 'source_end': end/rate,
                                          'overlapping_event_ids': overlaps,
                                          'has_other_proposed_events': len(overlaps) > 1}


def consistency(results):
    candidates = [r['direction_candidate'] for r in results]
    predictions = [r['prediction'] for r in results]
    stable_candidate = len(set(candidates)) == 1
    strict = len(set(predictions)) == 1 and predictions[0] in ('up', 'down')
    accepted = {p for p in predictions if p in ('up', 'down')}
    return {'candidate_direction_stable': stable_candidate,
            'strict_accepted_direction_stable': strict,
            'stable_accepted_direction': predictions[0] if strict else None,
            'candidate_direction_flip': not stable_candidate,
            'accepted_direction_flip': len(accepted) > 1}


def boundary_stress(samples, rate, intervals, index):
    """Fixed actual boundary errors; each one trims source audio by 20 ms."""
    a, b = intervals[index]
    result = []
    for name, left, right in [('core_only', a, b), ('trim_start_20ms', a+.02, b), ('trim_end_20ms', a, b-.02)]:
        if right <= left:
            raise ValueError('Event too short for boundary stress.')
        changed = list(intervals)
        changed[index] = (left, right)
        pcm, metadata = isolate(samples, rate, changed, index, 0)
        result.append((name, pcm, metadata))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    segmentation_path = ROOT/'diagnostics/continuous/segmentation/report.json'
    segmentation = json.loads(segmentation_path.read_text())
    review_path = ROOT/'diagnostics/continuous/label_review.json'
    ids = json.loads(review_path.read_text())['take_ids']
    reference = {r['session']: r for r in segmentation['records'] if r['session'] in ids}
    protected = json.loads((ROOT/'diagnostics/continuous/research_review.json').read_text())['protected_sha256'].copy()
    for sid in ids:
        for name in ('session.json', 'continuous.wav'):
            path = ROOT/'data'/sid/name
            protected[str(path.relative_to(ROOT))] = sha(path)
    protected[str(segmentation_path.relative_to(ROOT))] = sha(segmentation_path)
    protected[str(review_path.relative_to(ROOT))] = sha(review_path)
    for name, expected in protected.items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Protected input differs: {name}')
    plan = {'status': 'Fixed before this audit run; exploratory, recordings previously inspected.',
            'events': 'Unchanged hysteresis_6 intervals from prior segmentation report',
            'padding_seconds': PADS, 'isolation': 'Keep actual bounded context; center unchanged samples in zero-filled 2 s array',
            'control': 'Unmodified real 2 s window centered on each event, shifted within recording bounds',
            'matching': 'Existing frozen preview model/thresholds; no fitting or parameter selection',
            'recurrence': 'Nearest DTW match in another take using fixed 60 ms context features; compare candidate labels only as consistency',
            'boundary_stress': 'Added after context-only audit, archived in context_only_v1. Fixed core-only, first20ms removed, last20ms removed; no duration/threshold selection. Tests damage sensitivity, not validated true boundaries.',
            'prohibited': ['training', 'using arrow times as truth', 'forcing alternation', 'live changes', 'accuracy claims'],
            'protected_sha256': protected, 'script_sha256': sha(Path(__file__))}
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    service = PreviewService()
    rows, sequences, plot_data = [], [], []
    for sid in ids:
        samples, rate, _ = decode_wav((ROOT/'data'/sid/'continuous.wav').read_bytes())
        x = np.asarray(samples)
        intervals = reference[sid]['methods']['hysteresis_6']['intervals_seconds']
        activity = reference[sid]['methods']['baseline_6']['intervals_seconds']
        plot_data.append((sid, x, rate, intervals))
        for index, (a, b) in enumerate(intervals):
            variants = []
            for padding in PADS:
                pcm, metadata = isolate(x, rate, intervals, index, padding)
                result = predict(pcm, rate, service.model)
                variants.append({'padding_seconds': padding, **metadata, **result})
                if padding == .06:
                    sequences.append(features(pcm, rate, str(service.model['feature_input']))[0])
            control, meta = real_context(x, rate, intervals, index)
            control_result = predict(control, rate, service.model)
            stress = [{'variant': name, **metadata, **predict(pcm, rate, service.model)}
                      for name, pcm, metadata in boundary_stress(x, rate, intervals, index)]
            source_cores = []
            for variant in variants:
                quality = variant['quality']
                source_cores.append([variant['source_start']+quality[key]-variant['destination_start']
                                     for key in ('event_start_seconds', 'event_end_seconds')])
            overlapped_bursts = [i for i, (left, right) in enumerate(activity)
                                if min(right, b)-max(left, a) >= .04-1e-9]
            rows.append({'take': sid, 'event_id': index, 'start': a, 'end': b, 'duration': b-a,
                         'baseline_activity_overlap': overlapped_bursts,
                         'possible_grouped_activity': len(overlapped_bursts) > 1,
                         'variants': variants, 'control': {**meta, **control_result},
                         'automatic_core_spread_seconds': np.ptp(source_cores, axis=0).tolist(),
                         'nearest_distance_spread': float(np.ptp([v['nearest_distance'] for v in variants])),
                         'boundary_stress': stress, 'boundary_stress_consistency': consistency(stress),
                         **consistency(variants)})
    for i, row in enumerate(rows):
        others = [j for j, other in enumerate(rows) if other['take'] != row['take']]
        distances = [dtw(sequences[i], sequences[j], int(service.model['radius'])) for j in others]
        j = others[int(np.argmin(distances))]
        match = rows[j]
        row['cross_take_match'] = {'take': match['take'], 'event_id': match['event_id'],
                                  'distance': float(min(distances)),
                                  'candidate_direction_agrees': row['variants'][1]['direction_candidate'] == match['variants'][1]['direction_candidate']}
    summary = []
    for sid in ids:
        group = [r for r in rows if r['take'] == sid]
        count = lambda key: sum(r[key] for r in group)
        summary.append({'take': sid, 'events': len(group),
                        'candidate_stable': count('candidate_direction_stable'),
                        'strict_accepted_stable': count('strict_accepted_direction_stable'),
                        'candidate_flips': count('candidate_direction_flip'),
                        'accepted_flips': count('accepted_direction_flip'),
                        'possible_grouped_activity': count('possible_grouped_activity'),
                        'control_contains_other_events': sum(r['control']['has_other_proposed_events'] for r in group),
                        'central_vs_control_candidate_agrees': sum(r['variants'][1]['direction_candidate'] == r['control']['direction_candidate'] for r in group),
                        'central_predictions': {label: sum(r['variants'][1]['prediction'] == label for r in group)
                                                for label in ('up', 'down', 'uncertain', 'background')},
                        'boundary_stress_candidate_flips': sum(r['boundary_stress_consistency']['candidate_direction_flip'] for r in group),
                        'boundary_stress_accepted_flips': sum(r['boundary_stress_consistency']['accepted_direction_flip'] for r in group),
                        'boundary_stress_strict_accepted_stable': sum(r['boundary_stress_consistency']['strict_accepted_direction_stable'] for r in group),
                        'boundary_stress_prediction_changes': sum(len({v['prediction'] for v in r['boundary_stress']}) > 1 for r in group),
                        'median_auto_core_spread_seconds': np.median([r['automatic_core_spread_seconds'] for r in group], axis=0).tolist(),
                        'median_nearest_distance_spread': float(np.median([r['nearest_distance_spread'] for r in group])),
                        'cross_take_neighbor_candidate_agrees': sum(r['cross_take_match']['candidate_direction_agrees'] for r in group)})
    unchanged = all(sha(ROOT/name) == expected for name, expected in protected.items())
    if not unchanged:
        raise ValueError('Protected data/model changed during the audit.')
    report = {'status': 'Prediction consistency only; no verified continuous direction or boundary accuracy.',
              'protected_unchanged': unchanged, 'plan_sha256': sha(OUT/'plan.json'),
              'summary': summary, 'events': rows,
              'limits': ['Zero-fill changes noise subtraction and gate statistics; frozen acceptance may not be calibrated for this input.',
                         'Added-context stability is partly automatic energy-cropping invariance, not evidence of onset/offset robustness. Fixed20ms trims are a separate destructive sensitivity test.',
                         'Fixed hysteresis IDs may merge movements or split one movement; pad stability does not validate segmentation.',
                         'Real contexts often contain multiple events; disagreement is not automatically a correction.',
                         'Cross-take neighbors use the same feature space as direction templates, so agreement is not independent evidence.',
                         'The three takes lack verified direction labels; no alternating labels or pseudo-label training.']}
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/swipe-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, layout='constrained')
    colors = {'up': '#24795b', 'down': '#b46036', 'uncertain': '#77708e', 'background': '#aaaaaa'}
    for ax, (sid, x, rate, intervals) in zip(axes, plot_data):
        width = round(rate*.01)
        env = np.sqrt((x.reshape(-1, width)**2).mean(axis=1))
        ax.plot((np.arange(len(env))+.5)*.01, 20*np.log10(np.maximum(env, 1e-8)), color='#48554d', lw=.6)
        for row in [r for r in rows if r['take'] == sid]:
            for lane, variant in enumerate(row['variants']):
                ax.plot([row['start'], row['end']], [-66-lane*8]*2,
                        color=colors[variant['prediction']], lw=4, solid_capstyle='butt')
            if row['candidate_direction_flip']:
                ax.text((row['start']+row['end'])/2, -91, 'flip', ha='center', fontsize=7, color='#983a45')
        ax.set_ylim(-96, 30)
        ax.set_ylabel('RMS dB re 1.0')
        ax.set_title(f'Take {ids.index(sid)+1}: context rows 30 / 60 / 100 ms', loc='left', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xlim(0, 15)
    axes[-1].set_xlabel('Seconds')
    fig.suptitle('Frozen-model boundary-context audit — predictions, not direction truth\nGreen: up · orange: down · purple: uncertain · gray: background', fontsize=12)
    fig.savefig(OUT/'stability.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'summary': summary, 'protected_unchanged': unchanged}, indent=2))


if __name__ == '__main__':
    main()
