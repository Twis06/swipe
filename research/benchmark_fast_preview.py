"""Serial, paired wall-clock benchmark of unchanged frozen inference outputs."""
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from swipe.audio_format import decode_wav
from swipe.fast_preview import batch_dtw, predict as fast
from swipe.gesture_model import dtw
from swipe.preview_model import features, predict as reference

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'diagnostics/performance'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing_summary(values):
    values = np.asarray(values)*1000
    return {'calls': len(values), 'p50_ms': float(np.median(values)),
            'p95_ms': float(np.quantile(values, .95)), 'mean_ms': float(np.mean(values))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected_paths = [ROOT/'models/preview/model.npz', ROOT/'models/preview/model.json',
                       ROOT/'swipe/preview_model.py', ROOT/'swipe/gesture_model.py', ROOT/'research/worklets/preview-worklet.js']
    recordings = []
    for sid in ('20261103T165139_cf7201', '20261103T200050_87dcb0'):
        folder = ROOT/'data'/sid
        session = json.loads((folder/'session.json').read_text())
        for clip in session['clips']:
            if clip['accepted']:
                path = folder/clip['filename']
                protected_paths.append(path)
                recordings.append((f'{sid}/{clip["filename"]}', path.read_bytes(), None))
    ids = json.loads((ROOT/'diagnostics/continuous/label_review.json').read_text())['take_ids']
    for sid in ids:
        path = ROOT/'data'/sid/'continuous.wav'
        protected_paths.append(path)
        for start in (0,4,8,12):
            recordings.append((f'{sid}@{start}s', path.read_bytes(), start))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected_paths}
    plan = {'method': 'Serial paired prediction timings; alternate/randomize old/new order, three repeats per input after warmup; no parallel benchmark work.',
            'inputs': '56 existing isolated clips and twelve unmodified two-second continuous windows at0,4,8,12s.',
            'scope': 'Model inference time, excludes microphone buffering/HTTP and WAV decoding; decode component timed separately.',
            'repeats': 3, 'seed': 821, 'protected_sha256': hashes,
            'fast_source_sha256': sha(ROOT/'swipe/fast_preview.py'), 'benchmark_source_sha256': sha(Path(__file__)),
            'system': {'platform': platform.platform(), 'python': platform.python_version(), 'numpy': np.__version__}}
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    with np.load(ROOT/'models/preview/model.npz', allow_pickle=False) as bank:
        model = {key: bank[key] for key in bank.files}
    inputs = []
    decode_times = []
    for name, blob, start in recordings:
        begin = time.perf_counter()
        samples, rate, _ = decode_wav(blob)
        elapsed = time.perf_counter()-begin
        if start is None:
            decode_times.append(elapsed)
        else:
            samples = samples[start*rate:(start+2)*rate]
        inputs.append((name, samples, rate))
    for _, samples, rate in inputs[:3]:
        reference(samples, rate, model)
        fast(samples, rate, model)
    timings = {'reference': [], 'fast': []}
    rows = []
    rng = np.random.default_rng(821)
    for repeat in range(3):
        for i in rng.permutation(len(inputs)):
            name, samples, rate = inputs[i]
            results = {}
            row = {'input': name, 'repeat': repeat}
            for method in rng.permutation(['reference', 'fast']):
                begin = time.perf_counter()
                result = (reference if method == 'reference' else fast)(samples, rate, model)
                elapsed = time.perf_counter()-begin
                results[method] = result
                timings[method].append(elapsed)
                row[method+'_ms'] = elapsed*1000
            if results['reference'] != results['fast']:
                raise AssertionError(f'Frozen output changed for {name}')
            row['prediction'] = results['fast']['prediction']
            rows.append(row)
    components = {'features': [], 'dtw_reference': [], 'dtw_fast': []}
    for _, samples, rate in inputs:
        begin = time.perf_counter()
        sequence, _, _ = features(samples, rate, str(model['feature_input']))
        components['features'].append(time.perf_counter()-begin)
        begin = time.perf_counter()
        expected = [dtw(sequence, template, int(model['radius'])) for template in model['templates']]
        components['dtw_reference'].append(time.perf_counter()-begin)
        begin = time.perf_counter()
        actual = batch_dtw(sequence, model['templates'], int(model['radius']))
        components['dtw_fast'].append(time.perf_counter()-begin)
        np.testing.assert_array_equal(actual, expected)
    summary = {name: timing_summary(values) for name, values in timings.items()}
    speedup = summary['reference']['p50_ms']/summary['fast']['p50_ms']
    unchanged = all(sha(ROOT/path) == expected for path, expected in hashes.items())
    assert unchanged, 'Protected file changed during benchmark.'
    report = {'status': 'Exact-output inference optimization; no model retraining or claimed accuracy change.',
              'unique_inputs': len(inputs), 'paired_calls': len(rows), 'exact_result_dict_parity': True,
              'protected_files_unchanged': unchanged, 'summary': summary, 'median_speedup': speedup,
              'components': {name: timing_summary(values) for name, values in components.items()},
              'isolated_wav_decode_component': timing_summary(decode_times),
              'plan_sha256': sha(OUT/'plan.json'), 'timings': rows,
              'limits': 'Wall-clock timings reflect this machine/load. Feature extraction is unchanged. This does not shorten the existing one-second capture wait, HTTP, or browser scheduling.'}
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'timings'}, indent=2))


if __name__ == '__main__':
    main()
