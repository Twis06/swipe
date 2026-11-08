"""Deterministic reference representation conversion, without classifier fitting.

Writes a separate conditional candidate bank, never changes the active service.
The existing mixed-session raw gate is copied exactly. A previously frozen
legacy-domain distance cutoff is reused without estimating a new threshold.
"""
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from aligned_preview import direction_features, predict
from audio_format import decode_wav
from fast_preview import predict as baseline

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'diagnostics/aligned_preview'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'plan.json').exists():
        raise ValueError('Candidate build already exists; preserve its fixed plan and artifacts.')
    source_path = ROOT/'diagnostics/preview/model.npz'
    source_config_path = source_path.with_suffix('.json')
    config = json.loads(source_config_path.read_text())
    old_path = ROOT/'diagnostics/improvement/noise_aware/pilot_model.npz'
    with np.load(io.BytesIO(source_path.read_bytes()), allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    with np.load(io.BytesIO(old_path.read_bytes()), allow_pickle=False) as legacy:
        cutoff = float(legacy['distance_cutoff'])
    assert cutoff == .3342443569796734
    records=[]
    protected=[source_path, source_config_path, old_path, ROOT/'preview_service.py', ROOT/'preview.js', ROOT/'cascade-preview-worklet.js']
    for sid in config['sessions']:
        session_path=ROOT/'data'/sid/'session.json'
        protected.append(session_path)
        session=json.loads(session_path.read_text())
        for clip in session['clips']:
            if clip['accepted']:
                path=session_path.parent/clip['filename']
                protected.append(path)
                records.append({'session':sid,'file':clip['filename'],'label':clip['label'],'path':path,'sha256':clip['sha256']})
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    plan={'status':'Conditional deterministic reference conversion; no fitting, threshold search, live deployment, or raw-data changes.',
          'view':'Resample to44100 before clipping and legacyPCM16 quantization; preserve decodedPCM16 source references.',
          'raw_gate':'Copy every existing gate parameter from the frozen mixed-session preview model; evaluate on raw samples.',
          'distance_cutoff':cutoff,'cutoff_source':str(old_path.relative_to(ROOT)),
          'neighbors':int(source['neighbors']),'radius':int(source['radius']),'margin_threshold':float(source['margin_threshold']),
          'input_sha256':hashes,'source_sha256':sha(Path(__file__)),
          'limits':['Gate uses all existing clips; full-bank checks include self-reference and cannot estimate accuracy.',
                    'Legacy cutoff is reused, not recalibrated for this mixed aligned bank.',
                    'LiveFloat32 always quantizes, including replayed decodedPCM16 values; no source metadata survives microphone input.',
                    'New held-out and stateful-capture replay checks are required before any promotion.']}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    templates=[];references=[];inputs=[]
    for record in records:
        if sha(record['path'])!=record['sha256']:
            raise ValueError('Audio differs from its original manifest.')
        samples,rate,encoding=decode_wav(record['path'].read_bytes(),array=True)
        inputs.append((record,samples,rate,encoding))
        if record['label'] in ('up','down'):
            templates.append(direction_features(samples,rate,encoding))
            references.append({k:record[k] for k in ('session','file','label','sha256')}|{'source_encoding':encoding})
    model={key:value.copy() for key,value in source.items()}
    model['templates']=np.asarray(templates)
    model['labels']=np.array([r['label'] for r in references])
    model['distance_cutoff']=np.array(cutoff)
    model['direction_view']=np.array('legacy_encoding_v1')
    model['reference_sessions']=np.array([r['session'] for r in references])
    model['reference_files']=np.array([r['file'] for r in references])
    model['reference_encodings']=np.array([r['source_encoding'] for r in references])
    model['reference_sha256']=np.array([r['sha256'] for r in references])
    np.savez_compressed(OUT/'model.npz',**model)
    manifest={'status':'Conditional candidate; not active. Existing examples converted deterministically, no fitting.',
              'feature_input':'raw','direction_view':'legacy_encoding_v1','training_clips':config['training_clips'],
              'sessions':config['sessions'],'reference_swipes':len(references),'references':references,
              'source_model_path':str(source_path.relative_to(ROOT)),'source_model_sha256':sha(source_path),
              'model_sha256':sha(OUT/'model.npz'),'distance_cutoff':cutoff,'cutoff_source_sha256':sha(old_path),
              'plan_sha256':sha(OUT/'plan.json'),
              'implementation_sha256':{name:sha(ROOT/name) for name in ('aligned_preview.py','fast_preview.py','preview_model.py','gesture_model.py')},
              'limits':plan['limits']}
    (OUT/'model.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Integration/timing only: every example is in the source bank. Queries use
    # the liveFloat32 branch, including oldPCM16 examples sent by a worklet.
    timings={'baseline':[],'aligned':[]};rows=[]
    for _,samples,rate,_ in inputs[:3]:baseline(samples,rate,source);predict(samples,rate,model)
    rng=np.random.default_rng(9021)
    for repeat in range(3):
        for index in rng.permutation(len(inputs)):
            record,samples,rate,encoding=inputs[index];results={}
            for method in rng.permutation(['baseline','aligned']):
                start=time.perf_counter()
                results[method]=(baseline(samples,rate,source) if method=='baseline' else predict(samples,rate,model))
                timings[method].append((time.perf_counter()-start)*1000)
            assert results['baseline']['swipe_gate_score']==results['aligned']['swipe_gate_score']
            assert results['baseline']['quality']==results['aligned']['quality']
            if repeat==0:
                rows.append({'session':record['session'],'file':record['file'],'label':record['label'],
                             'source_encoding':encoding,'query_encoding':'float32',**results})
    unchanged=all(sha(ROOT/name)==expected for name,expected in hashes.items())
    if not unchanged:raise ValueError('Protected source changed during candidate construction.')
    report={'status':'Full-bank integration check including reference examples; NOT validation accuracy.',
            'source_unchanged':unchanged,'gate_and_quality_exact':True,'timings_ms':{name:{'calls':len(values),'p50':float(np.median(values)),'p95':float(np.quantile(values,.95))} for name,values in timings.items()},
            'changed_output_count':sum(r['baseline']['prediction']!=r['aligned']['prediction'] for r in rows),
            'rows':rows,'limits':plan['limits']}
    (OUT/'integration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
