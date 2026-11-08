"""Fixed encoding-domain alignment probe. No fitting or live/raw changes.

The single alternative follows the already documented legacy encoder diagnostic:
resample Float32 copies to 44.1kHz, clamp and quantize like protocol1. Already
encoded PCM16 references remain unchanged to avoid a second quantization.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from audio_format import decode_wav
from fast_preview import batch_dtw
from gesture_model import canonical_audio, decision, direction
from preview_model import features

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'diagnostics/accuracy_alignment'
SESSIONS=['20261103T165139_cf7201','20261103T200050_87dcb0']
GATES={SESSIONS[0]:'diagnostics/improvement/noise_aware/pilot_model.npz',
       SESSIONS[1]:'diagnostics/calibration/20261103T200050_87dcb0/pilot_model.npz'}
METHODS=['raw','legacy_encoding']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def align_signal(samples, rate, encoding):
    x=canonical_audio(samples,rate).copy()
    if encoding=='pcm16':
        return x
    if encoding!='float32':
        raise ValueError('Expected known PCM16 or Float32 encoding.')
    x=np.clip(x,-1,1)
    return np.where(x<0,x*32768,x*32767).astype(np.int16).astype(np.float64)/32768


def reference_indices(query, records, split):
    key='session' if split=='cross_session' else 'group' if split=='leave_group_out' else None
    if key is None:raise ValueError('Unknown split')
    return np.array([i for i,r in enumerate(records) if r['label'] in ('up','down') and r[key]!=records[query][key]],dtype=int)


def summary(rows, rejection=False):
    swipes=[r for r in rows if r['truth'] in ('up','down')]
    result={'forced_correct':sum(r['direction_candidate']==r['truth'] for r in swipes),'swipes':len(swipes),
            'by_class':{label:{'correct':sum(r['truth']==label and r['direction_candidate']==label for r in swipes),
                               'total':sum(r['truth']==label for r in swipes)} for label in ('up','down')}}
    if rejection:
        accepted=[r for r in swipes if r['prediction'] in ('up','down')]
        neg=[r for r in rows if r['truth'] not in ('up','down')]
        result.update({'accepted':len(accepted),'accepted_correct':sum(r['prediction']==r['truth'] for r in accepted),
                       'accepted_wrong':sum(r['prediction']!=r['truth'] for r in accepted),
                       'negatives':len(neg),'negative_activations':sum(r['prediction'] in ('up','down') for r in neg)})
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():raise RuntimeError('Existing plan/results preserved; no unrecorded rerun.')
    records=[]
    protected=[ROOT/name for name in ['preview_model.py','gesture_model.py','fast_preview.py','preview_service.py',
                                     'cascade-preview-worklet.js','diagnostics/preview/model.npz','diagnostics/preview/model.json']]
    gates={}
    for sid in SESSIONS:
        folder=ROOT/'data'/sid
        protected.append(folder/'session.json')
        session=json.loads((folder/'session.json').read_text())
        for clip in session['clips']:
            if clip['accepted']:
                path=folder/clip['filename'];protected.append(path)
                records.append({'session':sid,'file':clip['filename'],'label':clip['label'],
                                'group':f'{sid}_{clip["block"]}_{clip["pace"]}','path':path,'sha256':clip['sha256']})
        protected.append(ROOT/GATES[sid])
        with np.load(ROOT/GATES[sid],allow_pickle=False) as bank:
            gates[sid]={k:bank[k] for k in bank.files}
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    plan={'status':'Two fixed methods, no fitting, no parameter search, no deployment.',
          'methods':{'raw':'Original raw preview features.',
                     'legacy_encoding':'Already-PCM16 samples preserved; Float32 copies resampled to44100, clipped to[-1,1], negative*32768 /positive*32767 truncated toint16 then decoded /32768. Same feature extractor for both.'},
          'reference_exclusion':['Entire query session','Entire session/block/pace group'],
          'direction':{'neighbors':3,'radius':4},
          'rejection':'Cross-session only: original raw gate features and already-frozen OTHER-session gate/cutoff/margin. No threshold recalibration; transformed template-distance calibration may be invalid. Grouped acceptance omitted because stored full-session gates include held-out groups.',
          'source_sha256':sha(Path(__file__)),'input_sha256':hashes,
          'limits':['All recordings previously inspected; prior format-emulation improvement already motivated this test.',
                    'First session is clipped PCM16; session and encoding remain confounded.',
                    'This discards information only on analysis copies; does not repair clipping or modify raw audio.',
                    'No continuous direction labels or pseudo-labels used.']}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    sequences={m:[] for m in METHODS};timings={m:[] for m in METHODS};gate_features=[];silent=[]
    for record in records:
        if sha(record['path'])!=record['sha256']:raise ValueError('Raw audio checksum mismatch')
        samples,rate,encoding=decode_wav(record['path'].read_bytes(),array=True)
        record['encoding']=encoding
        t=time.perf_counter();seq,gate,quality=features(samples,rate,'raw');timings['raw'].append(time.perf_counter()-t)
        sequences['raw'].append(seq);gate_features.append(gate);silent.append(quality['silent'])
        t=time.perf_counter();aligned=align_signal(samples,rate,encoding)
        sequences['legacy_encoding'].append(features(aligned,44100,'raw')[0]);timings['legacy_encoding'].append(time.perf_counter()-t)
    labels=np.array([r['label'] for r in records])
    report={'status':'Already-inspected data, deterministic alignment only; no fresh validation.',
            'plan_sha256':sha(OUT/'plan.json'),'methods':{},'limits':plan['limits']}
    for method in METHODS:
        seq=np.asarray(sequences[method]);distances=np.array([batch_dtw(q,seq,4) for q in seq])
        splits={}
        for split in ('cross_session','leave_group_out'):
            rows=[]
            for i,record in enumerate(records):
                refs=reference_indices(i,records,split)
                candidate,margin,distance=direction(distances[i,refs],labels[refs],3)
                row={'session':record['session'],'file':record['file'],'group':record['group'],'truth':record['label'],
                     'direction_candidate':candidate,'margin':margin,'distance':distance,'reference_indices':refs.tolist()}
                if split=='cross_session':
                    other=next(s for s in SESSIONS if s!=record['session']);model=gates[other]
                    z=(gate_features[i]-model['gate_mean'])/model['gate_scale']
                    score=float(1/(1+np.exp(-np.clip(float(z@model['gate_coef']+model['gate_intercept']),-700,700))))
                    prediction,reason=decision(candidate,margin,distance,float(model['distance_cutoff']),score,silent[i],float(model['margin_threshold']))
                    row.update({'prediction':prediction,'reason':reason,'gate_reference_session':other,
                                'gate_score':score,'distance_cutoff':float(model['distance_cutoff'])})
                rows.append(row)
            key='session' if split=='cross_session' else 'group'
            splits[split]={'summary':summary(rows,split=='cross_session'),
                           'folds':{v:summary([r for r in rows if r[key]==v],split=='cross_session') for v in sorted({r[key] for r in rows})},
                           'predictions':rows}
        report['methods'][method]=splits
    report['feature_times_ms']={m:{'median':float(np.median(v)*1000),'p95':float(np.quantile(v,.95)*1000)} for m,v in timings.items()}
    report['paired_changes']={split:[{'session':a['session'],'file':a['file'],'truth':a['truth'],
                                     'raw':a['direction_candidate'],'aligned':b['direction_candidate'],
                                     'raw_accepted':a.get('prediction'),'aligned_accepted':b.get('prediction')}
                                    for a,b in zip(report['methods']['raw'][split]['predictions'],report['methods']['legacy_encoding'][split]['predictions'])
                                    if a['direction_candidate']!=b['direction_candidate'] or a.get('prediction')!=b.get('prediction')]
                              for split in ('cross_session','leave_group_out')}
    report['record_order']=[{k:r[k] for k in ('session','file','label','group','encoding')} for r in records]
    report['protected_unchanged']=all(sha(ROOT/path)==expected for path,expected in hashes.items())
    if not report['protected_unchanged']:raise ValueError('Protected inputs changed during run')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    compact={m:{'cross_session':v['cross_session']['summary'],'session_folds':v['cross_session']['folds'],
                'grouped':v['leave_group_out']['summary']} for m,v in report['methods'].items()}
    (OUT/'summary.json').write_text(json.dumps(compact,indent=2)+'\n')
    print(json.dumps(compact,indent=2))


if __name__=='__main__':main()
