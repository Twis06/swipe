"""Stateful cascade comparison of frozen raw and deterministic aligned views.

No fitting. Worker signal time pauses for inference, so reported capture delay
excludes actual model/network/browser timing. All waveform queries are Float32.
"""
import base64
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from swipe.aligned_preview import predict as aligned_predict
from swipe.fast_preview import predict as raw_predict
from swipe.preview_service import PreviewService
from research.compare_cascade_capture import accepted, displayed

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    name='mfcc8'
    from swipe.selective_preview import predict as selective_predict
    from swipe.mfcc_preview import predict as mfcc_predict,load_model
    candidate_model,_=load_model()
    OUT=ROOT/'diagnostics/mfcc_preview/replay'
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'report.json').exists():
        raise RuntimeError('Completed replay preserved; use an explicitly new experiment for changes.')
    ids=['20261103T165139_cf7201','20261103T200050_87dcb0','20261107T185348_9cbd41']+json.loads((ROOT/'diagnostics/continuous/label_review.json').read_text())['take_ids']
    cases=[];protected=[]
    for sid in ids:
        folder=ROOT/'data'/sid
        session=json.loads((folder/'session.json').read_text())
        protected.append(folder/'session.json')
        for clip in session['clips']:
            if not clip['accepted']:continue
            protected.append(folder/clip['filename'])
            for lead in (['none'] if session.get('kind')=='continuous' else ['zero','repeated_recorded_background']):
                for recognizer in ('raw',name):
                    cases.append({'session':sid,'file':clip['filename'],'truth':clip['label'],
                                  'kind':session.get('kind','isolated'),'lead':lead,
                                  'variant':'web/cascade-preview-worklet.js','recognizer':recognizer})
    protected.extend(ROOT/name for name in ['models/preview/model.npz','models/preview/model.json',
        'web/preview.js','web/cascade-preview-worklet.js','swipe/gesture_model.py','swipe/preview_model.py','swipe/aligned_preview.py',
        'models/aligned_preview/model.npz','models/aligned_preview/model.json'])
    protected.extend(ROOT/p for p in ['swipe/selective_preview.py','swipe/mfcc_preview.py','swipe/mfcc_components.py','models/mfcc_preview/model.npz','models/mfcc_preview/model.json'])
    hashes={str(path.relative_to(ROOT)):sha(path) for path in protected}
    plan={'status':'Written before paired stateful replay. No fitting or threshold search.',
          'recognizers':['currently deployed selective engine',name],
          'candidate_policy':'Separate60-template MFCC8 reference bank; fixed.2 margin, reference-only95th percentile cutoff, unchanged raw gate.',
          'capture':'Same deployed cascade, same seamlessly framed samples, accepted early results acknowledge before next block.',
          'leads':['No padding for continuous','1.2s syntheticzero +1.5s zero tail','1.2s repeatedfirst250ms +1.5s zero tail'],
          'promotion_rule':'No increase in accepted wrong gestures, negative activations, or duplicates; no drop in correct accepted windows and no previously correct clip loses acceptance in either isolated lead condition. Continuous outputs are unverified.',
          'protected_sha256':hashes,'script_sha256':sha(Path(__file__)),
          'worker_sha256':sha(ROOT/'research/compare_cascade_capture.cjs')}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    raw_model=PreviewService().model
    with np.load(ROOT/'models/aligned_preview/model.npz',allow_pickle=False) as bank:
        aligned_model={key:bank[key] for key in bank.files}
    worker=subprocess.Popen(['node',str(ROOT/'research/compare_cascade_capture.cjs')],stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    def rpc(value):
        worker.stdin.write(json.dumps(value)+'\n');worker.stdin.flush()
        line=worker.stdout.readline()
        if not line:raise RuntimeError(worker.stderr.read())
        response=json.loads(line)
        if 'error' in response:raise RuntimeError(response['error'])
        return response
    rows=[]
    try:
        for case in cases:
            info=rpc({'cmd':'load',**case});row={**case,'rate':info['rate'],'windows':[]}
            while True:
                window=rpc({'cmd':'step'})
                if window.pop('done'):break
                pcm=np.frombuffer(base64.b64decode(window.pop('pcm_base64')),dtype='<f4')
                if case['recognizer']=='raw':result=selective_predict(pcm,row['rate'],raw_model,aligned_model)
                else:result=mfcc_predict(pcm,row['rate'],candidate_model)
                status=rpc({'cmd':'classify','window':window,'result':result})
                row['windows'].append({**window,**status,'result':result,
                    'delay_seconds':(window['emitFrame']-window['onsetFrame'])/row['rate']})
            rows.append(row)
    finally:
        worker.stdin.close();worker.wait(timeout=10)
    summary={}
    for lead in ('zero','repeated_recorded_background','none'):
        summary[lead]={}
        for recognizer in ('raw',name):
            group=[r for r in rows if r['lead']==lead and r['recognizer']==recognizer]
            if lead=='none':
                summary[lead][recognizer]=[{'take':r['session'],'candidate_windows':len(r['windows']),
                    'displayed_results':len(displayed(r)),'accepted_unverified_directions':len(accepted(r))} for r in group]
                continue
            swipes=[r for r in group if r['truth'] in ('up','down')]
            negatives=[r for r in group if r['truth'] not in ('up','down')]
            delays=[displayed(r)[0]['delay_seconds']*1000 for r in swipes if displayed(r)]
            summary[lead][recognizer]={'swipes':len(swipes),'negative_clips':len(negatives),
                'correct_accepted_windows':sum(w['result']['prediction']==r['truth'] for r in swipes for w in accepted(r)),
                'accepted_wrong_clips':sum(any(w['result']['prediction']!=r['truth'] for w in accepted(r)) for r in swipes),
                'negative_activations':sum(bool(accepted(r)) for r in negatives),
                'multiple_displayed_clips':sum(len(displayed(r))>1 for r in swipes),
                'first_display_median_ms':float(np.median(delays)) if delays else None,
                'first_display_p95_ms':float(np.quantile(delays,.95)) if delays else None}
    passed=True
    for lead in ('zero','repeated_recorded_background'):
        old,new=summary[lead]['raw'],summary[lead][name]
        passed &= new['correct_accepted_windows']>=old['correct_accepted_windows']
        for key in ('accepted_wrong_clips','negative_activations','multiple_displayed_clips'):
            passed &= new[key]<=old[key]
    oldrows={(r['session'],r['file'],r['lead']):r for r in rows if r['recognizer']=='raw' and r['kind']=='isolated'}
    changes=[]
    for row in rows:
        if row['recognizer']!=name or row['kind']!='isolated':continue
        before=oldrows[(row['session'],row['file'],row['lead'])]
        labels=lambda r:[w['result']['prediction'] for w in displayed(r)]
        if row['truth'] in ('up','down') and row['truth'] in labels(before) and row['truth'] not in labels(row):
            passed=False
        if labels(before)!=labels(row):
            changes.append({k:row[k] for k in ('session','file','lead','truth')}|
                           {'before':labels(before),'after':labels(row)})
    unchanged=all(sha(ROOT/path)==digest for path,digest in hashes.items())
    if not unchanged:raise ValueError('A protected input changed during replay.')
    report={'status':'Reused-data integration evidence, not fresh accuracy. Candidate references and novelty cutoff were rebuilt; gate weights unchanged.',
            'summary':summary,'isolated_nonregression_pass':bool(passed),'changes':changes,
            'records':rows,'protected_unchanged':unchanged,'plan_sha256':sha(OUT/'plan.json'),
            'limits':['Existing exemplars are in reference banks; isolated replay cannot measure generalization.',
                      'Continuous gesture directions/boundaries unknown; counts are not accuracy.',
                      'Signal-clock delays exclude real compute/network/UI latency.']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','changes')},indent=2))


if __name__=='__main__':main()
