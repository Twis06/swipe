"""Interactive fixed cascade replay; model inference ack changes capture state.

Signal time pauses during Python classification. Timing is idealized worklet
delay, not HTTP/browser latency; parent integration separately tests busy queues.
"""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from preview_model import predict
from preview_service import PreviewService

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'diagnostics/capture_latency/cascade_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def displayed(row):return [w for w in row['windows'] if w['displayed']]
def accepted(row):return [w for w in displayed(row) if w['result']['prediction'] in ('up','down')]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ids=['20261103T165139_cf7201','20261103T200050_87dcb0']+json.loads((ROOT/'diagnostics/continuous/label_review.json').read_text())['take_ids']
    cases=[];protected=[]
    for sid in ids:
        s=json.loads((ROOT/'data'/sid/'session.json').read_text());protected.append(ROOT/'data'/sid/'session.json')
        for clip in s['clips']:
            if not clip['accepted']:continue
            protected.append(ROOT/'data'/sid/clip['filename'])
            for lead in (['none'] if s.get('kind')=='continuous' else ['zero','repeated_recorded_background']):
                for variant in ['preview-worklet.js','cascade-preview-worklet.js']:
                    cases.append({'session':sid,'file':clip['filename'],'truth':clip['label'],'kind':s.get('kind','isolated'),'lead':lead,'variant':variant})
    protected.extend(ROOT/n for n in ['preview-worklet.js','preview.js','preview_model.py','gesture_model.py','diagnostics/preview/model.npz','diagnostics/preview/model.json'])
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    plan={'status':'One fixed cascade protocol after v2 acceptance regression, before cascade replay. No tuning/fitting.',
          'protocol':'Early source-isolated candidate100ms preroll;200ms min onset age+100ms quiet. Accept Up/Down acknowledges matchingID and cancels fallback. Otherwise original real2s window at onset+1s.400ms cooldown after acknowledgment or fallback. Early uncertainty/background hidden.',
          'leads':['none for real continuous','synthetic1.2s zeros','synthetic1.2s repeated originalfirst250ms'],
          'tail':'1.5s zeros only for isolated clips',
          'timing':'Persistent Node processor stops signal clock per candidate; Python classifies and sends acceptedID before resume. Ideal immediate acknowledgment; excludes model/network/UI wallclock delays.',
          'promotion_rule':'No isolated acceptance/correctness/negative/duplicate regression in eitherlead plus faster displayedresult latency. Continuous acceptance is not accuracy.',
          'protected_sha256':hashes,'source_sha256':{n:sha(ROOT/n) for n in ['cascade-preview-worklet.js','compare_cascade_capture.cjs','compare_cascade_capture.py']}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    service=PreviewService();rows=[]
    worker=subprocess.Popen(['node',str(ROOT/'compare_cascade_capture.cjs')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    def rpc(value):
        worker.stdin.write(json.dumps(value)+'\n');worker.stdin.flush();line=worker.stdout.readline()
        if not line:raise RuntimeError(worker.stderr.read())
        response=json.loads(line)
        if 'error' in response:raise RuntimeError(response['error'])
        return response
    try:
        for case in cases:
            info=rpc({'cmd':'load',**case});row={**case,'rate':info['rate'],'windows':[]}
            while True:
                window=rpc({'cmd':'step'})
                if window.pop('done'):break
                pcm=np.frombuffer(base64.b64decode(window.pop('pcm_base64')),dtype='<f4')
                result=predict(pcm,row['rate'],service.model)
                status=rpc({'cmd':'classify','window':window,'result':result})
                row['windows'].append({**window,**status,'result':result,'delay_seconds':(window['emitFrame']-window['onsetFrame'])/row['rate']})
            rows.append(row)
    finally:
        worker.stdin.close();worker.wait(timeout=10)
    segmentation=json.loads((ROOT/'diagnostics/continuous/segmentation/report.json').read_text())
    activity={r['session']:r['methods']['baseline_6']['intervals_seconds'] for r in segmentation['records'] if r['kind']=='continuous'}
    summary={}
    for lead in ['zero','repeated_recorded_background','none']:
        summary[lead]={}
        for variant in ['preview-worklet.js','cascade-preview-worklet.js']:
            group=[r for r in rows if r['lead']==lead and r['variant']==variant]
            if lead=='none':
                records=[]
                for r in group:
                    for w in r['windows']:
                        a=w.get('sourceStartFrame',w['startFrame'])/r['rate'];b=w.get('sourceEndFrame',w['emitFrame'])/r['rate']
                        w['source_activity_overlap']=sum(min(b,y)-max(a,x)>=.04 for x,y in activity[r['session']])
                    delays=[w['delay_seconds']*1000 for w in displayed(r)]
                    early= [w for w in accepted(r) if w['phase']=='early']
                    records.append({'session':r['session'],'candidate_windows':len(r['windows']),'displayed_results':len(displayed(r)),
                                    'accepted_directions':len(accepted(r)),'accepted_early':len(early),
                                    'accepted_early_with_multi_activity':sum(w['source_activity_overlap']>1 for w in early),
                                    'displayed_delay_median_ms':float(np.median(delays)) if delays else None,
                                    'displayed_delay_p95_ms':float(np.quantile(delays,.95)) if delays else None})
                summary[lead][variant]=records
            else:
                sw=[r for r in group if r['truth'] in ('up','down')];neg=[r for r in group if r['truth'] not in ('up','down')]
                first=[displayed(r)[0]['delay_seconds']*1000 for r in sw if displayed(r)]
                summary[lead][variant]={'swipe_clips':len(sw),'swipes_with_candidate':sum(bool(r['windows']) for r in sw),
                  'accepted_windows':sum(len(accepted(r)) for r in sw),'correct_accepted_windows':sum(w['result']['prediction']==r['truth'] for r in sw for w in accepted(r)),
                  'wrong_accepted_clips':sum(any(w['result']['prediction']!=r['truth'] for w in accepted(r)) for r in sw),
                  'clips_with_multiple_displayed':sum(len(displayed(r))>1 for r in sw),
                  'negative_clips':len(neg),'negative_clips_with_accepted':sum(bool(accepted(r)) for r in neg),
                  'accepted_early':sum(w['phase']=='early' for r in sw for w in accepted(r)),
                  'first_displayed_delay_median_ms':float(np.median(first)),'first_displayed_delay_p95_ms':float(np.quantile(first,.95))}
    baseline={(r['session'],r['file'],r['lead']):r for r in rows if r['kind']=='isolated' and r['variant']=='preview-worklet.js'}
    changed=[]
    for r in rows:
        if r['variant']!='cascade-preview-worklet.js' or r['kind']!='isolated':continue
        old=baseline[(r['session'],r['file'],r['lead'])];labels=lambda row:[w['result']['prediction'] for w in displayed(row)]
        if labels(old)!=labels(r):changed.append({'session':r['session'],'file':r['file'],'lead':r['lead'],'truth':r['truth'],'baseline':labels(old),'cascade':labels(r)})
    unchanged=all(sha(ROOT/p)==digest for p,digest in hashes.items())
    report={'status':'Fixed cascade replay, no fitting. Signal-clock delay excludes actual inference/network latency.',
            'summary':summary,'changed_isolated_results':changed,'records':rows,'protected_unchanged':unchanged,
            'limits':['Old clips are training/development data; not fresh accuracy.','Three continuous takes have no verified directions.','Earlyzero-fill changes noise/gate calibration.','Ideal immediateack; delayedrequestqueue separately requires clienttesting.']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'summary':summary,'changed_isolated_results':changed,'protected_unchanged':unchanged},indent=2))

if __name__=='__main__':main()
