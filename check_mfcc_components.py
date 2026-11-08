"""Predeclared candidate parity, fixed sensitivity checks and paired timings."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import resample
from audio_format import decode_wav
from preview_model import features
from mfcc_components import extract_variant
from fast_preview import batch_dtw
from gesture_model import direction

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'diagnostics/mfcc_components_v1'
OUT=BASE/'sensitivity'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    plan={'candidate':'Four named follow-ups after the single-component comparison: mfcc_base,cepstra8,smooth_delta,preemphasis. No combinations or additional parameter search. All evidence exploratory.',
          'variants':['clean','gain0.25','20dB full-clip-RMS SNR using first idle clip from same session','110% resampled length at unchanged rate (slows time AND lowers pitch, not a physical speed simulation)'],
          'matching':'Fixed3-neighbor DTW radius4; entire query session excluded, reference features untouched.',
          'parity':'All84 cached features must match standalone extractor within1e-10.',
          'timing':'3 paired randomized repeats across60 swipes; extraction+DTW against40 references; decision/gate/HTTP/capture excluded. Warmup4 calls.',
          'files':{n:sha(ROOT/n) for n in ['mfcc_components.py','check_mfcc_components.py','diagnostics/preview/model.npz','diagnostics/preview/engine.json','preview_service.py']}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    records=json.loads((BASE/'records.json').read_text())
    with np.load(BASE/'features.npz') as bank:cache={m:bank[m] for m in ('mfcc_base','cepstra8','smooth_delta','preemphasis')}
    audio=[]
    for i,r in enumerate(records):
        x,rate,_=decode_wav((ROOT/'data'/r['session']/r['file']).read_bytes(),array=True);audio.append((x,rate))

        for method in cache:np.testing.assert_allclose(extract_variant(x,rate,method),cache[method][i],atol=1e-10,rtol=1e-10)
    ids=[i for i,r in enumerate(records) if r['label'] in ('up','down')]
    labels=np.array([r['label'] for r in records]);rows=[]
    for i in ids:
        r=records[i];x,rate=audio[i]
        ni=next(j for j,n in enumerate(records) if n['session']==r['session'] and n['label']=='idle')
        noise,nrate=audio[ni];assert nrate==rate
        noise=np.resize(noise,len(x));noise-=noise.mean()
        scaled=noise*np.sqrt(np.mean(x*x))/max(np.sqrt(np.mean(noise*noise)),1e-30)/10
        variants={'clean':x,'gain0.25':x*.25,'noise20dB':x+scaled,'stretch110pct':resample(x,round(len(x)*1.1))}
        refs=np.array([j for j in ids if records[j]['session']!=r['session']])
        for variant,y in variants.items():
            for method in cache:
                seq=extract_variant(y,rate,method)
                p,m,d=direction(batch_dtw(seq,cache[method][refs],4),labels[refs],3)
                rows.append({'session':r['session'],'file':r['file'],'truth':r['label'],'variant':variant,'method':method,'prediction':p,'margin':m})
    times={m:[] for m in cache};rng=np.random.default_rng(8127)
    for method in cache:
        for i in ids[:4]:
            x,rate=audio[i];seq=extract_variant(x,rate,method)
            batch_dtw(seq,cache[method][ids[:40]],4)
    for repeat in range(3):
        for i in rng.permutation(ids):
            x,rate=audio[i]
            for method in rng.permutation(list(cache)):
                start=time.perf_counter();seq=extract_variant(x,rate,method)
                batch_dtw(seq,cache[method][ids[:40]],4)
                times[method].append((time.perf_counter()-start)*1000)
    summary={}
    for v in variants:
        summary[v]={m:{'correct':sum(r['prediction']==r['truth'] for r in rows if r['variant']==v and r['method']==m),
                      'by_session':{sid:sum(r['prediction']==r['truth'] for r in rows if r['variant']==v and r['method']==m and r['session']==sid) for sid in sorted({r['session'] for r in records})}}
                    for m in cache}
    report={'parity_clips':len(records),'summary':summary,'timings_ms':{m:{'median':float(np.median(v)),'p95':float(np.quantile(v,.95)),'calls':len(v)} for m,v in times.items()},'rows':rows,
            'protected_unchanged':all(sha(ROOT/n)==h for n,h in plan['files'].items()),
            'limits':['Reused recordings; candidate selected using previous scores.','Forced direction results do not establish usable rejection or false-activation rates.','Synthetic perturbations do not replace real recordings.','Timing excludes gate, decision, HTTP and capture.']}
    assert report['protected_unchanged']
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
