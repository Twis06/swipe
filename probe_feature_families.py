"""Seven fixed direction representations, whole-session exclusion, no fitting."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from swipe_feature_families import METHODS,extract_all
from audio_format import decode_wav
from fast_preview import batch_dtw
from gesture_model import direction

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'diagnostics/feature_families_v1'
SESSIONS=['20261103T165139_cf7201','20261103T200050_87dcb0','20261107T185348_9cbd41']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def summarize(rows):
    return {'correct':sum(r['prediction']==r['truth'] for r in rows),'total':len(rows),
            'by_class':{c:{'correct':sum(r['prediction']==c for r in rows if r['truth']==c),'total':sum(r['truth']==c for r in rows)} for c in ('up','down')}}

def main():
    OUT.mkdir(exist_ok=False)
    records=[];protected={}
    for sid in SESSIONS:
        path=ROOT/'data'/sid/'session.json';protected[str(path.relative_to(ROOT))]=sha(path)
        s=json.loads(path.read_text());assert s['metadata']['microphone_label']=='MacBook Pro Microphone (Built-in)'
        for c in s['clips']:
            if not c['accepted']:continue
            path=ROOT/'data'/sid/c['filename'];assert sha(path)==c['sha256']
            protected[str(path.relative_to(ROOT))]=sha(path)
            records.append({'session':sid,'file':c['filename'],'label':c['label'],'group':f'{sid}_{c["block"]}_{c["pace"]}'})
    for name in ['diagnostics/preview/model.npz','diagnostics/preview/engine.json','preview_service.py','gesture_model.py','preview_model.py','selective_preview.py','diagnostics/aligned_preview/model.npz']:
        protected[name]=sha(ROOT/name)
    plan={'methods':METHODS,'fixed_comparison':{'neighbors':3,'dtw_radius':4,'frames':32,'sample_rate':44100},
          'protocol':'Forced direction only: each query session is excluded from references. Secondary session/block/pace group exclusion. Negatives extracted and saved but not assigned directions.',
          'nested_selection':'For each outer held-out session, choose method using cross-session predictions between the other two sessions only; ties retain earlier method (raw first). No fitted weights.',
          'candidate_rule':'More total correct than raw, no session-level or pooled-class regression, and no grouped-total regression. Eligibility is for further development, not live deployment.',
          'no_calibration':'New distance scales invalidate existing novelty thresholds. No accepted-gesture or false-activation claims for candidates.',
          'sources':['https://librosa.org/doc/main/api/generated/librosa.feature.mfcc.html','https://librosa.org/doc/0.11.0/_modules/librosa/core/spectrum.html','https://github.com/librosa/librosa/blob/main/librosa/onset.py','https://arxiv.org/abs/1807.08869'],
          'limits':['All three sessions already inspected; these are reused-data development scores.','Three sessions from one person; PCM16 clipping is confounded with oldest session.','Candidate families share current crop, with512-point STFT for new spectral families; this is a bounded family comparison, not a single-variable ablation.','Multiscale Gaussian envelopes are not a full wavelet scattering implementation.','Feature selection even with nested reference exclusion does not undo prior design on these recordings.'],
          'protected_sha256':protected,'implementation_sha256':{name:sha(ROOT/name) for name in ['swipe_feature_families.py','probe_feature_families.py']}}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    cache={m:[] for m in METHODS};timings=[]
    for r in records:
        x,rate,_=decode_wav((ROOT/'data'/r['session']/r['file']).read_bytes(),array=True)
        before=x.copy();start=time.perf_counter();values=extract_all(x,rate);timings.append((time.perf_counter()-start)*1000)
        np.testing.assert_array_equal(x,before)
        for m in METHODS:cache[m].append(values[m])
    arrays={m:np.array(v) for m,v in cache.items()}
    np.savez_compressed(OUT/'features.npz',**arrays)
    (OUT/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    ids=np.array([i for i,r in enumerate(records) if r['label'] in ('up','down')]);sw=[records[i] for i in ids]
    labels=np.array([r['label'] for r in sw]);distances={};report={'methods':{},'limits':plan['limits']}
    for method in METHODS:
        seq=arrays[method][ids];start=time.perf_counter()
        d=np.array([batch_dtw(q,seq,4) for q in seq]);elapsed=(time.perf_counter()-start)*1000/len(seq)
        distances[method]=d;splits={}
        for split,key in [('cross_session','session'),('leave_group_out','group')]:
            rows=[]
            for i,r in enumerate(sw):
                refs=np.array([j for j,q in enumerate(sw) if q[key]!=r[key]])
                assert i not in refs and all(sw[j][key]!=r[key] for j in refs)
                p,margin,nearest=direction(d[i,refs],labels[refs],3)
                rows.append({**r,'truth':r['label'],'prediction':p,'margin':margin,'distance':nearest,'references':refs.tolist()})
            splits[split]={'summary':summarize(rows),'folds':{sid:summarize([r for r in rows if r['session']==sid]) for sid in SESSIONS},'rows':rows}
        report['methods'][method]={**splits,'dtw_ms_per_query_60_references':elapsed}
    base=report['methods']['raw'];eligible=[]
    for m,r in report['methods'].items():
        ok=m!='raw' and r['cross_session']['summary']['correct']>base['cross_session']['summary']['correct']
        ok &= all(r['cross_session']['folds'][s]['correct']>=base['cross_session']['folds'][s]['correct'] for s in SESSIONS)
        ok &= all(r['cross_session']['summary']['by_class'][c]['correct']>=base['cross_session']['summary']['by_class'][c]['correct'] for c in ('up','down'))
        ok &= r['leave_group_out']['summary']['correct']>=base['leave_group_out']['summary']['correct']
        r['eligible_for_further_study']=bool(ok)
        if ok:eligible.append(m)
    nested=[]
    for sid in SESSIONS:
        train=np.array([i for i,r in enumerate(sw) if r['session']!=sid]);scores={}
        for m in METHODS:
            score=0
            for i in train:
                refs=np.array([j for j in train if sw[j]['session']!=sw[i]['session']])
                score+=direction(distances[m][i,refs],labels[refs],3)[0]==labels[i]
            scores[m]=score
        chosen=max(METHODS,key=lambda m:scores[m]);outer=report['methods'][chosen]['cross_session']['folds'][sid]
        nested.append({'session':sid,'chosen':chosen,'inner_scores':scores,'outer':outer})
    report.update({'eligible':eligible,'nested_selection':nested,'nested_correct':sum(r['outer']['correct'] for r in nested),
        'feature_bundle_ms':{'median':float(np.median(timings)),'p95':float(np.quantile(timings,.95)),'note':'Cost of ALL seven features together; not optimized single-method live latency.'},
        'protected_unchanged':all(sha(ROOT/p)==h for p,h in protected.items()),'plan_sha256':sha(OUT/'plan.json')})
    assert report['protected_unchanged']
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for m,r in report['methods'].items():print(m,json.dumps({'cross':r['cross_session']['summary'],'session_correct':[r['cross_session']['folds'][s]['correct'] for s in SESSIONS],'grouped':r['leave_group_out']['summary']['correct'],'eligible':r['eligible_for_further_study']}))
    print('nested',report['nested_correct'],json.dumps(nested));print('timing',report['feature_bundle_ms'])

if __name__=='__main__':main()
