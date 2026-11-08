"""Reference-only novelty calibration; frozen gates exclude each query session."""
from pathlib import Path
import hashlib,json
import numpy as np
from audio_format import decode_wav
from preview_model import features
from fast_preview import batch_dtw
from gesture_model import direction,decision
from mfcc_components import METHODS

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'diagnostics/mfcc_components_v1'
OUT=BASE/'rejection'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def cutoff_for_references(distances,labels,sessions,refs):
    nearest=[]
    for i in refs:
        others=[j for j in refs if sessions[j]!=sessions[i] and labels[j]==labels[i]]
        if not others:raise ValueError('Calibration needs another reference session for each direction.')
        nearest.append(float(np.min(distances[i,others])))
    return float(np.quantile(nearest,.95))

def summarize(rows):
    sw=[r for r in rows if r['truth'] in ('up','down')];neg=[r for r in rows if r['truth'] not in ('up','down')]
    return {'correct_accepted':sum(r['prediction']==r['truth'] for r in sw),
            'wrong_accepted':sum(r['prediction'] in ('up','down') and r['prediction']!=r['truth'] for r in sw),
            'rejected_swipes':sum(r['prediction'] not in ('up','down') for r in sw),
            'negative_activations':sum(r['prediction'] in ('up','down') for r in neg),'swipes':len(sw),'negatives':len(neg)}

def main():
    OUT.mkdir(exist_ok=False)
    records=json.loads((BASE/'records.json').read_text());sessions=np.array([r['session'] for r in records]);labels=np.array([r['label'] for r in records]);sids=sorted(set(sessions))
    gate_paths={sids[0]:'diagnostics/calibration/20261103T200050_87dcb0/pilot_model.npz',sids[1]:'diagnostics/improvement/noise_aware/pilot_model.npz',sids[2]:'diagnostics/preview/model.npz'}
    protected={name:sha(ROOT/name) for name in list(gate_paths.values())+['preview_service.py','diagnostics/preview/engine.json']}
    plan={'cutoff':'For each outer query session, use only the other two sessions. Each reference swipe finds its closest same-direction example in the other reference session. Novelty cutoff is fixed95th percentile of those40 distances.',
          'margin':.2,'neighbors':3,'radius':4,'gate_paths_by_query_session':gate_paths,
          'gate_note':'All three gate choices were fitted earlier without their query session. No new gate or classifier fitting.',
          'calibration':'Novelty thresholds are newly estimated from reference labels only; no threshold grid search.',
          'variants':list(METHODS),'scope':'All60 swipes and24 negatives; hold out entire query session from feature bank, cutoff and gate.',
          'limits':['Exploratory reused-data check; feature design has already seen these recordings.','Existing margin0.2 is a fixed heuristic, not a probability.','Twenty-four idle/typing clips do not represent all background sounds.'],
          'protected_sha256':protected,'feature_cache_sha256':sha(BASE/'features.npz'),'source_sha256':sha(Path(__file__))}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    gate_scores=[];silences=[]
    models={}
    for sid,path in gate_paths.items():
        with np.load(ROOT/path,allow_pickle=False) as bank:models[sid]={k:bank[k] for k in bank.files}
    for r in records:
        x,rate,_=decode_wav((ROOT/'data'/r['session']/r['file']).read_bytes(),array=True)
        _,g,q=features(x,rate,'raw');model=models[r['session']]
        z=(g-model['gate_mean'])/model['gate_scale']
        gate_scores.append(float(1/(1+np.exp(-np.clip(float(z@model['gate_coef']+model['gate_intercept']),-700,700)))))
        silences.append(q['silent'])
    report={'methods':{},'limits':plan['limits']}
    with np.load(BASE/'features.npz') as bank:
        for method in METHODS:
            seq=bank[method];d=np.array([batch_dtw(q,seq,4) for q in seq]);rows=[];cutoffs={}
            for sid in sids:
                refs=np.where((sessions!=sid)&np.isin(labels,['up','down']))[0];queries=np.where(sessions==sid)[0]
                cutoff=cutoff_for_references(d,labels,sessions,refs);cutoffs[sid]=cutoff
                # Changing every held-out label/distance must not alter calibration.
                changed=d.copy();changed[queries,:]=1e9;changed[:,queries]=1e9
                wrong_labels=labels.copy();wrong_labels[queries]='up'
                assert cutoff_for_references(changed,wrong_labels,sessions,refs)==cutoff
                for i in queries:
                    p,m,near=direction(d[i,refs],labels[refs],3)
                    pred,reason=decision(p,m,near,cutoff,gate_scores[i],silences[i],.2)
                    rows.append({**records[i],'truth':labels[i],'prediction':pred,'reason':reason,'candidate':p,'margin':m,'distance':near,'cutoff':cutoff,'gate_score':gate_scores[i]})
            report['methods'][method]={'summary':summarize(rows),'folds':{sid:summarize([r for r in rows if r['session']==sid]) for sid in sids},'cutoffs':cutoffs,'rows':rows}
    report['protected_unchanged']=all(sha(ROOT/n)==h for n,h in protected.items())
    assert report['protected_unchanged']
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,r in report['methods'].items():print(name,json.dumps({'summary':r['summary'],'folds':r['folds']}))

if __name__=='__main__':main()
