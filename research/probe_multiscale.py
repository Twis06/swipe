"""One fixed temporal-resolution consensus rescue; no fitting or threshold search."""
import json
from pathlib import Path
import numpy as np
from research.encoding_alignment import SESSIONS, GATES, reference_indices, sha
from swipe.audio_format import decode_wav
from swipe.aligned_preview import direction_features
from swipe.preview_model import features
from swipe.fast_preview import batch_dtw
from swipe.gesture_model import direction, decision

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/multiscale'

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'plan.json').exists():raise RuntimeError('Preserving completed experiment')
    protected=[ROOT/p for p in ['models/preview/model.npz','models/preview/engine.json','swipe/preview_service.py','swipe/selective_preview.py']]
    protected+=list((ROOT/'data').glob('*/session.json'))+list((ROOT/'data').glob('*/*.wav'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    plan={'candidate':'Keep active selective result unless uncertain. Rescue only if raw32 k1 accepts with existing cutoff/margin and raw16 (adjacent frame pair means) k1 agrees with margin0.2. Raw16 is a consistency vote only, without a novelty cutoff. Radius2 keeps the same relative warp allowance. Never change accepted or background results.',
          'methods':1,'fitting':False,'splits':['entire query session excluded','entire session/block/pace group excluded'],
          'gate':'Frozen other-session for cross-session; full mixed gate for descriptive grouped check.',
          'promotion':'Correct accepted count increases in aggregate, no per-session loss, no new accepted errors or negative activations in either split; stateful replay required before activation.',
          'limits':['Reused data, exploratory policy.','Grouped gate sees held-out groups.','Aligned legacy cutoff uses first-session labels.','Temporal resolutions are correlated, not independent evidence. The16-frame margin is an exploratory fixed check, not calibrated.'],
          'protected_sha256':hashes,'source_sha256':sha(Path(__file__))}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    records=[];raw=[];aligned=[];query_aligned=[];gates=[];quality=[];models={}
    for sid in SESSIONS:
        with np.load(ROOT/GATES[sid],allow_pickle=False) as b:models[sid]={k:b[k] for k in b.files}
        session=json.loads((ROOT/'data'/sid/'session.json').read_text())
        for c in session['clips']:
            if not c['accepted']:continue
            x,rate,encoding=decode_wav((ROOT/'data'/sid/c['filename']).read_bytes(),array=True)
            seq,g,q=features(x,rate,'raw')
            raw.append(seq);gates.append(g);quality.append(q)
            aligned.append(direction_features(x,rate,encoding))
            query_aligned.append(direction_features(x,rate,'float32'))
            records.append({'session':sid,'file':c['filename'],'label':c['label'],'group':f'{sid}_{c["block"]}_{c["pace"]}'})
    with np.load(ROOT/'models/preview/model.npz',allow_pickle=False) as b:mixed={k:b[k] for k in b.files}
    labels=np.array([r['label'] for r in records]);raw=np.array(raw);aligned=np.array(aligned)
    dr=np.array([batch_dtw(q,raw,4) for q in raw]);da=np.array([batch_dtw(q,aligned,4) for q in query_aligned])
    coarse=raw.reshape(len(raw),16,2,8).mean(axis=2)
    dc=np.array([batch_dtw(q,coarse,2) for q in coarse])
    report={'plan_sha256':sha(OUT/'plan.json'),'splits':{},'limits':plan['limits']};passed=True;gained=False
    def metrics(rows,key):
        return {'correct':sum(r[key]==r['label'] for r in rows if r['label'] in ('up','down')),
                'wrong':sum(r[key] in ('up','down') and r[key]!=r['label'] for r in rows if r['label'] in ('up','down')),
                'negative_activations':sum(r[key] in ('up','down') for r in rows if r['label'] not in ('up','down'))}
    for split in ('cross_session','leave_group_out'):
        rows=[]
        for i,r in enumerate(records):
            refs=reference_indices(i,records,split)
            model=models[next(s for s in SESSIONS if s!=r['session'])] if split=='cross_session' else mixed
            z=(gates[i]-model['gate_mean'])/model['gate_scale']
            score=float(1/(1+np.exp(-np.clip(float(z@model['gate_coef']+model['gate_intercept']),-700,700))))
            def classify(dist,k,cutoff):
                p,m,d=direction(dist[i,refs],labels[refs],k)
                return decision(p,m,d,cutoff,score,quality[i]['silent'],.2)
            raw3=classify(dr,3,float(model['distance_cutoff']));align3=classify(da,3,.3342443569796734)
            base=align3[0] if raw3[1]=='unfamiliar_sound' and align3[0] in ('up','down') else raw3[0]
            r1=classify(dr,1,float(model['distance_cutoff']));a1=classify(dc,1,float('inf'))
            candidate=r1[0] if base=='uncertain' and r1[0] in ('up','down') and r1[0]==a1[0] else base
            rows.append({**r,'baseline':base,'candidate':candidate,'raw_k1':r1,'coarse_k1':a1})
        old,new=metrics(rows,'baseline'),metrics(rows,'candidate')
        passed &= new['correct']>=old['correct'] and new['wrong']<=old['wrong'] and new['negative_activations']<=old['negative_activations']
        gained |= new['correct']>old['correct']
        report['splits'][split]={'baseline':old,'candidate':new,'changes':[r for r in rows if r['baseline']!=r['candidate']],'rows':rows}
    report['eligible_for_replay']=bool(passed and gained)
    report['protected_unchanged']=all(sha(ROOT/p)==h for p,h in hashes.items())
    assert report['protected_unchanged']
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='splits'}))
    for split,r in report['splits'].items():print(split,json.dumps({k:v for k,v in r.items() if k!='rows'}))

if __name__=='__main__':main()
