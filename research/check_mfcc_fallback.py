"""Reference-excluded check for the predeclared MFCC-first fallback policy."""
from pathlib import Path
import json,hashlib
import numpy as np
from swipe.audio_format import decode_wav
from swipe.mfcc_preview import load_model,predict as primary_predict
from swipe.mfcc_fallback import predict
from swipe.aligned_preview import load_model as load_aligned
from swipe.selective_preview import predict as legacy_predict

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/mfcc_preview/fallback_checks'

def summary(rows,key):
    sw=[r for r in rows if r['truth'] in ('up','down')];neg=[r for r in rows if r['truth'] not in ('up','down')]
    return {'correct':sum(r[key]==r['truth'] for r in sw),'wrong':sum(r[key] in ('up','down') and r[key]!=r['truth'] for r in sw),
            'rejected':sum(r[key] not in ('up','down') for r in sw),'negative_activations':sum(r[key] in ('up','down') for r in neg)}

def main():
    if (OUT/'report.json').exists():raise RuntimeError('Preserve completed evaluation')
    plan=json.loads((OUT/'plan.json').read_text())
    assert hashlib.sha256((ROOT/'swipe/mfcc_fallback.py').read_bytes()).hexdigest()==plan['source_sha256']
    model,config=load_model();aligned,_=load_aligned()
    rejection=json.loads((ROOT/'diagnostics/mfcc_components_v1/rejection/report.json').read_text())['methods']['cepstra8']
    sids=config['sessions'];paths={sids[0]:'diagnostics/calibration/20261103T200050_87dcb0/pilot_model.npz',sids[1]:'diagnostics/improvement/noise_aware/pilot_model.npz',sids[2]:'models/preview/model.npz'}
    rows=[]
    for sid in sids:
        with np.load(ROOT/paths[sid]) as bank:raw={k:bank[k] for k in bank.files}
        raw['feature_input']=np.array('raw')
        candidate={k:v.copy() for k,v in model.items()}
        mask=np.array([r['session']!=sid for r in config['references']]);candidate['templates']=model['templates'][mask];candidate['labels']=model['labels'][mask]
        candidate['distance_cutoff']=np.array(rejection['cutoffs'][sid])
        for key in ['gate_mean','gate_scale','gate_coef','gate_intercept']:candidate[key]=raw[key]
        amask=aligned['reference_sessions']!=sid
        legacy_aligned={k:v.copy() for k,v in aligned.items()}
        legacy_aligned['templates']=aligned['templates'][amask];legacy_aligned['labels']=aligned['labels'][amask]
        for key in ['gate_mean','gate_scale','gate_coef','gate_intercept']:legacy_aligned[key]=raw[key]
        for expected in rejection['rows']:
            if expected['session']!=sid:continue
            x,rate,_=decode_wav((ROOT/'data'/sid/expected['file']).read_bytes(),array=True)
            primary=primary_predict(x,rate,candidate)
            assert primary['prediction']==expected['prediction']
            np.testing.assert_allclose(primary['nearest_distance'],expected['distance'],atol=1e-10)
            legacy=legacy_predict(x,rate,raw,legacy_aligned);hybrid=predict(x,rate,candidate,raw,legacy_aligned)
            rows.append({'session':sid,'file':expected['file'],'truth':expected['truth'],'legacy':legacy['prediction'],
                         'primary':primary['prediction'],'hybrid':hybrid['prediction']})
    report={'summary':{k:summary(rows,k) for k in ['legacy','primary','hybrid']},
            'folds':{sid:{k:summary([r for r in rows if r['session']==sid],k) for k in ['legacy','primary','hybrid']} for sid in sids},'rows':rows,
            'limits':['Reused development data; not fresh accuracy.','Legacy first-session cutoff remains a previously estimated heuristic.','Legacy banks contain only eligible older sessions; MFCC contains both eligible other sessions.']}
    report['passes']=all(v['hybrid']['correct']>=v['legacy']['correct'] and v['hybrid']['wrong']<=v['legacy']['wrong'] and v['hybrid']['negative_activations']<=v['legacy']['negative_activations'] for v in report['folds'].values())
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
