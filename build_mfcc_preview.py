"""Build a separate deterministic reference bank and reference-only cutoff."""
from pathlib import Path
import hashlib,json
import numpy as np
from fast_preview import batch_dtw
from probe_mfcc_rejection import cutoff_for_references

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'diagnostics/mfcc_preview'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    source=ROOT/'diagnostics/mfcc_components_v1'
    records=json.loads((source/'records.json').read_text())
    ids=np.array([i for i,r in enumerate(records) if r['label'] in ('up','down')])
    refs=[records[i] for i in ids]
    source_plan=json.loads((source/'plan.json').read_text())
    for name,digest in source_plan['protected_sha256'].items():assert sha(ROOT/name)==digest
    rejection_plan=json.loads((source/'rejection/plan.json').read_text())
    assert sha(source/'features.npz')==rejection_plan['feature_cache_sha256']
    plan={'variant':'cepstra8','reference_swipes':60,'sessions':sorted({r['session'] for r in refs}),
          'calibration':'95th percentile of nearest same-class distance to another session for each of60 references; fixed margin.2/k3/radius4. Gate copied unchanged from old mixed model.',
          'scope':'Development integration only; reference examples are not independent validation. No live changes.',
          'feature_cache_sha256':sha(source/'features.npz'),'source_model_sha256':sha(ROOT/'diagnostics/preview/model.npz')}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with np.load(source/'features.npz') as bank:templates=bank['cepstra8'][ids]
    labels=np.array([r['label'] for r in refs]);sessions=np.array([r['session'] for r in refs])
    d=np.array([batch_dtw(q,templates,4) for q in templates])
    cutoff=cutoff_for_references(d,labels,sessions,np.arange(len(refs)))
    with np.load(ROOT/'diagnostics/preview/model.npz') as bank:model={k:bank[k] for k in bank.files}
    model.update(templates=templates,labels=labels,distance_cutoff=np.array(cutoff),feature_input=np.array('cepstra8'))
    np.savez_compressed(OUT/'model.npz',**model)
    config={**plan,'references':refs,'distance_cutoff':cutoff,'model_sha256':sha(OUT/'model.npz'),
            'implementation_sha256':{name:sha(ROOT/name) for name in ['mfcc_preview.py','mfcc_components.py','swipe_feature_families.py','fast_preview.py','gesture_model.py','preview_model.py']}}
    (OUT/'model.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({'reference_swipes':len(refs),'cutoff':cutoff,'live_changed':False}))

if __name__=='__main__':main()
