"""Bounded post-validation diagnosis. No fitting, selection or deployment."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'diagnostics/prospective_validation/20261103_guided_validation_01'
OUT=BASE/'investigation_20261107'
SID='20261107T185348_9cbd41'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    frozen=BASE/'frozen'
    initial=json.loads((BASE/'plan.json').read_text())
    for name,digest in initial['frozen_sha256'].items():assert sha(frozen/name)==digest
    sys.path.insert(0,str(frozen))
    from preview_service import PreviewService
    from audio_format import decode_wav
    from fast_preview import batch_dtw
    from preview_model import features
    from gesture_model import direction,decision,canonical_audio,RATE,BANDS
    from scipy.signal import stft
    service=PreviewService();model=service.model
    refs=json.loads((frozen/'diagnostics/aligned_preview/model.json').read_text())['references']
    paths=[ROOT/'data'/r['session']/r['file'] for r in refs]
    session=json.loads((ROOT/'data'/SID/'session.json').read_text())
    clips=[c for c in session['clips'] if c['accepted'] and c['label'] in ('up','down')]
    paths += [ROOT/'data'/SID/c['filename'] for c in session['clips']]
    paths += [ROOT/'data'/SID/'session.json',BASE/'evaluations'/SID/'report.json']
    protected={str(p.relative_to(ROOT)):sha(p) for p in paths}
    plan={'hypotheses':['New-session direction shape differs from old references: compare old nearest matches with opposite-block new-session references.','Energy crop includes weak tails or reset sounds: compare two fixed destructive20ms core-boundary trims, keeping original noise estimate and gate.','k3 voting contradicts nearest class: compare frozen k3 with k1 using same distances and rejection checks.'],
          'fixed_tests':['baseline','k1','trim_core_start20ms','trim_core_end20ms','new_session_opposite_block'],
          'no_fitting':True,'no_deployment':True,'source_sha256':sha(Path(__file__)),
          'protected_sha256':protected,'limits':['New-session data have now been inspected: probes are explanatory, not independent validation.','Opposite-block comparisons reuse this session and original frozen gate/cutoff; rejection counts descriptive only.','Trim variants alter duration and omit context; sensitivity cannot prove taps or reset sounds.','One user; cue labels are not independent motion ground truth.']}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    refquality=[]
    for i,r in enumerate(refs):
        x,rate,_=decode_wav((ROOT/'data'/r['session']/r['file']).read_bytes(),array=True)
        seq,_,q=features(x,rate,'raw')
        np.testing.assert_array_equal(seq,model['templates'][i])
        assert r['label']==model['labels'][i]
        refquality.append(q)
    rows=[];sequences=[];signals=[]
    def with_bounds(x,left,right):
        normalized=x/max(float(np.max(np.abs(x))),1e-30)
        event=normalized[max(0,left):min(len(x),right)]
        if len(event)<256:event=np.pad(event,(0,256-len(event)))
        f,_,z=stft(event,fs=RATE,nperseg=256,noverlap=192,boundary=None)
        _,_,full=stft(normalized,fs=RATE,nperseg=256,noverlap=192,boundary=None)
        spectrum=np.maximum(abs(z)**2-np.median(abs(full)**2,axis=1)[:,None]/np.log(2),0)
        bands=np.stack([spectrum[(f>=a)&(f<b)].sum(axis=0) for a,b in zip(BANDS[:-1],BANDS[1:])],axis=1)
        logged=np.log10(bands/max(float(bands.max()),1e-30)+1e-5)
        return np.array([np.interp(np.linspace(0,1,32),np.linspace(0,1,len(logged)),logged[:,i]) for i in range(8)]).T
    for clip in clips:
        blob=(ROOT/'data'/SID/clip['filename']).read_bytes()
        x,rate,_=decode_wav(blob,array=True);seq,_,q=features(x,rate,'raw')
        active=service.classify(blob);dist=batch_dtw(seq,model['templates'],4)
        canonical=canonical_audio(x,rate);start=round(q['event_start_seconds']*RATE);end=round(q['event_end_seconds']*RATE);pad=round(.02*RATE)
        original=with_bounds(canonical,start-pad,end+pad)
        np.testing.assert_allclose(original,seq,atol=1e-12,rtol=1e-12)
        def classify(ds,labels=model['labels'],k=3):
            p,m,d=direction(ds,labels,k)
            result,reason=decision(p,m,d,float(model['distance_cutoff']),active['swipe_gate_score'],q['silent'],float(model['margin_threshold']))
            return {'prediction':result,'reason':reason,'candidate':p,'margin':m,'distance':d}
        variants={'baseline':classify(dist),'k1':classify(dist,k=1)}
        for name,left,right in [('trim_core_start20ms',start+pad,end+pad),('trim_core_end20ms',start-pad,end-pad)]:
            variants[name]=classify(batch_dtw(with_bounds(canonical,left,right),model['templates'],4)) if right-left>=256 else None
        order=np.argsort(dist,kind='stable')[:5]
        row={'file':clip['filename'],'truth':clip['label'],'block':clip['block'],'pace':clip['pace'],'quality':q,
             'active_prediction':active['prediction'],'variants':variants,
             'nearest5':[{**refs[j],'distance':float(dist[j])} for j in order],
             'old_nearest_same':float(np.min(dist[model['labels']==clip['label']])),
             'old_nearest_opposite':float(np.min(dist[model['labels']!=clip['label']]))}
        rows.append(row);sequences.append(seq);signals.append((canonical,q))
    sequences=np.array(sequences);labels=np.array([r['truth'] for r in rows])
    for i,row in enumerate(rows):
        eligible=np.array([j for j,r in enumerate(rows) if r['block']!=row['block']])
        ds=batch_dtw(sequences[i],sequences[eligible],4)
        p,m,d=direction(ds,labels[eligible],3)
        # Direction-only comparison: no new fitted gate or distance cutoff.
        row['new_block_direction']={'candidate':p,'margin':m,'distance':d,
            'nearest_same':float(np.min(ds[labels[eligible]==row['truth']])),
            'nearest_opposite':float(np.min(ds[labels[eligible]!=row['truth']])),
            'nearest_file':rows[eligible[np.argmin(ds)]]['file']}
    summaries={}
    for variant in ('baseline','k1','trim_core_start20ms','trim_core_end20ms'):
        valid=[r for r in rows if r['variants'][variant] is not None]
        summaries[variant]={'correct':sum(r['variants'][variant]['prediction']==r['truth'] for r in valid),
            'wrong':sum(r['variants'][variant]['prediction'] in ('up','down') and r['variants'][variant]['prediction']!=r['truth'] for r in valid),
            'rejected':sum(r['variants'][variant]['prediction']=='uncertain' for r in valid),'evaluated':len(valid),
            'forced_correct':sum(r['variants'][variant]['candidate']==r['truth'] for r in valid)}
    summaries['new_opposite_block']={'forced_correct':sum(r['new_block_direction']['candidate']==r['truth'] for r in rows),
        'by_class':{label:sum(r['truth']==label and r['new_block_direction']['candidate']==label for r in rows) for label in ('up','down')}}
    profile={}
    for label in ('up','down'):
        group=[r for r in rows if r['truth']==label]
        profile[label]={'old_nearest_same_median':float(np.median([r['old_nearest_same'] for r in group])),
            'new_opposite_block_same_median':float(np.median([r['new_block_direction']['nearest_same'] for r in group])),
            'duration_median_ms':float(np.median([r['quality']['energy_duration_ms'] for r in group])),
            'old_duration_median_ms':float(np.median([q['energy_duration_ms'] for r,q in zip(refs,refquality) if r['label']==label])),
            'k3_disagrees_with_nearest':sum(r['variants']['baseline']['candidate']!=r['nearest5'][0]['label'] for r in group)}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,4,figsize=(14,12),sharex=True)
    for ax,row,(x,q) in zip(axes.flat,rows,signals):
        width=441;n=len(x)//width;en=np.sqrt(np.mean(x[:n*width].reshape(n,width)**2,axis=1));en/=max(en.max(),1e-30)
        ax.plot(np.arange(n)*.01,en,lw=1,color='#25634b')
        ax.axvspan(q['event_start_seconds'],q['event_end_seconds'],alpha=.14,color='#b78123')
        ax.set_title(f"{row['file'][:3]} {row['truth']} → {row['active_prediction']} ({row['pace']})",fontsize=9)
        ax.set_ylim(0,1.1);ax.set_xlim(0,2);ax.set_xlabel('seconds');ax.set_ylabel('normalized RMS')
    fig.suptitle('All new-session swipes: 10 ms RMS envelope and frozen energy crop',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'waveforms.png',dpi=140);plt.close(fig)
    report={'summary':summaries,'profiles':profile,'rows':rows,'limits':plan['limits'],'protected_unchanged':all(sha(ROOT/p)==h for p,h in protected.items())}
    assert report['protected_unchanged']
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
