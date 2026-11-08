"""Bounded, exploratory comparison on the original 20 swipe clips.

No model is deployed and no raw data is changed. See diagnostics/plan.json.
Run with .venv/bin/python diagnose_audio.py.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/swipe-matplotlib')
from pathlib import Path
import json
import numpy as np
from scipy.signal import stft, find_peaks
from scipy.spatial.distance import cdist
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.analyze import read_wav, features

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/20261103T165139_cf7201'
OUT = ROOT/'diagnostics'
OUT.mkdir(exist_ok=True)
BANDS = np.array([100,250,500,1000,2000,4000,8000,16000,22051])

def interpolate(sequence, length=32):
    return np.array([np.interp(np.linspace(0,1,length), np.linspace(0,1,len(sequence)), sequence[:,i]) for i in range(sequence.shape[1])]).T

def event_data(x, sr):
    cumulative = np.cumsum(x*x)
    a,b = np.searchsorted(cumulative/cumulative[-1],[.01,.99])
    # Pad the 98%-energy core by 20 ms on each side; no label-dependent alignment.
    left,right = max(0,a-int(sr*.02)), min(len(x),b+int(sr*.02))
    event=x[left:right]
    f,t,z=stft(event,fs=sr,nperseg=256,noverlap=192,boundary=None)
    power=abs(z)**2
    bands=np.stack([power[(f>=lo)&(f<hi)].sum(axis=0) for lo,hi in zip(BANDS[:-1],BANDS[1:])],axis=1)
    # Remove overall pressure/loudness while retaining relative spectral changes.
    log_bands=np.log10(bands/(bands.max()+1e-12)+1e-5)
    sequence=interpolate(log_bands)
    frames=np.array_split(event,32)
    envelope=np.array([np.sqrt(np.mean(a*a)) for a in frames])
    envelope/=np.linalg.norm(envelope)+1e-12
    return event,sequence,envelope,{'energy_start_seconds':float(a/sr),'energy_end_seconds':float(b/sr),'energy_duration_ms':float((b-a)/sr*1000),
        'clipped_fraction_in_energy_core':float(np.mean((x[a:b]<=-32767/32768)|(x[a:b]>=32766/32768))),
        'first_100ms_energy_fraction':float(np.sum(x[:sr//10]**2)/cumulative[-1]),
        'last_100ms_energy_fraction':float(np.sum(x[-sr//10:]**2)/cumulative[-1])}

def dtw(a,b):
    d=cdist(a,b,'sqeuclidean')/a.shape[1]
    cost=np.full((len(a)+1,len(b)+1),np.inf); cost[0,0]=0
    for i in range(1,len(a)+1):
        for j in range(max(1,i-8),min(len(b),i+8)+1):
            cost[i,j]=d[i-1,j-1]+min(cost[i-1,j],cost[i,j-1],cost[i-1,j-1])
    return cost[-1,-1]/(len(a)+len(b))

def main():
    session=json.loads((SOURCE/'session.json').read_text())
    clips=[c for c in session['clips'] if c['accepted'] and c['label'] in ['up','down']]
    whole,cropped,seqs,envelopes,rows,signals=[],[],[],[],[],[]
    for c in clips:
        x,sr,raw=read_wav(SOURCE/c['filename'])
        event,seq,env,info=event_data(x,sr)
        whole.append(features(x,sr));cropped.append(features(event,sr));seqs.append(seq);envelopes.append(env)
        rows.append({'file':c['filename'],'label':c['label'],'block':c['block'],'pace':c['pace'],**info})
        signals.append((x,sr,event))
    y=np.array([c['label'] for c in clips]); blocks=np.array([c['block'] for c in clips])
    groups=np.array([f'{c["block"]}_{c["pace"]}' for c in clips])
    whole,cropped,seqs,envelopes=map(np.array,[whole,cropped,seqs,envelopes])
    distance_dtw=np.zeros((len(y),len(y)))
    for i in range(len(y)):
        for j in range(i):distance_dtw[i,j]=distance_dtw[j,i]=dtw(seqs[i],seqs[j])
    candidates={
      'whole_clip_ridge':('ridge',whole),
      'event_cropped_ridge':('ridge',cropped),
      'normalized_envelope_nn':('distance',cdist(envelopes,envelopes,'cosine')),
      'event_time_frequency_ridge':('ridge',seqs.reshape(len(y),-1)),
      'event_time_frequency_nn':('distance',cdist(seqs.reshape(len(y),-1),seqs.reshape(len(y),-1),'cosine')),
      'band_shape_dtw_nn':('distance',distance_dtw)}
    def predict(name,train,test,labels=y):
        kind,data=candidates[name]
        if kind=='ridge':
            model=make_pipeline(StandardScaler(),RidgeClassifier(alpha=10))
            return model.fit(data[train],labels[train]).predict(data[test])
        return labels[train[np.argmin(data[np.ix_(test,train)],axis=1)]]
    report={'status':'Exploratory; same pilot reused after inspection. No independent validation.', 'hypothesis_checks':{},'models':{},'clips':rows}
    for name in candidates:
        splits={}
        for trainblock,testblock in [(1,2),(2,1)]:
            a,b=np.where(blocks==trainblock)[0],np.where(blocks==testblock)[0]
            pred=predict(name,a,b)
            splits[f'block_{trainblock}_to_{testblock}']={'correct':int(np.sum(pred==y[b])),'total':len(b),'predictions':[{'file':clips[i]['filename'],'truth':str(y[i]),'predicted':str(p)} for i,p in zip(b,pred)]}
        preds=np.empty(len(y),dtype='<U4')
        for group in np.unique(groups):
            a,b=np.where(groups!=group)[0],np.where(groups==group)[0]
            preds[b]=predict(name,a,b)
        splits['leave_block_pace_group_out']={'correct':int(np.sum(preds==y)),'total':len(y), 'predictions':list(map(str,preds))}
        report['models'][name]=splits
    # Nested selection: choose among fixed candidates using only each outer
    # training set. No preselection of the winner using outer test labels.
    def nested_score(labels, details=False):
        out=np.empty(len(y),dtype='<U4'); choices=[]
        for group in np.unique(groups):
            train,test=np.where(groups!=group)[0],np.where(groups==group)[0]
            scores={}
            for name in candidates:
                correct=0
                for inner in np.unique(groups[train]):
                    a=train[groups[train]!=inner];b=train[groups[train]==inner]
                    correct+=int(np.sum(predict(name,a,b,labels)==labels[b]))
                scores[name]=correct
            winner=max(scores,key=scores.get)
            out[test]=predict(winner,train,test,labels)
            choices.append({'held_out_group':str(group),'selected_model':winner,'correct':int(np.sum(out[test]==labels[test])),'total':len(test)})
        return int(np.sum(out==labels)),choices
    score,choices=nested_score(y)
    report['nested_selection']={'correct':score,'total':len(y),'folds':choices,'note':'Selection inside outer folds; still exploratory because feature families were chosen after inspecting this session.'}
    # Label-shuffle check for fixed NN candidates, preserving block/pace
    # strata. Maximum across the three choices accounts for choosing a winner.
    names=[n for n,(k,_) in candidates.items() if k=='distance']
    neighbors={n:np.empty(len(y),dtype=int) for n in names}
    for n in names:
        d=candidates[n][1]
        for g in np.unique(groups):
            tr,te=np.where(groups!=g)[0],np.where(groups==g)[0]
            neighbors[n][te]=tr[np.argmin(d[np.ix_(te,tr)],axis=1)]
    observed=max(np.sum(y[neighbors[n]]==y) for n in names)
    rng=np.random.default_rng(20261103); exceed=0; permutations=2000
    for _ in range(permutations):
        perm=y.copy()
        for g in np.unique(groups):
            ix=np.where(groups==g)[0];perm[ix]=rng.permutation(perm[ix])
        exceed+=max(np.sum(perm[neighbors[n]]==perm) for n in names)>=observed
    report['nearest_neighbor_shuffle_check']={'permutations':permutations,'observed_best_correct':int(observed),'total':len(y),'max_statistic_p':(exceed+1)/(permutations+1),'scope':'Three fixed nearest-neighbor families; block/pace-stratified label permutations. Does not correct for all researcher choices.'}
    report['hypothesis_checks']={
       'window_truncation':'All energy cores are inside the window; inspect overview plot for boundary evidence.',
       'clip_core_duration_ms_range':[min(r['energy_duration_ms'] for r in rows),max(r['energy_duration_ms'] for r in rows)],
       'clipped_core_fraction_range':[min(r['clipped_fraction_in_energy_core'] for r in rows),max(r['clipped_fraction_in_energy_core'] for r in rows)],
       'encoder':'Legacy encoder explicitly clamps samples to [-1,1]. Saved integer files cannot identify original float excursions or separate encoder from upstream saturation.'}
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(10,2,figsize=(14,18),sharex=True,layout='constrained')
    for label,col in [('up',0),('down',1)]:
        for ax,idx in zip(axes[:,col],np.where(y==label)[0]):
            x,sr,event=signals[idx];r=rows[idx];t=np.arange(len(x))/sr
            ax.plot(t[::16],x[::16],lw=.55,color='#14664e' if label=='up' else '#b15b32')
            ax.axvspan(r['energy_start_seconds'],r['energy_end_seconds'],alpha=.12,color='gray')
            ax.set_ylim(-1.1,1.1);ax.set_yticks([-1,0,1]);ax.set_title(f'{r["file"]} | {r["energy_duration_ms"]:.0f} ms | {100*r["clipped_fraction_in_energy_core"]:.0f}% core clipped',fontsize=9,loc='left')
    axes[-1,0].set_xlabel('Seconds after capture starts');axes[-1,1].set_xlabel('Seconds after capture starts')
    fig.suptitle('Existing pilot: the swipe is a short burst near the middle of each 2-second clip\nShading contains 98% of energy. Peaks at ±1 are saturated in the saved file.',fontsize=15)
    fig.savefig(OUT/'waveforms.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for col,label in enumerate(['up','down']):
        idx=np.where(y==label)[0][0];x,sr,event=signals[idx];r=rows[idx]
        peak=np.argmax(abs(np.diff(event)));a=max(0,peak-100)
        axes[0,col].plot(np.arange(a,min(a+300,len(event)))/sr*1000,event[a:a+300],lw=1)
        axes[0,col].axhline(1,color='r',alpha=.4);axes[0,col].axhline(-1,color='r',alpha=.4)
        axes[0,col].set_title(f'{label.capitalize()} example: flat-topped waveform');axes[0,col].set_xlabel('Milliseconds within cropped event')
        axes[1,col].imshow(seqs[idx].T,origin='lower',aspect='auto',cmap='magma',vmin=-5,vmax=0)
        axes[1,col].set_yticks(range(8),['100–250','250–500','500–1k','1–2k','2–4k','4–8k','8–16k','16–22k'])
        axes[1,col].set_xlabel('Normalized event time');axes[1,col].set_ylabel('Frequency band (Hz)')
    fig.suptitle('Clipping diagnosis and time-resolved features',fontsize=15)
    fig.savefig(OUT/'clipping_and_features.png',dpi=150);plt.close(fig)
    print(json.dumps({'models':{n:{s:{k:v for k,v in result.items() if k!='predictions'} for s,result in splits.items()} for n,splits in report['models'].items()},'nested_selection':report['nested_selection'],'shuffle':report['nearest_neighbor_shuffle_check'],'quality':report['hypothesis_checks']},indent=2))

if __name__=='__main__':main()
