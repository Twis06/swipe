"""Bounded, training-free recurrence probe; see diagnostics/continuous/motifs/plan.json."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/swipe-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
from scipy.spatial.distance import cdist
from audio_format import decode_wav

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'diagnostics/continuous/motifs'
BANDS = np.array([100,250,500,1000,2000,4000,8000,16000,24001])
HOP = .02


def features(x, rate):
    x = np.asarray(x, dtype=np.float64)
    n = round(rate*1024/48000)
    hop = round(rate*HOP)
    f,t,z = stft(x, fs=rate, nperseg=n, noverlap=n-hop, boundary=None, padded=False)
    powers = np.stack([(np.abs(z[(f>=a)&(f<b)])**2).sum(axis=0)
                       for a,b in zip(BANDS[:-1],BANDS[1:])], axis=1)
    # Relative floor preserves amplitude-scale invariance of the log features.
    floor = max(float(powers.max())*1e-12, np.finfo(float).tiny)
    logged = np.log(np.maximum(powers, floor))
    rms = np.array([np.sqrt(np.mean(x[i:i+n]**2)) for i in range(0,len(x)-n+1,hop)])
    return t,logged,rms


def subsequences(logged, rms, length):
    indices = np.arange(len(logged)-length+1)
    active = np.array([rms[i:i+length].max() > max(10*np.median(rms), 1e-15) for i in indices])
    indices = indices[active]
    if not len(indices):
        return indices, np.empty((0,length,logged.shape[1]))
    values = np.stack([logged[i:i+length] for i in indices])
    mean = values.mean(axis=(1,2), keepdims=True)
    scale = np.maximum(values.std(axis=(1,2), keepdims=True), 1e-12)
    return indices, (values-mean)/scale


def join(a,b):
    return cdist(a.reshape(len(a),-1), b.reshape(len(b),-1), 'sqeuclidean')/np.prod(a.shape[1:])


def disjoint_matches(distances, ia, ib, length, count=5):
    chosen=[]
    for flat in np.argsort(distances,axis=None,kind='stable'):
        i,j=np.unravel_index(flat,distances.shape)
        if any(abs(int(ia[i])-x['start_frame_a'])<length or abs(int(ib[j])-x['start_frame_b'])<length for x in chosen):
            continue
        chosen.append({'start_frame_a':int(ia[i]),'start_frame_b':int(ib[j]),'distance':float(distances[i,j])})
        if len(chosen)==count:break
    return chosen


def main():
    plan=json.loads((OUT/'plan.json').read_text())
    ids=json.loads((ROOT/'diagnostics/continuous/label_review.json').read_text())['take_ids']
    takes=[]
    for sid in ids:
        blob=(ROOT/'data'/sid/'continuous.wav').read_bytes()
        x,rate,_=decode_wav(blob)
        t,logged,rms=features(x,rate)
        gain_errors=[]
        for gain in (.1,10):
            _,gain_logged,gain_rms=features(np.asarray(x)*gain,rate)
            per_length=[]
            for duration in plan['durations_seconds']:
                length=round(duration/HOP)
                index,v=subsequences(logged,rms,length)
                gi,gv=subsequences(gain_logged,gain_rms,length)
                if not np.array_equal(index,gi):raise AssertionError('Gain changed activity candidates')
                per_length.append(float(np.max(np.abs(v-gv))))
            gain_errors.append({'gain':gain,'max_normalized_feature_error':max(per_length)})
        takes.append({'id':sid,'sha256':hashlib.sha256(blob).hexdigest(),'sample_rate':rate,
                      'gain_checks':gain_errors,'t':t,'logged':logged,'rms':rms})
    report={'scope':plan['scope'],'plan_sha256':hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'takes':[{k:v for k,v in take.items() if k not in ('t','logged','rms')} for take in takes],
            'joins':[]}
    for duration in plan['durations_seconds']:
        length=round(duration/HOP)
        permutation=np.random.default_rng(20261103+length).permutation(length)
        for ai,bi in ((0,1),(0,2),(1,2)):
            a,b=takes[ai],takes[bi]
            ia,va=subsequences(a['logged'],a['rms'],length)
            ib,vb=subsequences(b['logged'],b['rms'],length)
            d=join(va,vb)
            row={'take_a':ai+1,'take_b':bi+1,'duration_seconds':duration,
                 'source_ids':[a['id'],b['id']],'candidate_counts':[len(ia),len(ib)],
                 'median_nearest_distance':float(np.median(d.min(axis=1))),
                 'reversed_target_median_nearest_distance':float(np.median(join(va,vb[:,::-1]).min(axis=1))),
                 'permuted_target_median_nearest_distance':float(np.median(join(va,vb[:,permutation]).min(axis=1))),
                 'representative_matches':disjoint_matches(d,ia,ib,length)}
            for match in row['representative_matches']:
                for side,take in (('a',a),('b',b)):
                    start=match[f'start_frame_{side}']
                    match[f'start_seconds_{side}']=float(take['t'][start]-(round(take['sample_rate']*1024/48000)/2)/take['sample_rate'])
                    match[f'end_seconds_{side}']=match[f'start_seconds_{side}']+duration
                    peak=start+int(np.argmax(take['rms'][start:start+length]))
                    match[f'peak_seconds_{side}']=float(take['t'][peak])
            report['joins'].append(row)
    stability=[]
    for ai,bi in ((1,2),(1,3),(2,3)):
        rows=[r for r in report['joins'] if (r['take_a'],r['take_b'])==(ai,bi)]
        base=rows[1]['representative_matches']
        for other in (rows[0],rows[2]):
            matches=sum(any(abs(m['peak_seconds_a']-n['peak_seconds_a'])<=.1 and abs(m['peak_seconds_b']-n['peak_seconds_b'])<=.1
                            for n in other['representative_matches']) for m in base)
            stability.append({'pair':[ai,bi],'reference_duration':.4,'comparison_duration':other['duration_seconds'],
                              'matching_peak_pairs_within_100ms':matches,'reference_pairs':len(base)})
    report['duration_stability']=stability
    report['interpretation_limits']=plan['limitations']+' Gain invariance is mathematical by construction, not evidence of real pressure/device invariance. Different gains do not model clipping. Temporal controls are not inferential nulls. Recurrence can reflect generic impact, lifting, contact, attack/decay, or partial-event edges rather than swipe identity.'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Plot the strongest .4-second pair for each take-pair, fixed display duration.
    fig,axes=plt.subplots(3,2,figsize=(10,8),layout='constrained')
    for row,(left,right) in zip([r for r in report['joins'] if r['duration_seconds']==.4],axes):
        match=row['representative_matches'][0]
        for ax,side,index in ((left,'a',row['take_a']-1),(right,'b',row['take_b']-1)):
            take=takes[index];start=match[f'start_frame_{side}'];v=take['logged'][start:start+20]
            normalized=(v-v.mean())/max(v.std(),1e-12)
            ax.imshow(normalized.T,origin='lower',aspect='auto',extent=[0,.4,0,8],vmin=-2,vmax=2,cmap='viridis')
            ax.set_title(f'Take {index+1}, {match[f"start_seconds_{side}"]:.2f}–{match[f"end_seconds_{side}"]:.2f}s',fontsize=10)
            ax.set_yticks(np.arange(8)+.5,['100–250','250–500','500–1k','1–2k','2–4k','4–8k','8–16k','16–24k'])
            ax.set_ylabel('Hz');ax.set_xlabel('Seconds in matched window')
    fig.suptitle('Strongest cross-take acoustic matches at 0.4 s\nNormalized band energy; recurrence does not establish swipe direction',fontsize=12)
    fig.savefig(OUT/'matches.png',dpi=140);plt.close(fig)
    print(json.dumps({'gain_max_error':max(g['max_normalized_feature_error'] for t in report['takes'] for g in t['gain_checks']),
                      'joins':[{k:v for k,v in r.items() if k not in ('representative_matches','source_ids')} for r in report['joins']],
                      'duration_stability':stability},indent=2))


if __name__=='__main__':main()
