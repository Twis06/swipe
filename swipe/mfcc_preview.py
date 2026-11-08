"""Separate calibrated MFCC8 candidate; not activated by importing this file."""
import hashlib,io,json
from pathlib import Path
import numpy as np
from swipe.gesture_model import RATE,canonical_audio,direction,decision
from swipe.mfcc_components import extract_variant
from swipe.fast_preview import batch_dtw
from swipe.paths import ROOT, MODEL_DIR


def gate_quality(x):
    peak=float(np.max(np.abs(x)));normalized=x/max(peak,1e-30);power=normalized**2
    energy=float(power.sum());rms_normalized=float(np.sqrt(np.mean(power)));rms=peak*rms_normalized
    cumulative=np.r_[0.,np.cumsum(power)];width=int(RATE*.01);width200=int(RATE*.2)
    peak10=np.sqrt(np.max(cumulative[width:]-cumulative[:-width])/width)*peak
    concentration=np.max(cumulative[width200:]-cumulative[:-width200])/max(energy,1e-30)
    gate=np.array([np.log10(max(rms,1e-12)),np.log10(max(float(peak10),1e-12)),concentration,np.log10(max(1/max(rms_normalized,1e-12),1))])
    if energy<=1e-30 or peak<=1e-12:return gate,{'silent':True,'peak':peak,'rms':rms}
    frames=power[:len(power)//width*width].reshape(-1,width).mean(axis=1)
    floor=np.median(frames);excess=np.maximum(frames-floor-6*np.median(abs(frames-floor)),0)
    if excess.sum()>1e-30:
        first,last=np.searchsorted(np.cumsum(excess)/excess.sum(),[.01,.99]);start,end=first*width,min(len(x)-1,(last+1)*width)
    else:start,end=np.searchsorted(np.cumsum(power)/power.sum(),[.01,.99])
    return gate,{'silent':False,'peak':peak,'rms':rms,'energy_duration_ms':float((end-start)/RATE*1000),
                 'near_rail_fraction':float(np.mean(abs(x)>=.999)),
                 'event_start_seconds':float(start/RATE),'event_end_seconds':float(end/RATE)}

def predict(samples,rate,model):
    x=canonical_audio(samples,rate);gate,quality=gate_quality(x)
    sequence=extract_variant(x,RATE,'cepstra8')
    p,m,d=direction(batch_dtw(sequence,model['templates'],int(model['radius'])),model['labels'],int(model['neighbors']))
    z=(gate-model['gate_mean'])/model['gate_scale']
    score=float(1/(1+np.exp(-np.clip(float(z@model['gate_coef']+model['gate_intercept']),-700,700))))
    result,reason=decision(p,m,d,float(model['distance_cutoff']),score,quality['silent'],float(model['margin_threshold']))
    return {'prediction':result,'reason':reason,'direction_candidate':p,'distance_margin':m,'nearest_distance':d,
            'distance_cutoff':float(model['distance_cutoff']),'swipe_gate_score':score,'quality':quality}

def load_model():
    folder=MODEL_DIR/'mfcc_preview';config=json.loads((folder/'model.json').read_text());blob=(folder/'model.npz').read_bytes()
    if hashlib.sha256(blob).hexdigest()!=config['model_sha256']:raise ValueError('MFCC bank checksum changed')
    for name,digest in config['implementation_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'MFCC source checksum changed: {name}')
    with np.load(io.BytesIO(blob),allow_pickle=False) as bank:model={k:bank[k] for k in bank.files}
    return model,config
