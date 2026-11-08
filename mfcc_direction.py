"""Efficient fixed MFCC+delta candidate, with no live decision/rejection rule."""
import numpy as np
from scipy.signal import stft
from scipy.fft import dct
from gesture_model import RATE,canonical_audio
from swipe_feature_families import mel_bank,resize

MEL=mel_bank(np.fft.rfftfreq(512,1/RATE))

def extract_mfcc(samples,rate):
    x=canonical_audio(samples,rate)
    peak=float(np.max(abs(x)))
    if peak<=1e-12:return np.zeros((32,24))
    normalized=x/peak;power=normalized**2;width=round(RATE*.01)
    frames=power[:len(power)//width*width].reshape(-1,width).mean(axis=1)
    floor=np.median(frames);excess=np.maximum(frames-floor-6*np.median(abs(frames-floor)),0)
    if excess.sum()>1e-30:
        first,last=np.searchsorted(np.cumsum(excess)/excess.sum(),[.01,.99])
        start,end=first*width,min(len(x)-1,(last+1)*width)
    else:start,end=np.searchsorted(np.cumsum(power)/power.sum(),[.01,.99])
    pad=round(.02*RATE);event=normalized[max(0,start-pad):min(len(x),end+pad)]
    if len(event)<512:event=np.pad(event,(0,512-len(event)))
    _,_,z=stft(event,fs=RATE,nperseg=512,noverlap=384,boundary=None)
    _,_,full=stft(normalized,fs=RATE,nperseg=512,noverlap=384,boundary=None)
    spectrum=np.maximum(abs(z)**2-np.median(abs(full)**2,axis=1)[:,None]/np.log(2),0)
    mel=MEL@spectrum
    logged=np.log10(mel/max(float(mel.max()),1e-30)+1e-5)
    cepstra=resize(dct(logged,type=2,axis=0,norm='ortho')[1:13].T)
    cepstra-=cepstra.mean(axis=0,keepdims=True)
    sequence=np.concatenate([cepstra,2*np.gradient(cepstra,axis=0)],axis=1)
    return sequence/max(float(np.sqrt(np.mean(sequence**2))),1e-12)
