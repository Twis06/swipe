"""Fixed, per-recording feature extraction. No learned parameters or labels."""
import numpy as np
from scipy.fft import dct
from scipy.ndimage import gaussian_filter1d
from scipy.signal import stft, lfilter
from gesture_model import RATE, BANDS, canonical_audio
from preview_model import features

METHODS = {
    'raw': 'Existing32x8 noise-aware log-band spectrum, unchanged.',
    'envelope': '32-frame normalized log energy, signed first derivative x2, cumulative energy fraction. Three channels.',
    'spectral_motion': 'Energy, centroid, bandwidth,85% rolloff, flatness, positive spectral flux, centroid velocity, and their temporal gradients. Fourteen channels.',
    'mfcc_delta': '24 area-normalized HTK-mel triangles100Hz–20kHz; orthonormal DCT coefficients1–12; per-clip temporal centering; append2x gradient; global per-clip RMS normalization.24 channels.',
    'pcen': '24 mel power channels; causal60ms smoothing on full clip; gain.98, bias2, power.5, epsilon1e-6; crop then per-clip max normalization.24 channels. PCEN-style, not a library-default implementation.',
    'multiscale_envelope': 'Eight noise-subtracted band energies smoothed with Gaussian5ms and20ms widths before log compression.16 channels; not a full scattering transform.',
    'onset_offset': 'Positive and negative temporal gradients of existing32x8 log bands, kept separately.16 channels.',
}


def resize(sequence):
    return np.stack([np.interp(np.linspace(0,1,32),np.linspace(0,1,len(sequence)),sequence[:,i])
                     for i in range(sequence.shape[1])],axis=1)


def mel_bank(f):
    points=700*(10**(np.linspace(2595*np.log10(1+100/700),2595*np.log10(1+20000/700),26)/2595)-1)
    bank=np.maximum(0,np.minimum((f[None,:]-points[:-2,None])/(points[1:-1]-points[:-2])[:,None],
                                (points[2:,None]-f[None,:])/(points[2:]-points[1:-1])[:,None]))
    return bank/np.maximum(bank.sum(axis=1,keepdims=True),1e-30)


def extract_all(samples,rate):
    x=canonical_audio(samples,rate)
    baseline,_,q=features(x,RATE,'raw')
    if q['silent']:
        result={name:np.zeros((32,d)) for name,d in zip(METHODS,(8,3,14,24,24,16,16))}
        result['raw']=baseline
        return result
    x=x/max(float(np.max(abs(x))),1e-30)
    left=max(0,round((q['event_start_seconds']-.02)*RATE))
    right=min(len(x),round((q['event_end_seconds']+.02)*RATE))
    event=x[left:right]
    if len(event)<512:event=np.pad(event,(0,512-len(event)))
    f,_,z=stft(event,fs=RATE,nperseg=512,noverlap=384,boundary=None)
    _,times,full=stft(x,fs=RATE,nperseg=512,noverlap=384,boundary=None)
    spectrum=np.maximum(abs(z)**2-np.median(abs(full)**2,axis=1)[:,None]/np.log(2),0)
    total=spectrum.sum(axis=0)
    safe=np.maximum(total,1e-30)
    energy=resize((np.log10(total/max(float(total.max()),1e-30)+1e-5)/5)[:,None])[:,0]
    linear=resize((total/max(float(total.max()),1e-30))[:,None])[:,0]
    envelope=np.stack([energy,2*np.gradient(energy),np.cumsum(linear)/max(float(linear.sum()),1e-30)],axis=1)
    centroid=(f[:,None]*spectrum).sum(axis=0)/safe
    bandwidth=np.sqrt((((f[:,None]-centroid)**2)*spectrum).sum(axis=0)/safe)
    rolloff=f[np.argmax(np.cumsum(spectrum,axis=0)>=.85*total,axis=0)]
    flat=np.exp(np.mean(np.log(spectrum+1e-30),axis=0))/np.maximum(spectrum.mean(axis=0),1e-30)
    distribution=spectrum/safe
    flux=np.maximum(np.diff(distribution,axis=1,prepend=distribution[:,:1]),0).sum(axis=0)
    motion=resize(np.stack([np.log10(total/max(float(total.max()),1e-30)+1e-5)/5,
        centroid/(RATE/2),bandwidth/(RATE/2),rolloff/(RATE/2),flat,flux,np.gradient(centroid/(RATE/2))],axis=1))
    motion=np.concatenate([motion,2*np.gradient(motion,axis=0)],axis=1)
    bank=mel_bank(f);mel=bank@spectrum
    logged=np.log10(mel/max(float(mel.max()),1e-30)+1e-5)
    cepstra=resize(dct(logged,type=2,axis=0,norm='ortho')[1:13].T)
    cepstra-=cepstra.mean(axis=0,keepdims=True)
    cepstra=np.concatenate([cepstra,2*np.gradient(cepstra,axis=0)],axis=1)
    cepstra/=max(float(np.sqrt(np.mean(cepstra**2))),1e-12)
    fullmel=bank@(abs(full)**2)
    b=1-np.exp(-128/(RATE*.06))
    smooth,_=lfilter([b],[1,-(1-b)],fullmel,axis=1,zi=((1-b)*fullmel[:,:1]))
    pcen=(fullmel/(1e-6+smooth)**.98+2)**.5-np.sqrt(2)
    chosen=(times>=left/RATE)&(times<=right/RATE)
    # The fixed crop always includes at least one512-point frame centre.
    if not chosen.any():chosen[np.argmin(abs(times-(left+right)/(2*RATE)))]=True
    pcen=resize(pcen[:,chosen].T)
    pcen/=max(float(pcen.max()),1e-12)
    bands=np.stack([spectrum[(f>=a)&(f<b)].sum(axis=0) for a,b in zip(BANDS[:-1],BANDS[1:])],axis=1)
    scales=[]
    for seconds in (.005,.020):
        smoothed=gaussian_filter1d(bands,seconds*RATE/128,axis=0,mode='nearest')
        scales.append(resize(np.log10(smoothed/max(float(smoothed.max()),1e-30)+1e-5)))
    grad=np.gradient(baseline,axis=0)
    result={'raw':baseline,'envelope':envelope,'spectral_motion':motion,'mfcc_delta':cepstra,
            'pcen':pcen,'multiscale_envelope':np.concatenate(scales,axis=1),
            'onset_offset':np.concatenate([np.maximum(grad,0),np.maximum(-grad,0)],axis=1)}
    if any(not np.isfinite(a).all() for a in result.values()):raise ValueError('Nonfinite feature')
    return result
