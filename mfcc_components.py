"""Fixed single-component MFCC ablations; no learned feature weights."""
import numpy as np
from scipy.signal import stft,savgol_filter,lfilter
from scipy.fft import dct
from gesture_model import RATE,canonical_audio
from swipe_feature_families import resize

METHODS={
 'mfcc_base':'Original24mel/12cepstra, temporal mean removal,2x gradient,global RMS.',
 'bands32':'Change only mel filter count24→32.',
 'cepstra8':'Retain8 instead of12 non-energy cepstral coefficients.',
 'log_bands':'Use24 geometric-frequency triangles instead of HTK mel spacing.',
 'variance_norm':'After temporal centering, divide each coefficient by max(its temporal std,0.1).',
 'retain_mean':'Omit temporal mean subtraction; keep static timbre.',
 'smooth_delta':'Replace central-difference gradient with5-frame order2 Savitzky–Golay derivative, still multiplied by2.',
 'preemphasis':'Apply fixed x[t]−0.97*x[t−1] before spectral analysis; crop remains based on original waveform.',
 'no_delta':'Omit derivative channels; keep12 centered cepstral coefficients.',
}

def bank(count,geometric=False):
 f=np.fft.rfftfreq(512,1/RATE)
 points=(np.geomspace(100,20000,count+2) if geometric else
         700*(10**(np.linspace(2595*np.log10(1+100/700),2595*np.log10(1+20000/700),count+2)/2595)-1))
 value=np.maximum(0,np.minimum((f[None,:]-points[:-2,None])/(points[1:-1]-points[:-2])[:,None],
     (points[2:,None]-f[None,:])/(points[2:]-points[1:-1])[:,None]))
 return value/np.maximum(value.sum(axis=1,keepdims=True),1e-30)

BANKS={(24,False):bank(24),(32,False):bank(32),(24,True):bank(24,True)}

def extract_variant(samples,rate,variant='mfcc_base'):
 if variant not in METHODS:raise ValueError('Unknown fixed component variant')
 x=canonical_audio(samples,rate);peak=float(np.max(abs(x)))
 dimensions=16 if variant=='cepstra8' else 12 if variant=='no_delta' else 24
 if peak<=1e-12:return np.zeros((32,dimensions))
 x=x/peak;power=x*x;width=round(RATE*.01)
 frames=power[:len(power)//width*width].reshape(-1,width).mean(axis=1)
 floor=np.median(frames);excess=np.maximum(frames-floor-6*np.median(abs(frames-floor)),0)
 if excess.sum()>1e-30:
  first,last=np.searchsorted(np.cumsum(excess)/excess.sum(),[.01,.99]);start,end=first*width,min(len(x)-1,(last+1)*width)
 else:start,end=np.searchsorted(np.cumsum(power)/power.sum(),[.01,.99])
 if variant=='preemphasis':x=lfilter([1,-.97],[1],x)
 pad=round(.02*RATE);event=x[max(0,start-pad):min(len(x),end+pad)]
 if len(event)<512:event=np.pad(event,(0,512-len(event)))
 _,_,z=stft(event,fs=RATE,nperseg=512,noverlap=384,boundary=None)
 _,_,full=stft(x,fs=RATE,nperseg=512,noverlap=384,boundary=None)
 spectrum=np.maximum(abs(z)**2-np.median(abs(full)**2,axis=1)[:,None]/np.log(2),0)
 mel=BANKS[(32 if variant=='bands32' else 24,variant=='log_bands')]@spectrum
 logged=np.log10(mel/max(float(mel.max()),1e-30)+1e-5)
 n=8 if variant=='cepstra8' else 12
 c=resize(dct(logged,type=2,axis=0,norm='ortho')[1:n+1].T)
 if variant!='retain_mean':c-=c.mean(axis=0,keepdims=True)
 if variant=='variance_norm':c/=np.maximum(c.std(axis=0,keepdims=True),.1)
 derivative=savgol_filter(c,5,2,deriv=1,axis=0,mode='interp') if variant=='smooth_delta' else np.gradient(c,axis=0)
 sequence=c if variant=='no_delta' else np.concatenate([c,2*derivative],axis=1)
 return sequence/max(float(np.sqrt(np.mean(sequence**2))),1e-12)

def extract_all(samples,rate):return {m:extract_variant(samples,rate,m) for m in METHODS}
