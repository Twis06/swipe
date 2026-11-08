"""Small dependency-free reader for mono PCM16 and IEEE float32 RIFF WAV."""
import math
import struct

def decode_wav(blob, *, array=False):
    if len(blob)<12 or blob[:4]!=b'RIFF' or blob[8:12]!=b'WAVE':
        raise ValueError('Expected a RIFF WAV file.')
    if struct.unpack_from('<I',blob,4)[0]+8 != len(blob):
        raise ValueError('WAV size does not match its header.')
    position=12; fmt=None; pcm=None
    while position+8<=len(blob):
        tag=blob[position:position+4]; size=struct.unpack_from('<I',blob,position+4)[0]
        start=position+8; end=start+size
        if end>len(blob):raise ValueError('Incomplete WAV chunk.')
        if tag==b'fmt ':
            if fmt is not None or size<16:raise ValueError('Invalid WAV format chunk.')
            fmt=struct.unpack_from('<HHIIHH',blob,start)
        if tag==b'data':
            if pcm is not None:raise ValueError('Multiple WAV data chunks are unsupported.')
            pcm=blob[start:end]
        position=end+(size%2)
    if fmt is None or pcm is None:raise ValueError('Missing WAV format or samples.')
    encoding,channels,rate,byte_rate,align,bits=fmt
    if channels!=1 or (encoding,bits) not in {(1,16),(3,32)}:
        raise ValueError('Expected mono PCM16 or IEEE float32 WAV.')
    width=bits//8
    if align!=width or byte_rate!=rate*width or len(pcm)%width:
        raise ValueError('Inconsistent WAV sample layout.')
    if not 8000<=rate<=192000:raise ValueError('Unsupported sample rate.')
    if array:
        # Optional fast path for live inference. The collector's default path
        # remains dependency-free and shares every header/layout validation.
        import numpy as np
        samples=np.frombuffer(pcm,dtype='<i2' if encoding==1 else '<f4').astype(np.float64)
        if encoding==1:samples/=32768
        valid=bool(samples.size and np.isfinite(samples).all())
    else:
        if encoding==1:samples=[v[0]/32768 for v in struct.iter_unpack('<h',pcm)]
        else:samples=[v[0] for v in struct.iter_unpack('<f',pcm)]
        valid=bool(samples) and all(math.isfinite(x) for x in samples)
    if not valid:
        raise ValueError('Audio samples must be finite and nonempty.')
    return samples,rate,'pcm16' if encoding==1 else 'float32'

def recording_info(blob, duration_seconds=2):
    samples,rate,encoding=decode_wav(blob)
    frames=len(samples)
    if not duration_seconds-.2<=frames/rate<=duration_seconds+.2:
        raise ValueError(f'Recording must contain approximately {duration_seconds} seconds of audio.')
    high=sum(abs(x)>=.999 for x in samples)/frames
    # Flat, repeated near-rail samples are suspicious even when floats preserve
    # headroom. This is evidence of possible upstream limiting, not its cause.
    # Require at least three identical near-rail samples, rather than flagging
    # a single coincident pair at the turning point of an intact waveform.
    flat_samples=0; run=0; previous=None
    for sample in samples:
        run=(run+1 if sample==previous else 1) if abs(sample)>=.98 else 0
        if run==3:flat_samples+=3
        elif run>3:flat_samples+=1
        previous=sample
    flat=flat_samples/frames
    possible_clipping=flat>.001 or (encoding=='pcm16' and high>.001)
    return {'encoding':encoding,'sample_rate':rate,'frames':frames,'duration_seconds':frames/rate,
            'peak':max(abs(x) for x in samples),'rms':math.sqrt(sum(x*x for x in samples)/frames),
            'high_level_fraction':high,'flat_top_fraction':flat,
            'quality_warning':high>.001 or possible_clipping,
            'possible_clipping':possible_clipping,
            'level_assessment':'possible_clipping' if possible_clipping else 'high_peak' if high>.001 else 'normal',
            'quality_check_version':2}
