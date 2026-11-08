"""Freeze the exploratory candidate using the existing pilot, without new audio."""
import hashlib
import json
import numpy as np
from research.diagnose_audio import SOURCE, OUT, event_data, BANDS
from research.analyze import read_wav

s=json.loads((SOURCE/'session.json').read_text())
clips=[c for c in s['clips'] if c['accepted'] and c['label'] in ('up','down') and c['block']==1]
sequences=[]
for c in clips:
    x,sr,_=read_wav(SOURCE/c['filename'])
    sequences.append(event_data(x,sr)[1])
np.savez(OUT/'dtw_block1_templates.npz',sequences=np.array(sequences),labels=np.array([c['label'] for c in clips]),filenames=np.array([c['filename'] for c in clips]),sha256=np.array([c['sha256'] for c in clips]),bands=BANDS)
config={'status':'Frozen exploratory candidate; not deployed or independently validated',
        'source_session':s['id'],'training_block':1,'training_clips':len(clips),
        'pipeline':'98%-energy crop +20ms padding; 256-sample STFT, hop64; 8 frequency bands; log relative power; interpolate32 frames; band8 DTW; nearest reference',
        'input':'Mono audio; no filename, prompt, reaction time, duration, or absolute level as explicit model features',
        'implementation_sha256':hashlib.sha256((OUT.parent/'research/diagnose_audio.py').read_bytes()).hexdigest(),
        'future_evaluation':'Keep configuration fixed. Test on a different session before claiming generalization.',
        'not_included':'Continuous event detection, confidence/unknown rejection, system actions'}
(OUT/'candidate_config.json').write_text(json.dumps(config,indent=2))
print('Saved block-1 DTW template bank and frozen candidate configuration.')
