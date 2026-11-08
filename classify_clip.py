"""Offline direction demo for a known swipe; does not detect background/unknowns."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audio_format import decode_wav
from diagnose_audio import OUT, event_data, dtw

parser=argparse.ArgumentParser()
parser.add_argument('wav',type=Path)
args=parser.parse_args()
model=np.load(OUT/'dtw_block1_templates.npz',allow_pickle=False)
blob=args.wav.read_bytes();sha=hashlib.sha256(blob).hexdigest()
if sha in model['sha256']:
    raise SystemExit('This file is in the reference bank. Refusing a misleading self-match; choose a block-2 swipe.')
samples,rate,_=decode_wav(blob)
sequence=event_data(np.array(samples),rate)[1]
distances=np.array([dtw(sequence,reference) for reference in model['sequences']])
best=int(np.argmin(distances))
print(json.dumps({'prediction':str(model['labels'][best]),'nearest_reference':str(model['filenames'][best]),
                  'distance':float(distances[best]),'scope':'Known-swipe offline demo only. No background rejection; pilot scores are exploratory.'},indent=2))
