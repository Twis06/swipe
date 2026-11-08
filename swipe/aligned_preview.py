"""Conditional legacy-encoding direction view; existing raw gate stays frozen.

No raw audio is modified. This module is not the active service by itself.
Float32 live inputs always take the same transformation, with no peak-dependent
branch or best-margin selection. Stored PCM16 references preserve their values.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from swipe.fast_preview import batch_dtw
from swipe.gesture_model import RATE, canonical_audio, direction, decision
from swipe.preview_model import features
from swipe.paths import ROOT, MODEL_DIR



def align_signal(samples, rate, encoding='float32'):
    x = canonical_audio(samples, rate).copy()
    if encoding == 'pcm16':
        return x
    if encoding != 'float32':
        raise ValueError('Expected pcm16 or float32 source encoding.')
    x = np.clip(x, -1, 1)
    return np.where(x < 0, x*32768, x*32767).astype(np.int16).astype(np.float64)/32768


def direction_features(samples, rate, encoding='float32'):
    return features(align_signal(samples, rate, encoding), RATE, 'raw')[0]


def predict(samples, rate, model, encoding='float32'):
    # The raw gate, raw quality, and raw event timestamps preserve their original
    # definitions. Only direction comparison sees the common encoding view.
    _, gate, quality = features(samples, rate, 'raw')
    sequence = direction_features(samples, rate, encoding)
    distances = batch_dtw(sequence, model['templates'], int(model['radius']))
    predicted, margin, distance = direction(distances, model['labels'], int(model['neighbors']))
    standardized = (gate-model['gate_mean'])/model['gate_scale']
    logit = float(standardized @ model['gate_coef']+model['gate_intercept'])
    gate_score = float(1/(1+np.exp(-np.clip(logit, -700, 700))))
    result, reason = decision(predicted, margin, distance, float(model['distance_cutoff']), gate_score,
                              quality['silent'], float(model['margin_threshold']))
    return {'prediction': result, 'reason': reason, 'direction_candidate': predicted,
            'distance_margin': margin, 'nearest_distance': distance,
            'distance_cutoff': float(model['distance_cutoff']), 'swipe_gate_score': gate_score,
            'quality': quality}


def load_model(folder=None):
    folder = Path(folder) if folder is not None else MODEL_DIR/'aligned_preview'
    config = json.loads((folder/'model.json').read_text())
    blob = (folder/'model.npz').read_bytes()
    if hashlib.sha256(blob).hexdigest() != config['model_sha256']:
        raise ValueError('Aligned reference bank checksum changed.')
    for name, expected in config['implementation_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Aligned inference source changed: {name}')
    if hashlib.sha256((ROOT/config['source_model_path']).read_bytes()).hexdigest() != config['source_model_sha256']:
        raise ValueError('Original frozen gate source changed; review aligned-bank compatibility.')
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        model = {key: archive[key] for key in archive.files}
    return model, config
