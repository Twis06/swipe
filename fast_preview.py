"""Exact batched DTW for the frozen preview model; no new fitted parameters.

All templates advance along the same DP anti-diagonal. Cells on a diagonal are
independent; dependencies remain on the preceding one or two diagonals. Local
costs use the same SciPy cdist and floating-point operations as gesture_model.
"""
import numpy as np
from scipy.spatial.distance import cdist

from gesture_model import direction, decision
from preview_model import features


def batch_dtw(query, templates, radius=4):
    query = np.asarray(query, dtype=np.float64)
    templates = np.asarray(templates, dtype=np.float64)
    if query.ndim != 2 or templates.ndim != 3:
        raise ValueError('Expected time-by-feature query and template-by-time-by-feature references.')
    n, dimensions = query.shape
    count, m, template_dimensions = templates.shape
    if not n or not m or not dimensions or dimensions != template_dimensions:
        raise ValueError('Sequences must be nonempty and share their feature dimension.')
    if int(radius) != radius or radius < 0:
        raise ValueError('Radius must be a nonnegative integer.')
    radius = int(radius)
    if not count:
        return np.empty(0)
    local = cdist(query, templates.reshape(count*m, dimensions), 'sqeuclidean')
    local = local.reshape(n, count, m).transpose(1, 0, 2)/dimensions
    cost = np.full((count, n+1, m+1), np.inf)
    cost[:, 0, 0] = 0
    for diagonal in range(2, n+m+1):
        i = np.arange(max(1, diagonal-m), min(n, diagonal-1)+1)
        j = diagonal-i
        allowed = np.abs(i-j) <= radius
        i, j = i[allowed], j[allowed]
        if not len(i):
            continue
        previous = np.minimum(np.minimum(cost[:, i-1, j], cost[:, i, j-1]), cost[:, i-1, j-1])
        cost[:, i, j] = local[:, i-1, j-1]+previous
    return cost[:, -1, -1]/(n+m)


def predict(samples, rate, model):
    # Keep the feature extractor, gate, tie ordering and rejection rules frozen.
    sequence, gate, quality = features(samples, rate, str(model['feature_input']))
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
