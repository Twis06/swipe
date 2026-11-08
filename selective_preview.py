"""Conditional encoding fallback for raw unfamiliar-sound rejections only.

Accepted directions, background and ambiguous-direction results stay unchanged.
The second representation uses the frozen aligned reference bank and reuses the
original gate score, quality and event times. No additional parameters are fitted.
"""
from aligned_preview import direction_features
from fast_preview import batch_dtw, predict as raw_predict
from gesture_model import direction, decision


def classify_with_raw_result(samples, rate, raw_result, aligned_model):
    if raw_result['reason'] != 'unfamiliar_sound':
        return raw_result
    sequence = direction_features(samples, rate)
    distances = batch_dtw(sequence, aligned_model['templates'], int(aligned_model['radius']))
    predicted, margin, distance = direction(distances, aligned_model['labels'], int(aligned_model['neighbors']))
    quality = raw_result['quality']
    gate_score = raw_result['swipe_gate_score']
    result, reason = decision(predicted, margin, distance, float(aligned_model['distance_cutoff']), gate_score,
                              quality['silent'], float(aligned_model['margin_threshold']))
    if result not in ('up', 'down'):
        return raw_result
    return {'prediction': result, 'reason': reason, 'direction_candidate': predicted,
            'distance_margin': margin, 'nearest_distance': distance,
            'distance_cutoff': float(aligned_model['distance_cutoff']), 'swipe_gate_score': gate_score,
            'quality': quality, 'recognition_path': 'encoding_fallback'}


def predict(samples, rate, raw_model, aligned_model):
    raw_result = raw_predict(samples, rate, raw_model)
    return classify_with_raw_result(samples, rate, raw_result, aligned_model)
