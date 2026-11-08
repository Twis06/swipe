"""Consistent feature extraction for preview training and live inference.

Frozen gesture_model.py and prior model files are intentionally untouched.
"""
import numpy as np
from scipy.signal import stft

from gesture_model import RATE, BANDS, canonical_audio, extract, dtw, direction, decision


def features(samples, rate, kind):
    x = canonical_audio(samples, rate)
    baseline, gate, quality = extract(x, RATE, "noise_aware")
    if quality["silent"]:
        return baseline, gate, quality
    normalized = x / max(float(np.max(np.abs(x))), 1e-30)
    power = normalized ** 2
    width = int(RATE * .01)
    frames = power[:len(power) // width * width].reshape(-1, width).mean(axis=1)
    floor = np.median(frames)
    excess = np.maximum(frames - floor - 6 * np.median(np.abs(frames - floor)), 0)
    if excess.sum() > 1e-30:
        first, last = np.searchsorted(np.cumsum(excess) / excess.sum(), [.01, .99])
        start, end = first * width, min(len(x) - 1, (last + 1) * width)
    else:
        start, end = np.searchsorted(np.cumsum(power) / power.sum(), [.01, .99])
    quality={**quality,"event_start_seconds":float(start/RATE),"event_end_seconds":float(end/RATE)}
    if kind == "raw":
        return baseline, gate, quality
    padding = int(RATE * .02)
    left, right = max(0, start - padding), min(len(x), end + padding)
    if kind == "soft_polarity":
        scale = max(float(np.sqrt(np.mean(power[start:max(start + 1, end)]))) * .05, 1e-12)
        transformed = normalized / (np.abs(normalized) + scale)
    elif kind == "polarity":
        transformed = np.sign(normalized)
    else:
        raise ValueError("Unknown preview feature input.")
    event = transformed[left:right]
    if len(event) < 256:
        event = np.pad(event, (0, 256 - len(event)))
    f, _, z = stft(event, fs=RATE, nperseg=256, noverlap=192, boundary=None)
    _, _, background = stft(transformed, fs=RATE, nperseg=256, noverlap=192, boundary=None)
    noise = np.median(np.abs(background) ** 2, axis=1) / np.log(2)
    spectrum = np.maximum(np.abs(z) ** 2 - noise[:, None], 0)
    bands = np.stack([spectrum[(f >= a) & (f < b)].sum(axis=0) for a, b in zip(BANDS[:-1], BANDS[1:])], axis=1)
    logged = np.log10(bands / max(float(bands.max()), 1e-30) + 1e-5)
    sequence = np.array([np.interp(np.linspace(0, 1, 32), np.linspace(0, 1, len(logged)), logged[:, i]) for i in range(8)]).T
    return sequence, gate, quality


def predict(samples, rate, model):
    sequence, gate, quality = features(samples, rate, str(model["feature_input"]))
    distances = np.array([dtw(sequence, template, int(model["radius"])) for template in model["templates"]])
    predicted, margin, distance = direction(distances, model["labels"], int(model["neighbors"]))
    standardized = (gate - model["gate_mean"]) / model["gate_scale"]
    logit = float(standardized @ model["gate_coef"] + model["gate_intercept"])
    gate_score = float(1 / (1 + np.exp(-np.clip(logit, -700, 700))))
    result, reason = decision(predicted, margin, distance, float(model["distance_cutoff"]), gate_score,
                              quality["silent"], float(model["margin_threshold"]))
    return {"prediction": result, "reason": reason, "direction_candidate": predicted,
            "distance_margin": margin, "nearest_distance": distance, "distance_cutoff": float(model["distance_cutoff"]),
            "swipe_gate_score": gate_score, "quality": quality}
