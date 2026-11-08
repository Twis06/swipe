"""Small offline swipe recognizer. No microphone access or system actions."""
from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly, stft
from scipy.spatial.distance import cdist

RATE = 44100
BANDS = np.array([100, 250, 500, 1000, 2000, 4000, 8000, 16000, 22051])
MARGIN_THRESHOLD = 0.2


@dataclass(frozen=True)
class Configuration:
    features: str = "spectrum"
    radius: int = 8
    neighbors: int = 1

    @property
    def name(self):
        return f"{self.features}_r{self.radius}_k{self.neighbors}"


# Baseline first: ties must not replace it with an unproven extra parameter.
BASELINE = Configuration()
CANDIDATES = [BASELINE] + [
    Configuration(feature, radius, neighbors)
    for feature in ("spectrum", "spectrum_delta", "centered_delta")
    for radius in (4, 8)
    for neighbors in (1, 3)
    if Configuration(feature, radius, neighbors) != BASELINE
]


def canonical_audio(samples, rate):
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite, nonempty mono audio.")
    if not 8000 <= rate <= 192000 or int(rate) != rate:
        raise ValueError("Unsupported sample rate.")
    if not 0.25 <= len(x) / rate <= 5:
        raise ValueError("Use a clip between 0.25 and 5 seconds.")
    if rate != RATE:
        common = gcd(int(rate), RATE)
        x = resample_poly(x, RATE // common, int(rate) // common)
    return x


def extract(samples, rate, crop_mode="energy"):
    """Only per-clip transforms; no labels or global fitted statistics."""
    x = canonical_audio(samples, rate)
    # Scale before squaring so all finite Float32 input amplitudes are safe.
    peak = float(np.max(np.abs(x)))
    normalized = x / max(peak, 1e-30)
    power = normalized ** 2
    energy = float(power.sum())
    rms_normalized = float(np.sqrt(np.mean(power)))
    rms = peak * rms_normalized
    cumulative = np.r_[0.0, np.cumsum(power)]
    width10 = int(RATE * .01)
    width200 = int(RATE * .2)
    peak10 = np.sqrt(np.max(cumulative[width10:] - cumulative[:-width10]) / width10) * peak
    concentration = np.max(cumulative[width200:] - cumulative[:-width200]) / max(energy, 1e-30)
    gate = np.array([
        np.log10(max(rms, 1e-12)),
        np.log10(max(float(peak10), 1e-12)),
        concentration,
        np.log10(max(1 / max(rms_normalized, 1e-12), 1)),
    ])
    if energy <= 1e-30 or peak <= 1e-12:
        return np.full((32, 8), -5.0), gate, {"silent": True, "peak": peak, "rms": rms}
    if crop_mode == "energy":
        start, end = np.searchsorted(np.cumsum(power) / energy, [.01, .99])
    elif crop_mode == "noise_aware":
        # A short gesture occupies a minority of the isolated clip. Estimate
        # stationary background from 10ms frame energies, without labels.
        frames = power[:len(power) // width10 * width10].reshape(-1, width10).mean(axis=1)
        floor = np.median(frames)
        mad = np.median(np.abs(frames - floor))
        excess = np.maximum(frames - (floor + 6 * mad), 0)
        if excess.sum() > 1e-30:
            first, last = np.searchsorted(np.cumsum(excess) / excess.sum(), [.01, .99])
            start, end = first * width10, min(len(x) - 1, (last + 1) * width10)
        else:
            start, end = np.searchsorted(np.cumsum(power) / energy, [.01, .99])
    else:
        raise ValueError(f"Unknown crop mode: {crop_mode}")
    padding = int(RATE * .02)
    event = normalized[max(0, start - padding):min(len(x), end + padding)]
    if len(event) < 256:
        event = np.pad(event, (0, 256 - len(event)))
    f, _, z = stft(event, fs=RATE, nperseg=256, noverlap=192, boundary=None)
    spectrum = np.abs(z) ** 2
    if crop_mode == "noise_aware":
        # Median periodogram / ln(2) estimates stationary Gaussian noise
        # power; it is a heuristic for other backgrounds. Never fit to labels.
        _, _, full_z = stft(normalized, fs=RATE, nperseg=256, noverlap=192, boundary=None)
        noise_power = np.median(np.abs(full_z) ** 2, axis=1) / np.log(2)
        spectrum = np.maximum(spectrum - noise_power[:, None], 0)
    bands = np.stack([
        spectrum[(f >= lo) & (f < hi)].sum(axis=0)
        for lo, hi in zip(BANDS[:-1], BANDS[1:])
    ], axis=1)
    log_bands = np.log10(bands / max(float(bands.max()), 1e-30) + 1e-5)
    sequence = np.array([
        np.interp(np.linspace(0, 1, 32), np.linspace(0, 1, len(log_bands)), log_bands[:, i])
        for i in range(8)
    ]).T
    quality = {
        "silent": False, "peak": peak, "rms": rms,
        "energy_duration_ms": float((end - start) / RATE * 1000),
        "near_rail_fraction": float(np.mean(np.abs(x) >= .999)),
    }
    return sequence, gate, quality


def transform(sequence, kind):
    if kind == "spectrum":
        return sequence
    derivative = 2 * np.gradient(sequence, axis=0)
    if kind == "spectrum_delta":
        return np.concatenate([sequence, derivative], axis=1)
    if kind == "centered_delta":
        return np.concatenate([sequence - sequence.mean(axis=0), derivative], axis=1)
    raise ValueError(f"Unknown feature type: {kind}")


def dtw(a, b, radius=8):
    distances = cdist(a, b, "sqeuclidean") / a.shape[1]
    cost = np.full((len(a) + 1, len(b) + 1), np.inf)
    cost[0, 0] = 0
    for i in range(1, len(a) + 1):
        for j in range(max(1, i - radius), min(len(b), i + radius) + 1):
            cost[i, j] = distances[i - 1, j - 1] + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[-1, -1] / (len(a) + len(b)))


def direction(distances, labels, neighbors):
    labels = np.asarray(labels)
    if set(labels) != {"up", "down"}:
        raise ValueError("Reference bank must contain both directions.")
    nearest = np.argsort(distances, kind="stable")[:neighbors]
    votes_up = int(np.sum(labels[nearest] == "up"))
    votes_down = int(np.sum(labels[nearest] == "down"))
    prediction = str(labels[nearest[0]]) if votes_up == votes_down else ("up" if votes_up > votes_down else "down")
    same = float(np.min(distances[labels == prediction]))
    other = float(np.min(distances[labels != prediction]))
    # Signed contrast, not a calibrated probability. A vote that disagrees
    # with the closest class has negative margin and will be rejected.
    margin = (other - same) / max(other + same, 1e-12)
    return prediction, float(margin), float(np.min(distances))


def decision(prediction, margin, distance, cutoff, gate_probability, silent=False, threshold=MARGIN_THRESHOLD):
    if silent or gate_probability < .5:
        return "background", "silence" if silent else "background_gate"
    if distance > cutoff:
        return "uncertain", "unfamiliar_sound"
    if margin < threshold:
        return "uncertain", "ambiguous_direction"
    return prediction, "accepted"


def classify(samples, rate, model):
    sequence, gate, quality = extract(samples, rate, str(model["crop_mode"]) if "crop_mode" in model else "energy")
    config = Configuration(str(model["features"]), int(model["radius"]), int(model["neighbors"]))
    query = transform(sequence, config.features)
    distances = np.array([dtw(query, reference, config.radius) for reference in model["templates"]])
    predicted, margin, distance = direction(distances, model["labels"], config.neighbors)
    scaled = (gate - model["gate_mean"]) / model["gate_scale"]
    logit = float(scaled @ model["gate_coef"] + model["gate_intercept"])
    probability = float(1 / (1 + np.exp(-np.clip(logit, -700, 700))))
    result, reason = decision(predicted, margin, distance, float(model["distance_cutoff"]), probability,
                              quality["silent"], float(model["margin_threshold"]))
    return {
        "prediction": result, "reason": reason, "direction_candidate": predicted,
        "distance_margin": margin, "nearest_distance": distance,
        "distance_cutoff": float(model["distance_cutoff"]),
        "swipe_gate_score": probability, "quality": quality,
        "score_note": "Distance margin is not a probability; gate score is not calibrated reliability.",
    }
