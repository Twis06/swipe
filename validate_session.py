"""Evaluate the previously frozen models on a new completed session. No fitting."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from audio_format import decode_wav, recording_info
from gesture_model import classify

ROOT = Path(__file__).resolve().parent


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def summarize(rows):
    swipes = [r for r in rows if r["truth"] in ("up", "down")]
    negative = [r for r in rows if r["truth"] not in ("up", "down")]
    accepted = [r for r in swipes if r["prediction"] in ("up", "down")]
    confusion = {label: dict.fromkeys(("up", "down", "background", "uncertain"), 0)
                 for label in ("up", "down", "idle", "typing")}
    for r in rows:
        confusion[r["truth"]][r["prediction"]] += 1
    return {
        "forced_direction_correct": sum(r["direction_candidate"] == r["truth"] for r in swipes),
        "swipes": len(swipes), "accepted_swipes": len(accepted),
        "accepted_correct": sum(r["prediction"] == r["truth"] for r in accepted),
        "coverage": len(accepted) / len(swipes) if swipes else None,
        "accepted_accuracy": sum(r["prediction"] == r["truth"] for r in accepted) / len(accepted) if accepted else None,
        "background_false_activations": sum(r["prediction"] in ("up", "down") for r in negative),
        "background_clips": len(negative),
        "swipe_reasons": dict(Counter(r["reason"] for r in swipes)),
        "confusion": confusion,
        "by_direction": {
            label: {"total": sum(r["truth"] == label for r in swipes),
                    "forced_correct": sum(r["truth"] == label and r["direction_candidate"] == label for r in swipes),
                    "accepted": sum(r["truth"] == label for r in accepted),
                    "accepted_correct": sum(r["truth"] == label and r["prediction"] == label for r in accepted)}
            for label in ("up", "down")
        },
    }


def validate(directory):
    directory = Path(directory).resolve()
    session_blob = (directory / "session.json").read_bytes()
    session = json.loads(session_blob)
    clips = [c for c in session["clips"] if c["accepted"]]
    if sorted(c["trial_index"] for c in clips) != list(range(len(session["plan"]))):
        raise ValueError("Session is incomplete or has duplicate accepted trial indices.")
    primary_path = ROOT / "diagnostics/improvement/noise_aware/pilot_model.npz"
    comparison_path = ROOT / "diagnostics/improvement/pilot_model.npz"
    config = json.loads(primary_path.with_suffix(".json").read_text())
    if sha((ROOT / "gesture_model.py").read_bytes()) != config["implementation_sha256"]["gesture_model.py"]:
        raise ValueError("Frozen primary inference implementation has changed; review before evaluation.")
    output = ROOT / "diagnostics/validation" / session["id"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("A validation report already exists; it is preserved rather than overwritten.")
    paths = {"primary_noise_aware": primary_path, "comparison_original_crop": comparison_path}
    manifest = {
        "session": session["id"], "session_json_sha256": sha(session_blob),
        "primary": "primary_noise_aware", "implementation_sha256": sha((ROOT / "gesture_model.py").read_bytes()),
        "models": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path.read_bytes())}
                   for name, path in paths.items()},
        "protocol": "Evaluate every accepted clip. Do not tune, fit, select thresholds, or exclude high peaks. Secondary model is a comparison, not a replacement for the primary outcome.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    data, quality = [], []
    for c in clips:
        blob = (directory / c["filename"]).read_bytes()
        if sha(blob) != c["sha256"]:
            raise ValueError(f'Clip changed: {c["filename"]}')
        trial = session["plan"][c["trial_index"]]
        if any(c[key] != trial[key] for key in ("label", "block", "pace")):
            raise ValueError("Saved label disagrees with plan.")
        samples, rate, _ = decode_wav(blob)
        data.append((c, np.asarray(samples), rate))
        quality.append({"file": c["filename"], "label": c["label"], "sha256": sha(blob), **recording_info(blob)})
    report = {"session": session["id"], "status": "New-session evaluation of models frozen before collection; no fitting or tuning",
              "clips": len(clips), "labels": dict(Counter(c["label"] for c in clips)),
              "quality": quality, "models": {},
              "limitations": ["One user, one new session", "Protocol changed from clipped PCM16/44.1kHz to float/48kHz",
                              "No flat tops is not proof of distortion-free upstream audio", "Only idle/typing background examples"]}
    for name, path in paths.items():
        rows = []
        with np.load(path, allow_pickle=False) as model:
            training_hashes = set(model["training_sha256"])
            for c, samples, rate in data:
                if c["sha256"] in training_hashes:
                    raise ValueError("Training/evaluation file overlap.")
                rows.append({"file": c["filename"], "truth": c["label"], "block": c["block"], "pace": c["pace"],
                             **classify(samples, rate, model)})
        report["models"][name] = {"summary": summarize(rows), "predictions": rows}
        if sha(path.read_bytes()) != manifest["models"][name]["sha256"]:
            raise ValueError("Model changed during evaluation.")
    if (directory / "session.json").read_bytes() != session_blob:
        raise ValueError("Session changed during evaluation.")
    report["quality_summary"] = {
        "encoding": dict(Counter(c["encoding"] for c in quality)),
        "sample_rates": dict(Counter(c["sample_rate"] for c in quality)),
        "possible_clipping_clips": sum(c["possible_clipping"] for c in quality),
        "high_peak_clips": sum(c["level_assessment"] == "high_peak" for c in quality),
        "swipe_peak_range": [min(c["peak"] for c in quality if c["label"] in ("up", "down")),
                             max(c["peak"] for c in quality if c["label"] in ("up", "down"))],
        "all_hashes_verified": True,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(output), "quality": report["quality_summary"],
                      "models": {name: value["summary"] for name, value in report["models"].items()}}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    validate(parser.parse_args().session)
