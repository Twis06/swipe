"""Post-validation format diagnosis and fixed-method calibration, no parameter search."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from swipe.audio_format import decode_wav
from swipe.gesture_model import RATE, Configuration, canonical_audio, classify
from research.improve_model import Experiment
from research.validate_session import summarize

ROOT = Path(__file__).resolve().parents[1]


def main(directory):
    directory = Path(directory).resolve()
    session = json.loads((directory / "session.json").read_text())
    validation_path = ROOT / "diagnostics/validation" / session["id"] / "report.json"
    validation_blob = validation_path.read_bytes()
    output = ROOT / "diagnostics/calibration" / session["id"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("Calibration report already exists; preserved rather than overwritten.")
    frozen_path = ROOT / "diagnostics/improvement/noise_aware/pilot_model.npz"
    frozen_blob = frozen_path.read_bytes()
    with np.load(frozen_path, allow_pickle=False) as frozen:
        config = Configuration(str(frozen["features"]), int(frozen["radius"]), int(frozen["neighbors"]))
        plan = {
            "status": "Post-validation diagnosis and exploratory within-session calibration; not fresh independent validation",
            "source": session["id"], "configuration_fixed_from_old_model": config.name,
            "crop_mode": str(frozen["crop_mode"]), "parameter_search": False,
            "format_probe": "Resample new float audio to44100, then emulate legacy PCM16 clamp/quantization IN MEMORY only. Evaluate unchanged old model. This does not recreate all original capture conditions or repair audio.",
            "calibration": "Fit same fixed configuration on block1, evaluate block2; reverse blocks; six held-out block/pace groups. All templates, gate scalers and novelty thresholds use training clips only. Keep margin0.2 fixed.",
            "export": "Separate new-session-only and block1-only model banks; no change to frozen models or raw recordings.",
            "validation_sha256": hashlib.sha256(validation_blob).hexdigest(),
            "frozen_model_sha256": hashlib.sha256(frozen_blob).hexdigest(),
        }
        (output / "plan.json").write_text(json.dumps(plan, indent=2))
        probe_rows = []
        for clip in session["clips"]:
            if not clip["accepted"]:
                continue
            samples, rate, _ = decode_wav((directory / clip["filename"]).read_bytes())
            x = np.clip(canonical_audio(samples, rate), -1, 1)
            pcm = np.where(x < 0, x * 32768, x * 32767).astype(np.int16).astype(float) / 32768
            probe_rows.append({"file": clip["filename"], "truth": clip["label"], **classify(pcm, RATE, frozen)})
    experiment = Experiment("noise_aware", output, source=directory)
    report = {"status": plan["status"], "source": session["id"], "configuration": config.name,
              "format_probe": {"summary": summarize(probe_rows), "predictions": probe_rows},
              "block_evaluation": {}}
    for a, b in ((1, 2), (2, 1)):
        train, test = np.where(experiment.blocks == a)[0], np.where(experiment.blocks == b)[0]
        fitted = experiment.fit(train, config)
        rows = experiment.evaluate(fitted, test)
        report["block_evaluation"][f"{a}_to_{b}"] = {"summary": summarize(rows), "predictions": rows}
        if a == 1:
            experiment.save(fitted, "block1_model.npz", {"note": "No model selection; method fixed before calibration"})
    grouped = []
    folds = []
    for group in np.unique(experiment.groups):
        train, test = np.where(experiment.groups != group)[0], np.where(experiment.groups == group)[0]
        fitted = experiment.fit(train, config)
        rows = experiment.evaluate(fitted, test)
        grouped.extend(rows)
        folds.append({"group": str(group), "summary": summarize(rows)})
    report["group_evaluation"] = {"summary": summarize(grouped), "folds": folds, "predictions": grouped}
    fitted = experiment.fit(np.arange(len(experiment.y)), config)
    experiment.save(fitted, "pilot_model.npz", {"note": "No model selection; method fixed before calibration"})
    report["limitations"] = [
        "Same-session evaluation of calibration, not a second independent holdout",
        "New session was already inspected in frozen-model evaluation",
        "Format emulation is posthoc and approximate; session/gesture differences remain possible",
        "Only 20 swipe clips and8 negatives; no real continuous-use false-trigger rate",
        "Full-session exported model has no unused examples from this session for evaluation",
    ]
    if validation_path.read_bytes() != validation_blob or frozen_path.read_bytes() != frozen_blob:
        raise ValueError("Frozen validation/model changed during calibration.")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(output), "format_probe": report["format_probe"]["summary"],
                      "blocks": {k: v["summary"] for k, v in report["block_evaluation"].items()},
                      "grouped": report["group_evaluation"]["summary"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    main(parser.parse_args().session)
