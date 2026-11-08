"""Run the bounded, nested pilot experiment in diagnostics/improvement_plan.json."""
import hashlib
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from swipe.audio_format import decode_wav
from swipe.gesture_model import (BASELINE, CANDIDATES, RATE, MARGIN_THRESHOLD,
                           canonical_audio, extract, transform, dtw, direction, decision)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/20261103T165139_cf7201"
OUT = ROOT / "diagnostics/improvement"


def metrics(rows):
    swipes = [r for r in rows if r["truth"] in ("up", "down")]
    background = [r for r in rows if r["truth"] not in ("up", "down")]
    accepted = [r for r in swipes if r["prediction"] in ("up", "down")]
    confusion = {label: {p: 0 for p in ("up", "down", "background", "uncertain")}
                 for label in ("up", "down", "background")}
    for row in rows:
        truth = row["truth"] if row["truth"] in ("up", "down") else "background"
        confusion[truth][row["prediction"]] += 1
    return {
        "forced_direction_correct": sum(r["direction_candidate"] == r["truth"] for r in swipes),
        "swipes": len(swipes),
        "accepted_correct": sum(r["prediction"] == r["truth"] for r in accepted),
        "accepted_swipes": len(accepted),
        "swipe_coverage": len(accepted) / len(swipes) if swipes else None,
        "accepted_accuracy": sum(r["prediction"] == r["truth"] for r in accepted) / len(accepted) if accepted else None,
        "background_false_activations": sum(r["prediction"] in ("up", "down") for r in background),
        "background_clips": len(background),
        "background_gate_passes": sum(r["swipe_gate_score"] >= .5 and not r["silent"] for r in background),
        "swipes_rejected_by_gate": sum(r["reason"] in ("background_gate", "silence") for r in swipes),
        "confusion": confusion,
    }


class Experiment:
    def __init__(self, crop_mode="energy", output=OUT, source=SOURCE):
        self.crop_mode, self.output = crop_mode, output
        self.session = json.loads((source / "session.json").read_text())
        self.clips = [c for c in self.session["clips"] if c["accepted"]]
        self.y = np.array([c["label"] for c in self.clips])
        self.swipe = np.isin(self.y, ["up", "down"])
        self.groups = np.array([f'{c["block"]}_{c["pace"]}' for c in self.clips])
        self.blocks = np.array([c["block"] for c in self.clips])
        self.signals, self.sequences, self.gates, self.quality = [], [], [], []
        for c in self.clips:
            blob = (source / c["filename"]).read_bytes()
            if hashlib.sha256(blob).hexdigest() != c["sha256"]:
                raise ValueError(f'Original hash mismatch: {c["filename"]}')
            samples, rate, _ = decode_wav(blob)
            x = canonical_audio(samples, rate)
            sequence, gate, quality = extract(x, RATE, self.crop_mode)
            self.signals.append(x)
            self.sequences.append(sequence)
            self.gates.append(gate)
            self.quality.append(quality)
        self.sequences, self.gates = np.array(self.sequences), np.array(self.gates)
        self.distance_cache = {}

    def distances(self, config):
        key = (config.features, config.radius)
        if key not in self.distance_cache:
            sequences = [transform(s, config.features) for s in self.sequences]
            distances = np.zeros((len(sequences), len(sequences)))
            for i in range(len(sequences)):
                for j in range(i):
                    distances[i, j] = distances[j, i] = dtw(sequences[i], sequences[j], config.radius)
            self.distance_cache[key] = distances
        return self.distance_cache[key]

    def select(self, train):
        train = train[self.swipe[train]]
        scores = {}
        for config in CANDIDATES:
            distance = self.distances(config)
            correct = 0
            for group in np.unique(self.groups[train]):
                fit = train[self.groups[train] != group]
                test = train[self.groups[train] == group]
                for index in test:
                    predicted, _, _ = direction(distance[index, fit], self.y[fit], config.neighbors)
                    correct += predicted == self.y[index]
            scores[config.name] = int(correct)
        winner = max(CANDIDATES, key=lambda c: scores[c.name])
        return winner, scores

    def fit(self, train, config):
        refs = train[self.swipe[train]]
        calibration_distances = []
        distance = self.distances(config)
        for index in refs:
            # Entire source group excluded, not merely the exact query file.
            same = refs[(self.groups[refs] != self.groups[index]) & (self.y[refs] == self.y[index])]
            if not len(same):
                raise ValueError("Not enough training groups to calibrate novelty rejection.")
            calibration_distances.append(float(np.min(distance[index, same])))
        cutoff = float(np.quantile(calibration_distances, .95))
        gate = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", random_state=0))
        gate.fit(self.gates[train], self.swipe[train].astype(int))
        return {"config": config, "refs": refs, "train": train, "cutoff": cutoff, "gate": gate}

    def evaluate(self, fitted, test, sequences=None, gates=None, quality=None):
        config, refs = fitted["config"], fitted["refs"]
        clean = sequences is None
        sequences = self.sequences if sequences is None else sequences
        gates = self.gates if gates is None else gates
        quality = self.quality if quality is None else quality
        probabilities = fitted["gate"].predict_proba(gates[test])[:, 1]
        rows = []
        for index, probability in zip(test, probabilities):
            if clean:
                distances = self.distances(config)[index, refs]
            else:
                query = transform(sequences[index], config.features)
                distances = np.array([dtw(query, transform(self.sequences[r], config.features), config.radius) for r in refs])
            predicted, margin, distance = direction(distances, self.y[refs], config.neighbors)
            result, reason = decision(predicted, margin, distance, fitted["cutoff"], probability, quality[index]["silent"])
            rows.append({
                "file": self.clips[index]["filename"], "truth": str(self.y[index]),
                "group": str(self.groups[index]), "selected_model": config.name,
                "prediction": result, "reason": reason, "direction_candidate": predicted,
                "margin": margin, "nearest_distance": distance, "distance_cutoff": fitted["cutoff"],
                "swipe_gate_score": float(probability), "silent": quality[index]["silent"],
            })
        return rows

    def save(self, fitted, filename, scores):
        config, refs = fitted["config"], fitted["refs"]
        scaler = fitted["gate"].named_steps["standardscaler"]
        logistic = fitted["gate"].named_steps["logisticregression"]
        np.savez_compressed(
            self.output / filename,
            crop_mode=self.crop_mode,
            features=config.features, radius=config.radius, neighbors=config.neighbors,
            templates=np.array([transform(self.sequences[i], config.features) for i in refs]),
            labels=self.y[refs], filenames=np.array([self.clips[i]["filename"] for i in refs]),
            training_sha256=np.array([self.clips[i]["sha256"] for i in fitted["train"]]),
            gate_mean=scaler.mean_, gate_scale=scaler.scale_, gate_coef=logistic.coef_[0],
            gate_intercept=float(logistic.intercept_[0]), distance_cutoff=fitted["cutoff"],
            margin_threshold=MARGIN_THRESHOLD,
        )
        config_json = {
            "status": "Exploratory offline model; not independently validated or connected to system actions",
            "selected_model": config.name, "crop_mode": self.crop_mode, "training_clips": len(fitted["train"]),
            "reference_swipes": len(refs), "training_groups": sorted(set(self.groups[fitted["train"]])),
            "training_only_selection_scores": scores, "margin_threshold": MARGIN_THRESHOLD,
            "distance_cutoff": fitted["cutoff"], "canonical_sample_rate": RATE,
            "source_session": self.session["id"],
            "training_files": [self.clips[i]["filename"] for i in fitted["train"]],
            "implementation_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                      for name in ('swipe/gesture_model.py', 'research/improve_model.py')},
            "limitations": ["Recording quality requires separate review", "One session only", "Only idle/typing negatives",
                            "Gate score and distance margin are not calibrated probabilities",
                            "Offline isolated clips only; no continuous detector", "No independent session validation after this fit"],
        }
        (self.output / filename.replace(".npz", ".json")).write_text(json.dumps(config_json, indent=2))


def margin_curve(rows):
    curve = []
    for threshold in (0, .1, .2, .3, .4):
        changed = []
        for row in rows:
            result, reason = decision(row["direction_candidate"], row["margin"], row["nearest_distance"],
                                      row["distance_cutoff"], row["swipe_gate_score"], row["silent"], threshold)
            changed.append({**row, "prediction": result, "reason": reason})
        curve.append({"margin_threshold": threshold, **metrics(changed)})
    return curve


def perturb(experiment, kind):
    sequences, gates, quality = [], [], []
    rng = np.random.default_rng(20261103)
    for x in experiment.signals:
        if kind == "amplitude_0.25":
            changed = x * .25
        elif kind == "shift_200ms":
            shift = int(RATE * .2)
            changed = np.r_[np.zeros(shift), x[:-shift]]
        elif kind.startswith("speed_"):
            # Changing sample count shifts pitch too. Diagnostic only.
            num, den = (20, 17) if kind == "speed_0.85" else (20, 23)
            changed = resample_poly(x, num, den)
            changed = np.pad(changed, (0, max(0, len(x) - len(changed))))[:len(x)]
        elif kind == "noise_20db_event":
            energy = np.cumsum(x * x)
            if energy[-1] > 0:
                start, end = np.searchsorted(energy / energy[-1], [.01, .99])
                event_rms = np.sqrt(np.mean(x[start:max(start + 1, end)] ** 2))
            else:
                event_rms = 0
            changed = x + rng.normal(0, event_rms / 10, size=len(x))
        else:
            raise ValueError(kind)
        sequence, gate, info = extract(changed, RATE, experiment.crop_mode)
        sequences.append(sequence); gates.append(gate); quality.append(info)
    return np.array(sequences), np.array(gates), quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=("energy", "noise_aware"), default="energy")
    args = parser.parse_args()
    output = OUT if args.crop == "energy" else OUT / "noise_aware"
    output.mkdir(parents=True, exist_ok=True)
    experiment = Experiment(args.crop, output)
    all_indices = np.arange(len(experiment.y))
    report = {
        "status": "Exploratory reuse of one already-inspected pilot; no independent accuracy claim",
        "plan": "diagnostics/improvement_plan.json", "source_hashes_verified": len(experiment.clips),
        "crop_mode": args.crop,
        "candidates": {}, "nested_group_evaluation": {}, "block_evaluation": {},
    }
    folds = [(g, np.where(experiment.groups != g)[0], np.where(experiment.groups == g)[0])
             for g in np.unique(experiment.groups)]
    # Fixed-model forced accuracy: descriptive only, never used as an outer estimate of selection.
    for config in CANDIDATES:
        correct = 0
        for _, train, test in folds:
            refs = train[experiment.swipe[train]]
            for index in test[experiment.swipe[test]]:
                predicted, _, _ = direction(experiment.distances(config)[index, refs], experiment.y[refs], config.neighbors)
                correct += predicted == experiment.y[index]
        report["candidates"][config.name] = {"correct": int(correct), "total": int(experiment.swipe.sum())}
    fitted_folds = []
    all_rows, baseline_rows, selections = [], [], []
    for group, train, test in folds:
        config, scores = experiment.select(train)
        fitted = experiment.fit(train, config)
        fitted_folds.append((fitted, test))
        rows = experiment.evaluate(fitted, test)
        all_rows.extend(rows)
        baseline_rows.extend(experiment.evaluate(experiment.fit(train, BASELINE), test))
        selections.append({"held_out_group": str(group), "selected_model": config.name,
                           "training_selection_scores": scores, "train_files": [experiment.clips[i]["filename"] for i in train],
                           "test_files": [experiment.clips[i]["filename"] for i in test], **metrics(rows)})
    report["nested_group_evaluation"] = {
        "selected": metrics(all_rows), "fixed_baseline_with_same_rejection": metrics(baseline_rows),
        "folds": selections, "predictions": all_rows, "baseline_predictions": baseline_rows,
        "margin_curve": margin_curve(all_rows),
    }
    for trainblock, testblock in ((1, 2), (2, 1)):
        train, test = np.where(experiment.blocks == trainblock)[0], np.where(experiment.blocks == testblock)[0]
        config, scores = experiment.select(train)
        fitted = experiment.fit(train, config)
        rows = experiment.evaluate(fitted, test)
        report["block_evaluation"][f"{trainblock}_to_{testblock}"] = {
            "selected_model": config.name, "selected": metrics(rows),
            "fixed_baseline_with_same_rejection": metrics(experiment.evaluate(experiment.fit(train, BASELINE), test)),
            "training_selection_scores": scores, "predictions": rows,
        }
        if trainblock == 1:
            experiment.save(fitted, "block1_model.npz", scores)
    report["robustness"] = {}
    for kind in ("amplitude_0.25", "shift_200ms", "speed_0.85", "speed_1.15", "noise_20db_event"):
        sequences, gates, quality = perturb(experiment, kind)
        rows = []
        for fitted, test in fitted_folds:
            rows.extend(experiment.evaluate(fitted, test, sequences, gates, quality))
        report["robustness"][kind] = {**metrics(rows), "predictions": rows}
    config, scores = experiment.select(all_indices)
    experiment.save(experiment.fit(all_indices, config), "pilot_model.npz", scores)
    report["full_pilot_candidate"] = config.name
    report["limitations"] = json.loads((ROOT / report["plan"]).read_text())["limits"]
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"candidates": report["candidates"],
                      "nested": report["nested_group_evaluation"]["selected"],
                      "baseline": report["nested_group_evaluation"]["fixed_baseline_with_same_rejection"],
                      "blocks": {k: {"model": v["selected_model"], **v["selected"]} for k, v in report["block_evaluation"].items()},
                      "robustness": {k: {kk: vv for kk, vv in v.items() if kk != "predictions"} for k, v in report["robustness"].items()},
                      "full_pilot_candidate": config.name}, indent=2))


if __name__ == "__main__":
    main()
