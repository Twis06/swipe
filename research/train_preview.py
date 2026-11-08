"""Bounded preprocessing comparison and preview-model export. No raw data writes."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from swipe.audio_format import decode_wav
from swipe.gesture_model import dtw, direction, decision
from swipe.preview_model import features
from research.validate_session import summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/preview"
SESSIONS = ("20261103T165139_cf7201", "20261103T200050_87dcb0")
CANDIDATES = [(kind, k) for kind in ("raw", "soft_polarity", "polarity") for k in (1, 3)]


class Experiment:
    def __init__(self):
        self.clips, self.sequences, self.gates, self.quality = [], {kind: [] for kind, _ in CANDIDATES}, [], []
        for sid in SESSIONS:
            s = json.loads((ROOT / "data" / sid / "session.json").read_text())
            for clip in s["clips"]:
                if not clip["accepted"]:
                    continue
                blob = (ROOT / "data" / sid / clip["filename"]).read_bytes()
                if hashlib.sha256(blob).hexdigest() != clip["sha256"]:
                    raise ValueError("Original audio changed.")
                x, sr, _ = decode_wav(blob)
                for kind in self.sequences:
                    sequence, gate, quality = features(x, sr, kind)
                    self.sequences[kind].append(sequence)
                self.gates.append(gate); self.quality.append(quality)
                self.clips.append({**clip, "session": sid})
        self.gates = np.array(self.gates)
        self.sequences = {k: np.array(v) for k, v in self.sequences.items()}
        self.y = np.array([c["label"] for c in self.clips])
        self.swipe = np.isin(self.y, ("up", "down"))
        self.sessions = np.array([c["session"] for c in self.clips])
        self.groups = np.array([f'{c["session"]}_{c["block"]}_{c["pace"]}' for c in self.clips])
        self.cache = {}

    def distances(self, kind):
        if kind not in self.cache:
            sequences = self.sequences[kind]
            d = np.zeros((len(self.y), len(self.y)))
            for i in range(len(self.y)):
                for j in range(i):
                    d[i, j] = d[j, i] = dtw(sequences[i], sequences[j], 4)
            self.cache[kind] = d
        return self.cache[kind]

    def select(self, train):
        refs = train[self.swipe[train]]
        scores = {}
        for kind, k in CANDIDATES:
            correct = 0
            d = self.distances(kind)
            for group in np.unique(self.groups[refs]):
                fit, test = refs[self.groups[refs] != group], refs[self.groups[refs] == group]
                for i in test:
                    predicted, _, _ = direction(d[i, fit], self.y[fit], k)
                    correct += predicted == self.y[i]
            scores[f"{kind}_k{k}"] = int(correct)
        return max(CANDIDATES, key=lambda c: scores[f"{c[0]}_k{c[1]}"]), scores

    def fit(self, train, candidate):
        kind, k = candidate
        refs = train[self.swipe[train]]
        d = self.distances(kind)
        calibration = []
        for i in refs:
            others = refs[(self.y[refs] == self.y[i]) & (self.groups[refs] != self.groups[i])]
            calibration.append(float(np.min(d[i, others])))
        gate = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", random_state=0))
        gate.fit(self.gates[train], self.swipe[train].astype(int))
        scaler, clf = gate.named_steps["standardscaler"], gate.named_steps["logisticregression"]
        model = dict(feature_input=np.array(kind), radius=np.array(4), neighbors=np.array(k),
                     templates=self.sequences[kind][refs], labels=self.y[refs],
                     gate_mean=scaler.mean_, gate_scale=scaler.scale_, gate_coef=clf.coef_[0],
                     gate_intercept=np.array(float(clf.intercept_[0])), distance_cutoff=np.array(float(np.quantile(calibration, .95))),
                     margin_threshold=np.array(.2), training_sha256=np.array([self.clips[i]["sha256"] for i in train]))
        return {"model": model, "refs": refs, "gate": gate, "candidate": candidate}

    def evaluate(self, fit, test):
        kind, k = fit["candidate"]
        model, refs = fit["model"], fit["refs"]
        gate = fit["gate"].predict_proba(self.gates[test])[:, 1]
        rows = []
        for i, score in zip(test, gate):
            predicted, margin, distance = direction(self.distances(kind)[i, refs], self.y[refs], k)
            result, reason = decision(predicted, margin, distance, float(model["distance_cutoff"]), score, self.quality[i]["silent"])
            rows.append({"file": self.clips[i]["filename"], "session": str(self.sessions[i]), "truth": str(self.y[i]),
                         "prediction": result, "reason": reason, "direction_candidate": predicted,
                         "distance_margin": margin, "nearest_distance": distance, "swipe_gate_score": float(score),
                         "model": f"{kind}_k{k}"})
        return rows


def main():
    OUT.mkdir(exist_ok=True)
    e = Experiment()
    all_ids = np.arange(len(e.y))
    report = {"status": "Exploratory reuse; both sessions already inspected", "plan": "diagnostics/preview_plan.json",
              "cross_session": {}, "fixed_candidates": {}}
    for sid in SESSIONS:
        train, test = np.where(e.sessions == sid)[0], np.where(e.sessions != sid)[0]
        candidate, scores = e.select(train)
        fitted = e.fit(train, candidate)
        rows = e.evaluate(fitted, test)
        report["cross_session"][sid] = {"selected": list(candidate), "training_scores": scores,
                                       "summary": summarize(rows), "predictions": rows}
        # Both cases are retained as predeclared descriptive comparisons.
        report["cross_session"][sid]["all_fixed_methods"] = {
            f"{kind}_k{k}": summarize(e.evaluate(e.fit(train, (kind, k)), test)) for kind, k in CANDIDATES}
    grouped, folds = [], []
    for group in np.unique(e.groups):
        train, test = np.where(e.groups != group)[0], np.where(e.groups == group)[0]
        candidate, scores = e.select(train)
        rows = e.evaluate(e.fit(train, candidate), test)
        grouped.extend(rows)
        folds.append({"group": str(group), "selected": list(candidate), "summary": summarize(rows)})
    report["nested_grouped"] = {"summary": summarize(grouped), "folds": folds, "predictions": grouped}
    for candidate in CANDIDATES:
        rows = []
        for group in np.unique(e.groups):
            train, test = np.where(e.groups != group)[0], np.where(e.groups == group)[0]
            rows.extend(e.evaluate(e.fit(train, candidate), test))
        report["fixed_candidates"][f"{candidate[0]}_k{candidate[1]}"] = summarize(rows)
    candidate, scores = e.select(all_ids)
    fitted = e.fit(all_ids, candidate)
    np.savez_compressed(OUT / "model.npz", **fitted["model"])
    config = {"status": "Experimental preview, no system actions", "feature_input": candidate[0], "neighbors": candidate[1],
              "radius": 4, "margin_threshold": .2, "sessions": list(SESSIONS), "training_clips": len(e.y),
              "selection_scores": scores, "model_sha256": hashlib.sha256((OUT / "model.npz").read_bytes()).hexdigest(),
              "implementation_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('swipe/gesture_model.py', 'swipe/preview_model.py')},
              "limits": "No unused evaluation session remains. Preview feedback is transient and is not model training."}
    (OUT / "model.json").write_text(json.dumps(config, indent=2))
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"selected": candidate, "scores": scores,
                      "nested": report["nested_grouped"]["summary"],
                      "cross_session": {sid: {"selected": v["selected"], "summary": v["summary"], "fixed": {k: (m["forced_direction_correct"],m["accepted_correct"],m["accepted_swipes"]) for k,m in v["all_fixed_methods"].items()}} for sid,v in report["cross_session"].items()}}, indent=2))


if __name__ == "__main__":
    main()
