"""In-memory local inference. No recording, feedback storage, or system actions."""
import hashlib
import io
import json
from pathlib import Path
import threading
import time

import numpy as np

from audio_format import decode_wav
from fast_preview import predict

ROOT = Path(__file__).resolve().parent


class PreviewService:
    def __init__(self):
        folder = ROOT / "diagnostics/preview"
        config = json.loads((folder / "model.json").read_text())
        blob = (folder / "model.npz").read_bytes()
        if hashlib.sha256(blob).hexdigest() != config["model_sha256"]:
            raise ValueError("Preview model has changed; rebuild its manifest.")
        for name, expected in config["implementation_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Preview feature code changed; rebuild the model before using it.")
        engine = json.loads((folder / "engine.json").read_text())
        for name, expected in engine["implementation_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Preview inference engine changed; verify it and update its manifest.")
        with np.load(io.BytesIO(blob), allow_pickle=False) as bank:
            self.model = {key: bank[key] for key in bank.files}
        self.engine = engine['engine']
        self.aligned_model = None
        self.mfcc_model = None
        self.mfcc_config = None
        if self.engine in ('selective_encoding_fallback','mfcc8_fallback'):
            from aligned_preview import load_model
            self.aligned_model, _ = load_model()
        elif self.engine != 'exact_batched_dtw':
            raise ValueError('Unknown preview inference engine; restore a verified configuration.')
        if self.engine == 'mfcc8_fallback':
            from mfcc_preview import load_model
            self.mfcc_model, self.mfcc_config = load_model()
        self.config = config
        self.lock = threading.Lock()

    def info(self):
        if self.mfcc_config is not None:
            additional=sum(r['session'] not in self.config['sessions'] for r in self.mfcc_config['references'])
            return {"ready":True,"training_clips":self.config['training_clips']+additional,
                    "training_sessions":len(self.mfcc_config['sessions']),"reference_swipes":len(self.mfcc_config['references']),
                    "audio_saved":False,"system_actions":False,"status":"Experimental preview","recognizer":self.engine}
        return {"ready": True, "training_clips": self.config["training_clips"],
                "training_sessions": len(self.config["sessions"]), "audio_saved": False,
                "system_actions": False, "status": "Experimental preview", "recognizer": self.engine}

    def classify(self, blob):
        samples, rate, _ = decode_wav(blob, array=True)
        if not 1.8 <= len(samples) / rate <= 2.2:
            raise ValueError("Preview expects a two-second audio window.")
        # Reject concurrent work instead of building an unbounded audio queue.
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("Preview is busy. Wait for the current result, then swipe again.")
        try:
            start = time.perf_counter()
            if self.mfcc_model is not None:
                from mfcc_fallback import predict as mfcc_predict
                result = mfcc_predict(samples, rate, self.mfcc_model, self.model, self.aligned_model)
            elif self.aligned_model is None:
                result = predict(samples, rate, self.model)
            else:
                from selective_preview import predict as selective_predict
                result = selective_predict(samples, rate, self.model, self.aligned_model)
            return {**result, "processing_ms": round((time.perf_counter() - start) * 1000), "audio_saved": False}
        finally:
            self.lock.release()
