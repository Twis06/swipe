"""Classify an isolated WAV as up, down, background, or uncertain."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from swipe.audio_format import decode_wav
from swipe.gesture_model import classify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1] / "diagnostics/improvement/noise_aware/pilot_model.npz")
    args = parser.parse_args()
    try:
        blob = args.wav.read_bytes()
        with np.load(args.model, allow_pickle=False) as model:
            if hashlib.sha256(blob).hexdigest() in model["training_sha256"]:
                raise ValueError("This WAV was used to fit the model. Refusing a misleading training-file prediction. Use block1_model.npz for a block-2 example.")
            samples, rate, _ = decode_wav(blob)
            result = classify(samples, rate, model)
        print(json.dumps({**result, "model": str(args.model),
                          "scope": "Experimental isolated-clip recognition. No microphone or system actions. Pilot scores do not establish new-session reliability."}, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
