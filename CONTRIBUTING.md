# Contributing

Start with the setup in [README.md](README.md). The default runtime needs only NumPy and SciPy; tests also need Node.js 22+. No microphone permission is required to run the checks:

```sh
python scripts/check.py
python scripts/check_release.py
```

The explicit public suite avoids historical tests that depend on undistributed recordings or fit models. Do not use blanket unittest discovery as the public test command. Add portable regressions to `test_public_runtime.py` or register a new suite in `scripts/check.py`. Put temporary recordings in a temporary directory, not the real `data/` folder.

## Code map

| Area | Files |
| --- | --- |
| Local HTTP API and recording persistence | `server.py`, `continuous_recording.py`, `audio_format.py` |
| Live UI and capture | `preview.html`, `preview.js`, `preview.css`, `cascade-preview-worklet.js` |
| Guided / continuous collectors | `index.html`, `app.js`, `continuous.html`, `continuous.js`, `audio-worklet.js` |
| Active inference | `preview_service.py`, `mfcc_preview.py`, `mfcc_fallback.py`, `mfcc_components.py`, `fast_preview.py` |
| Shared features and legacy fallback | `gesture_model.py`, `preview_model.py`, `swipe_feature_families.py`, `selective_preview.py`, `aligned_preview.py` |
| Runtime banks and provenance | Allowlisted files under `diagnostics/preview`, `diagnostics/aligned_preview`, `diagnostics/mfcc_preview` |
| Research and data-dependent evaluation | Root `probe_*`, `compare_*`, `build_*`, `train_*`, `check_mfcc_*`, and related scripts |

Research tools are retained in place to preserve historical paths and manifest hashes. For those tools, install `python -m pip install -r requirements-analysis.txt`. Many need the original local recordings and intermediate diagnostic files; installing dependencies alone does not make private-data experiments reproducible. Read each script before running it. Some write new model banks or require output folders not to exist.

## Model changes

Preserve the original audio and the known usable model. Keep feature/calibration choices inside the reference split, report rejected and wrong accepted gestures separately, and evaluate background activations. Describe reused-data evidence as development evidence. Fresh-session validation requires freezing choices before collecting the session.

The runtime checks SHA-256 hashes of banks and feature code. If changing these files, verify behavior and update the corresponding manifests deliberately. Do not simply remove verification or regenerate hashes to hide an untested change. Keep source and model updates together in a reviewable commit. Algorithm changes should cover both isolated predictions and stateful capture behavior.

## Before publishing or opening a pull request

Run both check commands and inspect `git diff --cached --stat` plus the staged diff. Do not force-add `data/`, `checkpoints/`, local notes, audio exports, credentials, or experiment caches. `.gitignore` allowlists only the runtime model banks and selected evidence from diagnostics. The release checker also rejects these excluded artifacts if they have already been staged/tracked.

Describe what changed, why, and which checks ran. Include setup, microphone, gesture method, and rejection behavior when reporting a recognition issue. Do not attach audio containing private speech or identifying session/device metadata to public issues.

Contributions to the project are under its [MIT license](LICENSE). Dependencies retain their own licenses.
