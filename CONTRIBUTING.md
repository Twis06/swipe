# Contributing

Start with the setup in [README.md](README.md). The default runtime needs only NumPy and SciPy; tests also need Node.js 22+. No microphone permission is required to run the checks:

```sh
python scripts/check.py
python scripts/check_release.py
```

The explicit public suite avoids historical tests that depend on undistributed recordings or fit models. Do not use blanket unittest discovery as the public test command. Add portable regressions to `tests/test_public_runtime.py` or register a new suite in `scripts/check.py`. Put temporary recordings in a temporary directory, not the real `data/` folder.

## Code map

| Area | Location |
| --- | --- |
| Application entry point | `swipe/__main__.py` → `swipe/server.py`; run `python -m swipe` |
| HTTP API, capture persistence, audio decoding | `swipe/server.py`, `swipe/continuous_recording.py`, `swipe/audio_format.py` |
| Feature extraction and recognition | `swipe/preview_service.py`, `swipe/mfcc_preview.py`, `swipe/mfcc_fallback.py`, and supporting modules in `swipe/` |
| Resource locations | `swipe/paths.py`; checkout-relative paths independent of request routing |
| Browser application | `web/` (served through an explicit URL allowlist) |
| Bundled reference banks and runtime manifests | `models/preview/`, `models/aligned_preview/`, `models/mfcc_preview/` |
| Portable regression coverage | `tests/` and `tests/js/`; selected by `scripts/check.py` |
| Offline experiments and old worklets | `research/` and `research/worklets/` |
| Public evaluation evidence | `docs/evaluation/`; historical reports remain unchanged |
| Local recordings / generated work / saved baselines | `data/`, `diagnostics/`, `checkpoints/` (all Git-ignored) |

Keep application imports inside the `swipe` package. The runtime must not import experiments or tests. Tests and research can import runtime modules using `from swipe...`. Use explicit package imports; do not extend `sys.path` to revive the old flat layout. The standalone check scripts add the checkout root only when needed for direct execution.

Research tools run as modules from the checkout root, for example `python -m research.recognize_clip --help`. Install their optional packages with `python -m pip install -r requirements-analysis.txt`. Most experiments need the original local recordings and intermediate diagnostics; installing dependencies alone does not make private-data results reproducible. See [research/README.md](research/README.md). Some scripts fit models, execute on import, or require output folders not to exist, so read them before running them.

The application is currently run from its checkout rather than distributed as a wheel. `web/` and `models/` deliberately stay visible beside the source. `Start Swipe.command` launches the module from the correct directory.

## Model changes

Preserve the original audio and the known usable model. Keep feature/calibration choices inside the reference split, report rejected and wrong accepted gestures separately, and evaluate background activations. Describe reused-data evidence as development evidence. Fresh-session validation requires freezing choices before collecting the session.

The runtime checks SHA-256 hashes of banks and feature code. If changing these files, verify behavior and update the corresponding manifests deliberately. Do not simply remove verification or regenerate hashes to hide an untested change. Keep source and model updates together in a reviewable commit. Algorithm changes should cover both isolated predictions and stateful capture behavior.

## Before publishing or opening a pull request

Run both check commands and inspect `git diff --cached --stat` plus the staged diff. Do not force-add `data/`, `checkpoints/`, local notes, audio exports, credentials, or experiment caches. `.gitignore` excludes generated diagnostics and allowlists the reviewed runtime banks in `models/`. Published evidence is kept separately in `docs/evaluation/`. The release checker also rejects these excluded artifacts if they have already been staged/tracked.

Describe what changed, why, and which checks ran. Include setup, microphone, gesture method, and rejection behavior when reporting a recognition issue. Do not attach audio containing private speech or identifying session/device metadata to public issues.

Contributions to the project are under its [MIT license](LICENSE). Dependencies retain their own licenses.
