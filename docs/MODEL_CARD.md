# MFCC8 with legacy fallback

## Intended use

An experimental local preview for isolated up/down finger swipes on the left speaker grille of a 14-inch M3 Pro MacBook Pro, using its built-in microphone. The preview displays a direction or rejects an uncertain sound. It is not connected to system actions. The user's positive live-usability report on 2026-11-07 is qualitative feedback, not a measured accuracy result.

## Bundled artifacts

The active engine is `mfcc8_fallback` in [`models/preview/engine.json`](../models/preview/engine.json).

| Artifact | Role |
| --- | --- |
| `models/mfcc_preview/model.npz` | 60 MFCC8 swipe references, novelty cutoff, and copied gate parameters |
| `models/preview/model.npz` | Original spectral reference bank and gate |
| `models/aligned_preview/model.npz` | Legacy-encoding reference bank used by the old fallback |

Adjacent JSON manifests record reference provenance and SHA-256 hashes. NumPy banks contain derived feature templates, labels, and parameters, not raw WAV audio. These are examples derived from one contributor's recordings, not a broadly trained population model. Reference session IDs and clip filenames remain in the manifests/reports for provenance. Original session metadata, device identifiers, raw audio, intermediate caches, and complete local checkpoints are excluded from Git.

The active engine manifest is authoritative. Some bank metadata describes its earlier candidate status at creation time; it is retained unchanged for provenance.

## Signal pipeline

The browser provides a mono stream; the project does not expose individual physical microphones. Audio is decoded without clipping Float32 samples and resampled to 44.1 kHz. Noise-aware cropping and spectral subtraction precede 24 mel bands and eight non-energy cepstral coefficients. Per-clip temporal centering, temporal derivatives, and RMS normalization produce a 32 × 16 sequence.

Three-neighbor DTW matching uses radius 4 and a fixed margin threshold of 0.2. The active full-bank novelty cutoff, 0.3708665265780037, is the 95th percentile of nearest same-direction distances in another reference session. It was calibrated from labeled references. The earlier fitted background gate was copied unchanged. Thus feature extraction is deterministic, but the system includes fitted/calibrated parameters.

Only an uncertain primary result invokes the legacy matcher, and only an accepted legacy direction replaces it. Primary directions and background results are retained. The legacy matcher includes an in-memory conversion for the older PCM16 encoding; original audio is never rewritten.

The capture cascade may submit an early isolated event after at least 200 ms and 100 ms of quiet. If not accepted, a full two-second window is submitted at about one second after onset. Accepted-event IDs and source timestamps suppress duplicate display. One pending request and one queued fallback bound client work; the service refuses concurrent inference.

## Development evidence

Three built-in-microphone sessions contain 60 isolated swipes and 24 idle/typing negatives. The first used saturated PCM16 encoding; later sessions preserve Float32 samples. A separate headset-microphone session was excluded. Continuous takes include a mixture of contact reversals and lifted returns, without reliable direction timing, so they are not used for accuracy claims.

The [reference-excluded check](../docs/evaluation/mfcc_preview/fallback_checks/report.json) reported 49 correct accepted swipes, 1 wrong accepted swipe, 10 rejected swipes, and 0/24 background activations. The earlier matcher had 37 correct, 1 wrong, and 22 rejected. Gates and MFCC reference calibration exclude the query session; a reused legacy cutoff remains a limitation. Feature choice and fallback policy followed inspection of these same recordings. See the [check plan](../docs/evaluation/mfcc_preview/fallback_checks/plan.json).

The [stateful replay](../docs/evaluation/mfcc_preview/replay_fallback/report.json) preserved all previously correct accepted clips under both tested lead-ins. With repeated recorded background it produced 59/60 correct accepted directions, no wrong accepted directions, and no background activations. With zero lead-in it produced 50/60 correct accepted directions. References include these recordings, so these are integration checks, not held-out accuracy. The report's `raw` baseline key denotes the previous selective matcher. Replay assumes immediate acknowledgements and pauses signal time during inference.

The [paired benchmark](../docs/evaluation/mfcc_preview/benchmark.json) measured 252 calls per engine: 7.01 ms median / 8.59 ms p95 for the current system, versus 6.77 / 8.22 ms previously. It includes gate and fallback computation, but excludes WAV decoding, capture, HTTP, and rendering. See the [benchmark plan](../docs/evaluation/mfcc_preview/benchmark_plan.json).

## Limits and reproducibility

- One laptop/setup and a small reused dataset; no demonstrated generalization across users, hardware, days, pressure, or microphone gain.
- One accepted direction error remains in the reference-excluded check. Background tests cover only idle and typing, not speech, music, bumps, or all everyday noise.
- Holding a session out of references does not undo design decisions made after inspecting its results.
- Synthetic perturbations and integration replays are not fresh physical-gesture validation.
- CI tests loading/integrity, deterministic algorithms, and application behavior. It does not reproduce private-dataset accuracy metrics.
- Research scripts need original recordings and intermediate experiment artifacts that are not shipped. Reports are provided for inspection; full result reproduction is not possible from the public checkout alone.

Runtime files are hash-verified on load. Preserve the bank, manifest, and corresponding feature code together. The legacy rollback is available by setting the engine to `selective_encoding_fallback` and restarting the server; it uses the bundled old banks without fitting. New development should be compared against this saved usable baseline before replacing it.

## Repository layout migration

Application code now lives in `swipe/`, runtime banks in `models/`, browser assets in `web/`, and these saved reports in `docs/evaluation/`. The model arrays and calibration values were not changed. Runtime manifests were updated for the relocated source paths and import-only changes, and predictions were compared exactly on all 84 labeled development clips.

Historical report bodies and plans retain their original path strings and source hashes as evidence of the runs that produced them. Those historical hashes are not the active runtime verification manifests. To reproduce an old frozen experiment byte-for-byte, use its saved checkpoint or the corresponding Git commit; do not rewrite its report or disable an integrity check.
