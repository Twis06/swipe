# Recording gestures

## Guided pilot

Open `http://localhost:8765/`, connect the explicit built-in MacBook microphone, and confirm the input level responds to a gentle swipe. The short pilot contains 28 two-second clips in two randomized blocks: 10 up, 10 down, 4 idle, and 4 typing examples. A larger option collects 56 clips.

Up means toward the screen; down means toward the palm rest. Position during the countdown, make one swipe when Record appears, and lift your finger to return. Cues are silent. Keep the swipe location and microphone setup consistent within a session. Record notes when changing pressure, location, or environment.

Use Continue at a block break. Pause cancels an unfinished trial. Discard last & retry marks the prior take as rejected and repeats its prompt. On a save failure, keep the page open and retry: the completed audio remains in memory. A lost upload response can be retried without duplicating an accepted trial.

Files are saved under `data/<session_id>/`: raw WAV files and `session.json` with the plan, labels, pace, block, timestamps, accepted/rejected status, sample counts, audio settings, and hashes. The page can export a ZIP. These files are ignored by Git. They may still be synced by software outside the app.

## Continuous takes

Open `http://localhost:8765/continuous`, select the built-in microphone, and choose cued or free movement. Each take records 15 seconds after a countdown. Cancel or hiding the tab discards an unfinished take and releases the microphone.

Prompt times record what appeared on screen, not when a finger actually moved. If moving at your own pace, choose free mode and treat direction/timing as unlabeled until reviewed. Never infer true up/down labels from alternation, especially when the finger sometimes stays in contact and sometimes lifts for the return.

## Audio quality

Current recordings preserve mono IEEE Float32 samples at the AudioContext's actual rate. Floats above ±1 are retained rather than clamped. An amber high-peak indicator alone is not clipping evidence. Repeated flat-topped samples produce a red possible-clipping warning and pause collection for review. If that happens during normal swipes, lower macOS input volume and check again. This cannot repair distortion already introduced upstream.

The browser requests echo cancellation, noise suppression, and automatic gain control off, and stores the reported settings. This does not guarantee access to unprocessed hardware audio or separate microphone channels.

## Evaluation discipline

Use accepted clips, review quality flags, and split by whole session before selecting features or thresholds. Avoid splitting overlapping windows across training and evaluation. Cue labels are intended gestures, not independently verified ground truth. Idle and typing do not exhaust possible background sounds.

Recording does not update the preview automatically. The bundled model stays fixed until a developer explicitly builds, evaluates, and promotes a replacement. Do not overwrite the saved usable baseline to experiment.
