# Swipe for macOS

A small experimental native SwiftUI/AppKit menu-bar app. macOS 14 or later; build with Swift 6 or later. Apple Silicon is the tested target. No browser or HTTP server is required.

Map each direction to **Volume up**, **Volume down**, **Mute/unmute**, **Lock screen**, **Sleep Mac**, **Turn display off**, **Keyboard shortcut**, or **Do nothing**. Volume uses the current output device's CoreAudio controls. Output devices without software volume/mute control show a recoverable error. Lock uses macOS's standard Control–Command–Q shortcut. Sleep uses IOKit power management; display off uses the fixed system `pmset displaysleepnow` command.

## Build and run

From the repository root, set up the existing recognizer once:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash macos/scripts/build-app.sh
open dist/Swipe.app
```

The app bundle is ad-hoc signed for local use, not notarized for distribution. Build it locally with Swift 6 or later from Command Line Tools (`xcode-select --install`) or Xcode. The build script records this checkout's location only in the ignored local app bundle. You can move the app; if you move the checkout, use **Setup → Choose project…** to reconnect it.

This version is a **native app around the existing Python recognizer**, not an all-Swift ML rewrite. It uses the checkout's `.venv` and verified model banks, rather than embedding a second Python runtime. The worker launches when starting the listener and exits on pause/quit. The app bundle alone is not a standalone distribution. No packages or models are downloaded at runtime.

## Use

1. Choose an action for each direction; defaults are up for volume up and down for volume down. For a keyboard shortcut, choose **Record** and press the shortcut; Escape cancels.
2. Click **Start listening** and allow microphone access when macOS asks. The app selects the built-in input locally without changing the system's default input device.
3. Keep still for one second, then swipe the left speaker grille once toward the screen or palm rest. Lift your finger and wait for the result.
4. Pause or quit from the menu bar. Closing the settings window keeps the explicitly started listener running.

Every launch begins paused. Mappings and volume step are saved in macOS preferences; listening state is not. Lock screen, Sleep Mac, and Turn display off pause before executing, so waking/unlocking does not automatically resume. Only accepted, non-overlapping predictions cause actions; uncertain/background results are ignored. Changing mappings is disabled while listening. Use discrete swipes with a lift between them; continuous finger movement is not a supported control mode.

**Permissions:** volume/mute, system sleep, and display sleep do not need Accessibility. Keyboard shortcuts and lock screen do; click **Allow access**, enable Swipe in **System Settings → Privacy & Security → Accessibility**, then return. Accessibility is requested only through that action. Microphone denial has a separate recovery button. System-level actions should be mapped with the experimental recognizer's possible mistakes in mind.

## Architecture and checks

- `SwipeCore/GestureCapture.swift`: port of the released 128-frame early/fallback capture logic. Synthetic parity fixtures compare exact PCM hashes, frame origins, and emission times against the browser worklet at 44.1/48 kHz.
- `SwipeCore/EventGate`: accepted-result and duplicate-event gate.
- `SwipeApp/Microphone.swift`: native AVAudioEngine, built-in input selection, 128-frame reblocking, first-second warmup, rate/device validation.
- `SwipeApp/Recognizer.swift`: bounded one-request stdin/stdout bridge, timeout/exit handling, stale result protection via controller generation.
- `swipe/native_worker.py`: original `PreviewService` and model verification over private stdin/stdout; reads local models, with no network listener, audio recordings, or training.
- `SwipeApp/SystemActions.swift`: native action dispatch with required OS permissions.

```sh
swift test --package-path macos
.venv/bin/python -m unittest tests.test_native_worker
# Regenerate synthetic worklet reference metadata after an intentional capture change:
node macos/scripts/capture-fixtures.cjs
```

Native capture parity preserves software behavior; it does not establish recognition accuracy with a different OS microphone signal path. Live use remains an experimental prototype developed on a 14-inch M3 Pro MacBook Pro. No microphone data enters source control.
