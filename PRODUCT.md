# Grille Lab
<!-- impeccable:product-schema 1 -->

## Platform
web

## Users and purpose
Grille Lab explores microphone-based finger-swipe input on the left speaker grille of a 14-inch M3 Pro MacBook Pro. The current preview uses an MFCC8 reference bank from three built-in-microphone sessions, with a legacy matcher for uncertain results. It includes guided isolated-gesture collection and continuous 15-second recording for further experiments. The owner found the preview usable in live testing; transfer to other users or setups has not been established.

## Scope
The short-clip collector prompts swipe up/down and captures microphone audio with labels, including idle and typing negatives. Short-clip collection starts only after user microphone connection and a separate built-in-microphone confirmation. Preserve session/block provenance; avoid model accuracy claims before actual evaluation.

The continuous recorder lets the user choose a microphone, alternating cues or free up/down movement, and optional notes. Start requests microphone access, then gives a three-second countdown before capturing one uninterrupted 15-second take. Cued mode displays silent up/down prompts every 1.5 seconds; free mode follows the user's own pace. Capture ends automatically and disconnects the microphone before saving. Cancel or hiding the tab discards an unfinished take and releases the microphone. A save failure keeps the finished audio in page memory; Retry saving resubmits the same take, so the page must stay open until it succeeds. A saved-take confirmation and recording ZIP link appear after a successful save. Prompt metadata describes intended movements and displayed cue times, not verified gesture directions or boundaries; free movement remains an unlabeled sequence for later review.

The localhost-only live preview starts listening only when the user selects Start preview. It shows Up (toward the screen), Down (toward the palm rest), or Uncertain, with visible listening status and a short recent-results list. Preview audio and results are transient and are not saved to disk. No shortcuts or system actions run. Stop listening, leaving the page, or hiding the tab releases the microphone. This remains an experimental preview, not a validated everyday input device; scores are not presented as calibrated confidence percentages.

## Implementation decisions
Local Python standard-library server and plain HTML/CSS/JavaScript. The collector was chosen to avoid dependencies; the learned preview uses the project's analysis environment, as documented in README.md. Each pilot has 28 clips in two randomized blocks. The short-clip collector saves two-second mono WAV clips and JSON metadata in the project data folder. Each continuous take saves one 15-second mono Float32 WAV and session JSON containing the mode, microphone settings, notes, planned and displayed prompt timelines, and the explicit need to review gesture labels. Displayed cue times estimate audio-relative presentation time, not measured finger onset. Both recording workflows save inside the project data folder. A cloud-sync provider may sync those files if the project is in a synced directory. The app itself has no external network service. The preview processes audio in memory on this Mac and leaves saved recording sessions unchanged.
