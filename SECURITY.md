# Local-only scope

Swipe is a single-user localhost experiment, not a hosted application. The server binds to `127.0.0.1`, validates Host, uses a per-process token for writes/inference, and serves an explicit list of UI assets. Do not expose it through a tunnel, public interface, or reverse proxy. The token is not a multi-user authentication system and does not protect against other software running as the same local user.

Live preview audio and predictions stay in process memory and are not added to the dataset. The guided and continuous recorders deliberately save WAV audio and metadata under `data/`. Export downloads contain those recordings. Git ignores these files and complete local checkpoints, but cloud-sync or backup tools operate independently of the app.

Only use trusted model/source files. Model hashes detect inconsistent files, not malicious changes by someone able to edit both files and manifests. The runtime loads NumPy banks with `allow_pickle=False`.

For a security report, use GitHub's private vulnerability reporting if enabled on the repository. Otherwise contact the repository owner privately before posting sensitive details. Do not put credentials, private recordings, or device metadata in public issue attachments.

macOS keyboard output is opt-in and requires Accessibility permission. Only accepted predictions can mint a three-second, single-use ticket owned by the armed browser tab. Keys have a 350ms cooldown and a 20-second heartbeat lease. Stop and page exit disarm output; network failure can leave a lease alive until it expires. This limits accidental repeats, but gesture recognition remains experimental: choose bindings appropriate for occasional mistakes. No arbitrary shell commands are accepted.

The native macOS app uses a private, length-bounded stdin/stdout child process instead of HTTP. Native actions run only after explicitly starting listening and receiving an accepted, non-overlapping result. App launch is always paused; sleep/lock actions pause capture first. Microphone permission is required; keyboard posting also requires Accessibility access. Saved settings contain action choices, shortcut key codes and the local engine path, not microphone recordings. The app currently depends on a trusted local checkout and Python environment.
