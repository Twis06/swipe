# Local-only scope

Swipe is a single-user localhost experiment, not a hosted application. The server binds to `127.0.0.1`, validates Host, uses a per-process token for writes/inference, and serves an explicit list of UI assets. Do not expose it through a tunnel, public interface, or reverse proxy. The token is not a multi-user authentication system and does not protect against other software running as the same local user.

Live preview audio and predictions stay in process memory and are not added to the dataset. The guided and continuous recorders deliberately save WAV audio and metadata under `data/`. Export downloads contain those recordings. Git ignores these files and complete local checkpoints, but cloud-sync or backup tools operate independently of the app.

Only use trusted model/source files. Model hashes detect inconsistent files, not malicious changes by someone able to edit both files and manifests. The runtime loads NumPy banks with `allow_pickle=False`.

For a security report, use GitHub's private vulnerability reporting if enabled on the repository. Otherwise contact the repository owner privately before posting sensitive details. Do not put credentials, private recordings, or device metadata in public issue attachments.
