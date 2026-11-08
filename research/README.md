# Offline research

This directory contains development experiments, calibration, replay, diagnostics, and prototype worklets. The application never imports research code. These tools are optional and may require private recordings and caches not distributed with the repository.

Run modules from the checkout root:

```sh
python -m pip install -r requirements-analysis.txt
python -m research.recognize_clip --help
```

Research tools are not the live application entry point. Start the application with `python -m swipe`.

| Tools | Purpose |
| --- | --- |
| `probe_*`, `mfcc_direction.py`, `encoding_alignment.py` | Feature and domain-alignment experiments |
| `build_*`, `train_preview.py`, `improve_model.py`, `calibrate_session.py` | Build or fit candidate reference banks |
| `compare_*`, `replay_*.cjs`, `benchmark_*` | Stateful capture replay and performance measurements |
| `analyze*`, `diagnose_audio.py`, `investigate_validation.py` | Recording and error analysis |
| `evaluate_frozen_validation.py`, `validate_session.py` | Frozen-model evaluation |
| `worklets/` | Superseded capture prototypes retained for comparisons |

Inputs and generated outputs are resolved from the checkout root, not from the shell's location inside this directory. Invoke Python modules with `-m` so package imports are resolved normally. JavaScript replay workers locate inputs relative to their own file. Generated research output remains in the ignored `diagnostics/` tree; writing a candidate does not automatically replace the active `models/` banks.

Some older scripts execute work at module import time. Do not bulk-import research modules. Read the entry point and required files before running an experiment; some fit models or refuse to overwrite existing output folders. The public test runner deliberately selects only tests that do not fit models or need private audio.

Published reports in `docs/evaluation/` are immutable evidence from earlier runs. Their original paths and hashes are intentionally retained. Scripts tied to old feature/source hashes must be run from their original Git commit or saved local checkpoint for exact historical reproduction. The prospective-validation scripts deliberately load their pre-collection frozen modules, rather than the current application package.
