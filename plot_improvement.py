"""Render the saved experimental results, without fitting any models."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/grille-lab-matplotlib")
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "diagnostics/improvement"
old = json.loads((ROOT / "report.json").read_text())
new = json.loads((ROOT / "noise_aware/report.json").read_text())
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(bottom=.23, top=.78, wspace=.32)
positions = np.arange(2)
for offset, report, label, color in ((-.18, old, "Original crop + model selection", "#999c9f"),
                                    (.18, new, "Noise-aware crop + model selection", "#1d7458")):
    values = [report["nested_group_evaluation"]["selected"]["forced_direction_correct"],
              report["robustness"]["noise_20db_event"]["forced_direction_correct"]]
    bars = axes[0].bar(positions + offset, values, .36, label=label, color=color)
    axes[0].bar_label(bars, labels=[f"{n}/20" for n in values], padding=4)
axes[0].set_xticks(positions, ["Original recordings", "Added stationary noise"])
axes[0].set_ylim(0, 22)
axes[0].set_yticks(range(0, 21, 5))
axes[0].set_ylabel("Correct direction, when forced to choose")
axes[0].set_title("Direction recognition")
axes[0].legend(loc="upper left", bbox_to_anchor=(-.02, -.13), frameon=False, fontsize=9)

rows = [new["nested_group_evaluation"]["selected"], new["robustness"]["noise_20db_event"], new["robustness"]["speed_0.85"]]
correct = np.array([r["accepted_correct"] for r in rows])
wrong = np.array([r["accepted_swipes"] - r["accepted_correct"] for r in rows])
rejected = 20 - correct - wrong
axes[1].barh(range(3), correct, label="Accepted, correct", color="#1d7458")
axes[1].barh(range(3), wrong, left=correct, label="Accepted, wrong", color="#b64d35")
axes[1].barh(range(3), rejected, left=correct + wrong, label="Abstained", color="#dddddf")
for i, (a, b, c) in enumerate(zip(correct, wrong, rejected)):
    axes[1].text(a / 2, i, str(a), ha="center", va="center", color="white")
    if b:
        axes[1].text(a + b / 2, i, str(b), ha="center", va="center", color="white")
    axes[1].text(a + b + c / 2, i, str(c), ha="center", va="center")
axes[1].set_yticks(range(3), ["Original", "Added noise", "Slowed to 0.85×"])
axes[1].invert_yaxis()
axes[1].set_xticks(range(0, 21, 5))
axes[1].set_xlabel("Swipe clips (20 total)")
axes[1].set_title("Improved pipeline with rejection enabled")
axes[1].legend(loc="upper left", bbox_to_anchor=(-.02, -.13), frameon=False, fontsize=9, ncol=2)
fig.suptitle("Grille Lab: modest direction gain, conservative acceptance", fontsize=17, y=.97)
fig.text(.5, .88, "Model selection stays inside training folds. Eight idle/typing clips produced zero false activations.", ha="center", fontsize=10)
fig.text(.02, .025, "Exploratory reuse of one saturated pilot; no independent validation. Noise: 20 dB active-event SNR.\n"
         "Slowing by resampling also shifts pitch. Only 3/10 up swipes and 8/10 down swipes are accepted on the original clips.", fontsize=9, color="#555555")
fig.savefig(ROOT / "results.png", dpi=160)
print(ROOT / "results.png")
