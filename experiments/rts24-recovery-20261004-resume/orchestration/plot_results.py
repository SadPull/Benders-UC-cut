"""Plot measured incumbent curves, carrying finished runs forward to the common deadline."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
if not summary["complete"] or not summary["numerical_audit_passed"]:
    raise SystemExit("finish and audit the batch before plotting")
colors = {"A":"#64748b", "B":"#8b5cf6", "C":"#0284c7", "D":"#e11d48", "E":"#16a34a"}
reference = summary["reference_cost"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
for row in summary["runs"]:
    curve = row["curve"]
    if not curve:
        continue
    xs = [p["seconds"] for p in curve]+[1800]
    ys = [100*(p["objective"]/reference-1) for p in curve]
    ys.append(ys[-1])
    for axis in axes:
        axis.step(xs, ys, where="post", color=colors[row["group"]],
                  linestyle="-" if row["recovery"] else "--", alpha=.8,
                  label=f"{row['group']}/{row['seed']}"+ (" *" if row["recovery"] else ""))
for axis, top in zip(axes, (40, 1.5)):
    axis.set(xlim=(0,1800), ylim=(0,top), xlabel="Elapsed wall time (s)", ylabel="Difference from reference (%)")
    axis.grid(alpha=.2)
axes[0].set_title("All validated incumbent costs")
axes[1].set_title("Near-reference region")
axes[1].axhline(.5, color="#475569", linewidth=.8, alpha=.5)
axes[0].legend(fontsize=8, ncol=3, loc="upper right")
fig.suptitle("RTS-24 recovery: * prespecified reruns; dashed curves retained from earlier dispatch")
fig.savefig(root / "quality-curves.png", dpi=180)
fig.savefig(root / "quality-curves.svg")
plt.close(fig)
