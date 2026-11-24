"""Figure C: copy rate vs guidance scale w for the E3 models (c1024, c4096), reading cfg_metrics.json directly."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
})
COLORS = ["#0072B2", "#D55E00"]  # Okabe-Ito, as make_figures.py
root = Path(__file__).resolve().parent
out = root / "results_e3" / "figC.pdf"

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2), constrained_layout=True)
for ax, run, n, c in zip(axes, ("c1024", "c4096"), (1024, 4096), COLORS):
    ws = json.loads((root / "results_e3" / run / "cfg_metrics.json").read_text())["w"]
    rows = sorted(ws.values(), key=lambda r: r["w"])
    x = list(range(len(rows)))  # evenly spaced ticks, w values are unevenly spaced
    y = [r["copy_rate"] for r in rows]
    ax.plot(x, y, "-o", color=c, ms=2.5, lw=1.1)
    ax.fill_between(x, [r["copy_rate_lo"] for r in rows], [r["copy_rate_hi"] for r in rows], color=c, alpha=0.15, lw=0)
    ax.axhline(0.01, color="0.4", lw=0.6, ls="--")
    ax.set_xticks(x, [f"{r['w']:g}" for r in rows])
    ax.set_xlabel("Guidance scale $w$"); ax.set_ylabel("Copy rate")
    ax.set_title(f"$N={n}$")
    ax.tick_params(width=0.6, length=2.5)
fig.savefig(out)
print("wrote", out)
