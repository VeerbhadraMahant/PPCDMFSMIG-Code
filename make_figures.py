"""Build the results table (csv + tex) and Figures A and B from runs/*/metrics.json."""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from common import resolve

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
})


def load_runs(runs_dir):
    runs = []
    for f in sorted(runs_dir.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        if m.get("checkpoints"):
            runs.append((m["n"], f.parent, m))
    runs.sort(key=lambda r: r[0])
    for n in (256, 1024, 4096):
        if n not in [r[0] for r in runs]:
            print(f"WARNING: no evaluated run for N={n}, skipped")
    for n, d, m in runs:
        assert len(m["checkpoints"]) == 8, f"{d}: {len(m['checkpoints'])} evaluated checkpoints, expected 8"
        assert max(e["step"] for e in m["checkpoints"].values()) == m["steps"], f"{d}: final checkpoint is not step {m['steps']}"
    assert len({m["steps"] for _, _, m in runs}) == 1, f"runs differ in step count: {[(n, m['steps']) for n, _, m in runs]}"
    return runs


def series(m, key):
    pts = sorted(m["checkpoints"].values(), key=lambda e: e["step"])
    return (np.array([e["step"] for e in pts]), np.array([e[key] if e[key] is not None else np.nan for e in pts], float),
            [e for e in pts])


def table(runs, out):
    rows = []
    for n, _, m in runs:
        e = max(m["checkpoints"].values(), key=lambda e: e["step"])
        rows.append({"N": n, "step": e["step"], "n_samples": e["n_samples"], "mia_n_per_group": e["mia_n_per_group"],
                     "copy_threshold": e["copy_threshold"],
                     **{k: e[k] for k in ("copy_rate", "copy_rate_lo", "copy_rate_hi", "median_nn", "median_nn_lo",
                                          "median_nn_hi", "copy_rate_095", "copy_rate_095_lo", "copy_rate_095_hi", "auc", "auc_lo", "auc_hi",
                                          "tpr1", "tpr1_lo", "tpr1_hi")},
                     "fid": np.nan if e["fid"] is None else e["fid"]})
    with open(out / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    fmt = lambda r, k: f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"
    lines = ["% Final checkpoint per N. Requires booktabs.",
             "% TPR@1%FPR is coarse at N=256 (about 2.6 non-members at 1% FPR).",
             r"\begin{tabular}{rrcccccr}", r"\toprule",
             r"$N$ & Step & Copy rate [95\% CI] & Median NN sim & Copy rate at 0.95 [95\% CI] & MIA AUC [95\% CI] & TPR@1\%FPR [95\% CI] & FID \\",
             r"\midrule"]
    for r in rows:
        fid = "--" if np.isnan(r["fid"]) else f"{r['fid']:.1f}"
        lines.append(f"{r['N']} & {r['step']} & {fmt(r, 'copy_rate')} & {r['median_nn']:.3f} & {fmt(r, 'copy_rate_095')} & "
                     f"{fmt(r, 'auc')} & {fmt(r, 'tpr1')} & {fid} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table.tex").write_text("\n".join(lines) + "\n")


def figure_a(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2), constrained_layout=True)
    panels = [("copy_rate", "Copy rate", True), ("auc", "MIA AUC", True), ("fid", "FID", False)]
    for ax, (key, label, band) in zip(axes, panels):
        for i, (n, _, m) in enumerate(runs):
            x, y, pts = series(m, key)
            ax.plot(x, y, "-o", color=COLORS[i % 4], ms=2.5, lw=1.1, label=f"$N={n}$")
            if band:
                lo = np.array([e[key + "_lo"] for e in pts]); hi = np.array([e[key + "_hi"] for e in pts])
                ax.fill_between(x, lo, hi, color=COLORS[i % 4], alpha=0.15, lw=0)
        if key == "copy_rate":  # null rate by construction: 1% of non-members exceed the threshold
            ax.axhline(0.01, color="0.4", lw=0.6, ls="--", label="calibrated null (0.01)")
        if key == "auc":
            ax.axhline(0.5, color="0.5", lw=0.6, ls="--")
        ax.set_xlabel("Training step"); ax.set_ylabel(label)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        ax.tick_params(width=0.6, length=2.5)
    axes[0].legend(frameon=False, loc="best")
    fig.savefig(out / "figA.pdf")
    plt.close(fig)


def figure_b(runs, out, figb_n):
    cands = [r for r in runs if (r[1] / "figB.npz").exists() and (figb_n is None or r[0] == figb_n)]
    if not cands:
        print("figB: no figB.npz found, skipped")
        return
    n, d, _ = cands[0]
    z = np.load(d / "figB.npz")
    k = len(z["sims"])
    fig, axes = plt.subplots(2, k, figsize=(6.5, 1.75), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, k)
    for j in range(k):
        for r, img in enumerate((z["gen"][j], z["nn"][j])):
            ax = axes[r, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[1, j].set_xlabel(f"{z['sims'][j]:.3f}", labelpad=2)
    axes[0, 0].set_ylabel("Generated"); axes[1, 0].set_ylabel("Nearest\ntraining")
    fig.savefig(out / "figB.pdf")
    plt.close(fig)
    print(f"figB built from N={n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figb-n", type=int, default=None, help="N for Figure B (default: smallest N available)")
    a = ap.parse_args()
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(resolve(a.runs))
    if not runs:
        raise SystemExit("no runs/*/metrics.json found")
    table(runs, out)
    figure_a(runs, out)
    figure_b(runs, out, a.figb_n)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
