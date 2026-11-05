"""RQ5 figures of the paper (Section IV-E): Fig. 6 (change in strict F1 on valid intents and in rejection rate on invalid intents for each model and setup,
C1 to C3 pooled, against the original intents) and Fig. 7 (change in rejection rate per phrasing variant).

Inputs (released analysis results, in results/):
  overview_all_716_by_model_setup.csv         per configuration: F1 and rejection for the original and for C1+C2+C3 pooled (716 intents)
  H-08_per_model_and_setup_all_intents.csv    per configuration and per variant C1, C2, C3
Outputs: figures/fig7_opt_i_change_heatmaps.pdf/.png (Fig. 6) and figures/fig7_opt_v2_invalid_bars.pdf/.png (Fig. 7)
Run from anywhere: python3 rq5_robustness/make_figures.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7, "hatch.linewidth": 0.5})
MODELS = [("Qwen3.5-4B", "Qwen3.5-4B"), ("Qwen3.5-9B", "Qwen3.5-9B"), ("Gemma4-E2B", "Gemma-4-E2B"), ("Gemma4-E4B", "Gemma-4-E4B")]
SETUPS = [("zs3", "zs"), ("fs3", "fs"), ("zs3 CoT", "zs-cot"), ("fs3 CoT", "fs-cot")]
VAR = [("V0", "Original", "#9a9a9a", ""), ("C1", "C1 wrapped", "#2a78d6", "///"),
       ("C2", "C2 restructured", "#eb6834", ".."), ("C3", "C3 noisy list", "#1baf7a", "xx")]  # categorical slots pass the palette check


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---- (i) change per configuration (C1+C2+C3 pooled minus original), two diverging heatmaps ----
ov = pd.read_csv(RES / "overview_all_716_by_model_setup.csv").set_index(["model", "setup"])
dF = np.array([[100 * (ov.loc[(k, s), "F1_rob"] - ov.loc[(k, s), "F1_base"]) for s, _ in SETUPS] for k, _ in MODELS])
dR = np.array([[100 * (ov.loc[(k, s), "rej_rob"] - ov.loc[(k, s), "rej_base"]) for s, _ in SETUPS] for k, _ in MODELS])
lim = 20
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.85), sharey=True)
for ax, M, title in ((axes[0], dF, r"Valid: $\Delta F_1$ (points)"), (axes[1], dR, r"Invalid: $\Delta\rho$ (points)")):
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")  # diverging, neutral midpoint at 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=6, color="white" if abs(M[i, j]) > 12 else "#222222")
    ax.set_xticks(range(4))
    ax.set_xticklabels([s for _, s in SETUPS], family="monospace", fontsize=5.6, rotation=30, ha="right")
    ax.set_title(title, fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_yticks(range(len(MODELS)))
axes[0].set_yticklabels([n for _, n in MODELS])
fig.subplots_adjust(left=0.24, right=0.88, bottom=0.2, top=0.88, wspace=0.08)
cb = fig.colorbar(im, cax=fig.add_axes([0.9, 0.2, 0.02, 0.68]))
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=6)
save(fig, "fig7_opt_i_change_heatmaps")

MK = ["o", "s", "^"]


def _bands(ax, n=4):
    for g in range(n):
        if g % 2 == 0:
            ax.axvspan(g - 0.5, g + 0.5, color="#f3f3f3", zorder=0, lw=0)


metric, base, ylab, tag = "rejection", "rejection base", r"$\Delta\rho$ vs. original (points)", "invalid"
h8 = pd.read_csv(RES / "H-08_per_model_and_setup_all_intents.csv", index_col=0).set_index(["model", "setup"])
# (v-2) diverging bars from zero inside shaded setup bands, one panel per model
fig, axes = plt.subplots(2, 2, figsize=(3.45, 2.7), sharex=True, sharey=True)
for ax, (k, name) in zip(axes.flat, MODELS):
    _bands(ax)
    for j, (v, lab, col, h) in enumerate(VAR[1:]):
        xs = np.arange(4) + (j - 1) * 0.26
        ys = [100 * (h8.loc[(k, s), f"{metric} {v}"] - h8.loc[(k, s), base]) for s, _ in SETUPS]
        ax.bar(xs, ys, 0.24, color=col, hatch=h, edgecolor="white", linewidth=0.3, label=lab, zorder=2)
    ax.axhline(0, color="#666666", lw=0.6, zorder=3)
    ax.set_title(name, fontsize=6.4, pad=2)
    ax.set_xlim(-0.5, 3.5); ax.set_xticks(range(4))
    ax.set_xticklabels([s for _, s in SETUPS], family="monospace", fontsize=5.8)
    ax.tick_params(labelsize=5.8, length=0, axis="x"); clean(ax)
fig.supylabel(ylab, fontsize=6.4, x=0.01)
hs, ls = axes[0, 0].get_legend_handles_labels()
fig.legend(hs, ls, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6.2)
fig.tight_layout(pad=0.3, rect=(0.02, 0, 1, 0.93))
save(fig, "fig7_opt_v2_invalid_bars")

