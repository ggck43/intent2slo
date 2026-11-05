"""RQ4 figure of the paper (Section IV-D, Fig. 5): energy per intent against strict F1, one panel per model, four setups, arrows from the
setup without reasoning to the setup with reasoning.

Input (released analysis result, in results/): E-04_energy_and_accuracy_per_cell.csv
  'Wh per intent (1 at a time)' = whole-valid-set energy predicted by the cost model of E-03 (E = a + c x generated tokens) from the token counts of
                                  the accuracy runs, one request at a time (GPU energy, NVML counter)
  'F1'                          = strict micro-F1 of the same accuracy runs
Output: figures/fig6_opt_a_panels.pdf and .png
Run from anywhere: python3 rq4_energy/make_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
d = pd.read_csv(HERE / "results/E-04_energy_and_accuracy_per_cell.csv").set_index(["model", "config"])
MODELS = [("Qwen3.5-4B", "Qwen3.5-4B"), ("Qwen3.5-9B", "Qwen3.5-9B"), ("Gemma4-E2B", "Gemma-4-E2B"),
          ("Gemma4-E4B", "Gemma-4-E4B"), ("Ministral-3-3B", "Ministral-3-3B"), ("Ministral-3-8B", "Ministral-3-8B")]
# panel layout of option (a): one column per family, larger model on top, lighter model below (user, 2026-10-28)
PANELS = [("Qwen3.5-9B", "Qwen3.5-9B"), ("Gemma4-E4B", "Gemma-4-E4B"), ("Ministral-3-8B", "Ministral-3-8B"),
          ("Qwen3.5-4B", "Qwen3.5-4B"), ("Gemma4-E2B", "Gemma-4-E2B"), ("Ministral-3-3B", "Ministral-3-3B")]
SET = {"zs3": "zs", "fs3": "fs", "zs3_cot": "zs-cot", "fs3_cot": "fs-cot"}
STY = {"zs3": ("#2a78d6", "none", "o"), "fs3": ("#eb6834", "none", "s"),
       "zs3_cot": ("#2a78d6", "#2a78d6", "o"), "fs3_cot": ("#eb6834", "#eb6834", "s")}  # 2 colours (palette check: pass) + fill
WH, F1 = "Wh per intent (1 at a time)", "F1"
plt.rcParams.update({"font.family": "serif", "font.size": 7})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---- (a) small multiples: one panel per model, 4 setups, arrows base -> base+cot ----
fig, axes = plt.subplots(2, 3, figsize=(3.45, 2.75), sharex=True, sharey=True)
for ax, (k, name) in zip(axes.flat, PANELS):
    for base, cot, ls in (("zs3", "zs3_cot", (0, (2, 1.5))), ("fs3", "fs3_cot", "-")):
        x0, y0, x1, y1 = d.loc[(k, base), WH], d.loc[(k, base), F1], d.loc[(k, cot), WH], d.loc[(k, cot), F1]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=STY[base][0], lw=0.8, linestyle=ls, shrinkA=3, shrinkB=3))
    for c in SET:
        col, face, mk = STY[c]
        ax.scatter(d.loc[(k, c), WH], d.loc[(k, c), F1], s=16, marker=mk, facecolors=face, edgecolors=col, linewidths=0.9, zorder=3)
    ax.set_title(name, fontsize=6.6, pad=2)
    ax.set_xlim(0, 0.65)
    ax.set_ylim(0.35, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.grid(color="#ececec", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6)
    clean(ax)
fig.supxlabel("GPU energy per intent (Wh), one request at a time", fontsize=6.6, y=0.02)
fig.supylabel(r"Strict micro-$F_1$", fontsize=6.6, x=0.01)
handles = [plt.Line2D([], [], marker=STY[c][2], ls="", markerfacecolor=STY[c][1], markeredgecolor=STY[c][0], label=SET[c]) for c in SET]
leg = fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6.3, handletextpad=0.2)
for t in leg.get_texts():
    t.set_family("monospace")
fig.tight_layout(pad=0.3, rect=(0.02, 0.03, 1, 0.93))
save(fig, "fig6_opt_a_panels")
