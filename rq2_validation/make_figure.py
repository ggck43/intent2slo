"""RQ2 figure of the paper (Section IV-B, Fig. 3): rejection rate of every model and setup on the 195 invalid intents.

Input (released analysis result, in results/): I-01_rejection_rate_per_category.csv (paper-style rate rho, overall column).
Output: figures/fig3_opt_i_heatmap.pdf and .png
Run from anywhere: python3 rq2_validation/make_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
MODELS = [("Qwen3.5-4B", "Qwen3.5-4B"), ("Qwen3.5-9B", "Qwen3.5-9B"), ("Gemma4-E2B", "Gemma-4-E2B"), ("Gemma4-E4B", "Gemma-4-E4B"),
          ("Ministral-3-3B", "Ministral-3-3B"), ("Ministral-3-8B", "Ministral-3-8B"), ("Granite-4.1-3B", "Granite-4.1-3B"),
          ("Granite-4.1-8B", "Granite-4.1-8B"), ("Llama-3.2-3B", "Llama-3.2-3B"), ("Llama-3.1-8B", "Llama-3.1-8B")]
SETUPS = [("zs3", "zs"), ("fs3", "fs"), ("zs3 CoT", "zs-cot"), ("fs3 CoT", "fs-cot")]
plt.rcParams.update({"font.family": "serif", "font.size": 7, "hatch.linewidth": 0.5})

rej = pd.read_csv(RES / "I-01_rejection_rate_per_category.csv").set_index(["model", "prompt"])


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


# ---- (i) heatmap: overall rejection rate, models x setups ----
M = np.array([[rej.loc[(k, p), "overall"] for p, _ in SETUPS] for k, _ in MODELS])
fig, ax = plt.subplots(figsize=(3.45, 2.6))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.3, color="white" if M[i, j] > 0.6 else "#222222")
ax.set_xticks(range(4))
ax.set_xticklabels([s for _, s in SETUPS], family="monospace")
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels([n for _, n in MODELS])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
cb.set_label(r"Rejection rate $\rho$ (195 invalid intents)")
cb.outline.set_visible(False)
fig.tight_layout(pad=0.2)
save(fig, "fig3_opt_i_heatmap")
