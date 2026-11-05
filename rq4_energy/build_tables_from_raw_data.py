"""Rebuild the RQ4 result tables E-01 to E-05 from the raw energy measurements (needs the camera-ready archive).

The code of each block is the code of the authors' analysis notebook, with plotting and display calls removed and the output reduced to CSV files.

Inputs
  --raw   directory of the measurement run: <raw>/<model>/{requests.jsonl, phases.jsonl, power_trace.csv, env.json}   (scripts/energy_probe2.py)
  outputs/runs/                      the accuracy runs (F1, exact match, token counts) and the invalid-intent runs, used by E-04
  outputs/energy/ember_grid_factors_2025.csv   grid carbon factors, used by E-05 (also released in rq4_energy/results/)
Output
  --out   one CSV per table, named E-nn_<table>.csv (the files in rq4_energy/results/ were made by this code)
Run: python rq4_energy/build_tables_from_raw_data.py --raw outputs/energy/run_20261025 --out outputs/energyAnalysis_rebuilt
Refuses to write into a folder that already holds CSV files.
"""
import argparse
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.valid_analysis as VA  # noqa: E402
from src.energy_analysis import *  # noqa: E402,F401,F403
from src.energy_analysis import MODELS as EM, FAMILY as EF  # noqa: E402,F401

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="outputs/energy/run_20261025")
ap.add_argument("--out", default="outputs/energyAnalysis_rebuilt")
ARGS = ap.parse_args()
RUN = Path(ARGS.raw)
OUTDIR = Path(ARGS.out)
if OUTDIR.exists() and any(OUTDIR.glob("E-*/*.csv")):
    sys.exit(f"{OUTDIR} already holds results; choose a new --out folder")
D = load_run(RUN)
R, PH = D["req"], D["phases"]
MN = [m for m in EM if m in set(R.model)]
CF = [c for c in CFGS if c in set(R.config)]


def display(x):  # the notebook displayed tables; the script only writes them
    pass


def done(eid, title, kind, tables):
    """Write each table as <out>/<eid>/<eid>_<slug>.csv"""
    d = OUTDIR / eid
    d.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        df.to_csv(d / f"{eid}_{slug}.csv")
    print(eid, title, "->", d)


print("run:", RUN, "| models:", MN, "| configs:", CF, "| requests:", len(R))

# ===== E-01 =====
v = R[R.kind == "n1_valid"]
g = v.groupby(["model", "config"])
e01 = pd.DataFrame({"requests": g.size(), "mean Wh": g.energy_J.mean() / 3600, "median Wh": g.energy_J.median() / 3600, "p10 Wh": g.energy_J.quantile(.1) / 3600, "p90 Wh": g.energy_J.quantile(.9) / 3600,
                    "mean latency (s)": g.latency_s.mean(), "mean generated tokens": g.completion_tokens.mean(), "J per generated token": g.energy_J.sum() / g.completion_tokens.sum(),
                    "mean uncached prompt tokens": g.uncached_prompt_tokens.mean(), "mean cached share": g.cached_tokens.sum() / g.prompt_tokens.sum()})
e01 = e01.reindex(pd.MultiIndex.from_product([MN, CF], names=["model", "config"])).dropna(how="all"); display(e01.round(3))
iv = R[R.kind == "n1_invalid"].groupby(["model", "config"]); e01b = pd.DataFrame({"requests": iv.size(), "mean Wh (invalid intents)": iv.energy_J.mean() / 3600, "mean generated tokens": iv.completion_tokens.mean(), "J per generated token": iv.energy_J.sum() / iv.completion_tokens.sum()})
e01b = e01b.reindex(e01.index).dropna(how="all"); print("Invalid intents (rejections are short answers):"); display(e01b.round(3))
# noise check: the same 20 intents in three passes
first = v.groupby(["model", "config"])["index"].apply(lambda s: list(s)[:20])
nz = []
for (m, c), idx in first.items():
    s = R[(R.model == m) & (R.config == c) & R.kind.isin(["n1_valid", "n1_rep1", "n1_rep2"]) & R["index"].isin(idx)]
    p = s.pivot_table(index="index", columns="kind", values="energy_J"); nz.append({"model": m, "config": c, "mean CV of one intent over 3 passes": float((p.std(axis=1) / p.mean(axis=1)).mean()), "max CV": float((p.std(axis=1) / p.mean(axis=1)).max()), "pass-to-pass difference of the mean": float(p.mean().max() / p.mean().min() - 1)})
nz = pd.DataFrame(nz).set_index(["model", "config"]); print("Noise check (20 intents, three passes):"); display(nz.round(4))
done("E-01", "Energy, latency and J per generated token, one request at a time", "N", {"valid intents one at a time": e01, "invalid intents": e01b, "noise check": nz})

# ===== E-02 =====
rows = []
cl = PH[PH.phase.str.endswith(("_c16", "_c64"))]
for _, w in cl.iterrows():
    m, c = w["model_name"], w["config"]; rq = R[(R.model == m) & (R.phase == w["phase"])]
    if len(rq) == 0: continue
    s = steady_state(D["trace"][m], rq, w)
    rows.append({"model": m, "config": c, "N": int(w["concurrency"]), "requests": len(rq), "whole-window J per intent": s["whole_window_J"], "steady-state J per intent": s["steady_J"], "steady-state requests": s["n_steady"],
                 "counter / trace scale": s["scale"], "throughput (intents/s)": w["throughput_req_per_s"], "throughput (tokens/s)": w["throughput_tok_per_s"], "p95 latency (s)": w["latency_p95_s"], "mean power (W)": w["mean_power_W"], "contaminated": bool(w["contaminated"]),
                 "mean generated tokens": w["mean_completion_tokens"], "window (s)": w["window_s"]})
e02 = pd.DataFrame(rows).set_index(["model", "config", "N"]); e02 = e02.reindex([(m, c, n) for m in MN for c in CF for n in (16, 64) if (m, c, n) in e02.index]); display(e02.round(3))
n1 = R[R.kind == "n1_valid"].groupby(["model", "config"]).energy_J.mean()
sv = e02["steady-state J per intent"].fillna(e02["whole-window J per intent"]).unstack("N").rename(columns={16: "J per intent at 16", 64: "J per intent at 64"}); sv.insert(0, "J per intent at 1", n1)
sv["saving 1 -> 16"] = sv["J per intent at 1"] / sv["J per intent at 16"]; sv["saving 1 -> 64"] = sv["J per intent at 1"] / sv["J per intent at 64"]; sv = sv.reindex([(m, c) for m in MN for c in CF if (m, c) in sv.index]); print("Energy per intent by load and the batching saving:"); display(sv.round(2))
done("E-02", "Energy per intent under load", "N", {"closed-loop windows": e02, "energy per intent and batching saving": sv})

# ===== E-03 =====
fit = []
for m in MN:
    d = R[(R.model == m) & R.kind.isin(["n1_valid", "n1_invalid"]) & R.energy_J.notna()]
    coef, r2 = fit_cost2(d); cv = cv_cost2(d)
    fit.append({"model": m, "requests": len(d), "a: J per request independent of the answer length": coef[0], "c: J per generated token": coef[1], "R2": r2, "CV R2": cv["cv_r2"], "MAPE (one request)": cv["mape"], "error of summed energy": cv["sum_error"]})
e03a = pd.DataFrame(fit).set_index("model"); display(e03a.round(4))
# energy share of generation vs prefill, and prompt effects, per configuration
sh = []
for (m, c), d in R[R.kind == "n1_valid"].groupby(["model", "config"]):
    co = e03a.loc[m]; gen = co["c: J per generated token"] * d.completion_tokens.mean(); fx = co["a: J per request independent of the answer length"]
    sh.append({"model": m, "config": c, "generation share": gen / (gen + fx), "answer-independent share (prefill + fixed)": fx / (gen + fx), "mean generated tokens": d.completion_tokens.mean(), "mean uncached prompt tokens": d.uncached_prompt_tokens.mean()})
e03b = pd.DataFrame(sh).set_index(["model", "config"]); display(e03b.round(3))
mean_e = R[R.kind == "n1_valid"].groupby(["model", "config"]).energy_J.mean(); mean_t = R[R.kind == "n1_valid"].groupby(["model", "config"]).completion_tokens.mean(); mean_u = R[R.kind == "n1_valid"].groupby(["model", "config"]).uncached_prompt_tokens.mean()
ov = []
for m in MN:
    for a_, b_, lab in (("zs3", "zs3_cot", "CoT on zero-shot"), ("fs3", "fs3_cot", "CoT on few-shot"), ("zs3", "fs3", "few-shot examples on zero-shot"), ("zs3_cot", "fs3_cot", "examples on CoT")):
        if (m, a_) in mean_e.index and (m, b_) in mean_e.index:
            ov.append({"model": m, "comparison": lab, "energy ratio": mean_e[(m, b_)] / mean_e[(m, a_)], "generated-token ratio": mean_t[(m, b_)] / mean_t[(m, a_)], "uncached-prompt-token ratio": mean_u[(m, b_)] / mean_u[(m, a_)]})
e03c = pd.DataFrame(ov).set_index(["model", "comparison"]); print("Overheads (measured, one request at a time):"); display(e03c.round(2))
done("E-03", "Cost model: where the joules go", "N", {"cost model per model": e03a, "energy shares": e03b, "overheads of CoT and examples": e03c})

# ===== E-04 =====
va = VA.load_all()
pop = {}
for (m, c), d in R[R.kind == "n1_valid"].groupby(["model", "config"]):
    cached = d.cached_tokens.median(); co = e03a.loc[m]; preds = va[(m, CFG_NAME[c])]["preds"]
    pt = np.array([p["prompt_tokens"] for p in preds.values()], float); ct = np.array([p["completion_tokens"] for p in preds.values()], float)
    j = co["a: J per request independent of the answer length"] + co["c: J per generated token"] * ct
    pop[(m, c)] = {"predicted J per intent, whole valid set": j.mean(), "measured J per intent, 60-intent sample": d.energy_J.mean(), "mean generated tokens, whole set": ct.mean(), "mean generated tokens, sample": d.completion_tokens.mean()}
pp = pd.DataFrame(pop).T; pp.index.names = ["model", "config"]; pp["sample vs whole set"] = pp["measured J per intent, 60-intent sample"] / pp["predicted J per intent, whole valid set"] - 1; display(pp.round(3))
inv = R[R.kind == "n1_invalid"].groupby(["model", "config"]).energy_J.mean()
ia = __import__("src.invalid_analysis", fromlist=["x"]); IRj = ia.load_invalid()
acc = []
for (m, c) in pp.index:
    s = VA.summarize(va[(m, CFG_NAME[c])]["rows"]); rj = sum(x["outcome"] != "accepted" for x in IRj[(m, CFG_NAME[c])]["rows"]) / 195
    e1 = pp.loc[(m, c), "predicted J per intent, whole valid set"] / 3600; e64 = e02["steady-state J per intent"].get((m, c, 64), np.nan) / 3600
    acc.append({"model": m, "config": c, "Wh per intent (1 at a time)": e1, "Wh per intent (64 concurrent)": e64, "F1": s["F1"], "exact match": s["EM"], "Wh per exact-correct intent (1 at a time)": e1 / s["EM"], "Wh per exact-correct intent (64 concurrent)": e64 / s["EM"],
                "Wh per invalid intent (1 at a time)": inv.get((m, c), np.nan) / 3600, "rejection rate": rj})
e04 = pd.DataFrame(acc).set_index(["model", "config"]); display(e04.round(4))
pts = [(r["Wh per intent (1 at a time)"], r["F1"]) for _, r in e04.iterrows()]; front = pareto_front(pts)
print("Pareto-optimal (1 at a time, energy vs F1):", [e04.index[i] for i in front])
pts64 = [(r["Wh per intent (64 concurrent)"], r["F1"]) for _, r in e04.iterrows()]; front64 = [i for i in pareto_front([p for p in pts64 if not np.isnan(p[0])])]
ok64 = [i for i, p in enumerate(pts64) if not np.isnan(p[0])]; fr64 = [ok64[i] for i in pareto_front([pts64[i] for i in ok64])]; print("Pareto-optimal (64 concurrent, energy vs F1):", [e04.index[i] for i in fr64])
price = []
for m in MN:
    for a_, b_ in (("zs3", "zs3_cot"), ("fs3", "fs3_cot")):
        if (m, a_) in e04.index and (m, b_) in e04.index:
            dE = e04.loc[(m, b_), "Wh per intent (1 at a time)"] - e04.loc[(m, a_), "Wh per intent (1 at a time)"]; dF = (e04.loc[(m, b_), "F1"] - e04.loc[(m, a_), "F1"]) * 100
            price.append({"model": m, "step": f"{a_} -> {b_}", "extra Wh per intent": dE, "F1 points gained": dF, "extra Wh per F1 point": dE / dF if dF > 0 else np.nan})
e04b = pd.DataFrame(price).set_index(["model", "step"]); print("Energy price of CoT:"); display(e04b.round(4))
done("E-04", "Energy against accuracy", "N", {"energy and accuracy per cell": e04, "predicted whole-set energy": pp, "price of CoT": e04b})

# ===== E-05 =====
idle_load = PH[PH.phase.isin(["idle_loaded", "idle_loaded_after"])].groupby("model_name")["power_W"].mean()
cap = []
for _, w in cl.iterrows():
    m, c = w["model_name"], w["config"]
    if not len(R[(R.model == m) & (R.phase == w["phase"])]): continue
    marg = (w["mean_power_W"] - idle_load[m]) * w["window_s"] / w["requests"]
    cap.append({"model": m, "config": c, "N": int(w["concurrency"]), "throughput (intents/s)": w["throughput_req_per_s"], "p95 latency (s)": w["latency_p95_s"], "marginal J per intent": marg, "mean power (W)": w["mean_power_W"], "idle power (W)": idle_load[m]})
for (m, c), d in R[R.kind == "n1_valid"].groupby(["model", "config"]):
    w = PH[(PH.phase == f"{c}_n1_valid") & (PH.model_name == m)].iloc[0]; pw = w["window_energy_J"] / w["window_s"]
    cap.append({"model": m, "config": c, "N": 1, "throughput (intents/s)": len(d) / w["window_s"], "p95 latency (s)": d.latency_s.quantile(.95), "marginal J per intent": (pw - idle_load[m]) * w["window_s"] / len(d), "mean power (W)": pw, "idle power (W)": idle_load[m]})
cap = pd.DataFrame(cap).set_index(["model", "config", "N"]).sort_index(); display(cap.round(3))
L, UTIL, DAYS = 30.0, 0.5, 86400.0
def project(m, c, D, L=L):
    opts = cap.loc[(m, c)]; ok = opts[opts["p95 latency (s)"] <= L]
    if ok.empty: return None
    best = ok["throughput (intents/s)"].idxmax(); r = ok.loc[best]
    G = max(1, math.ceil(D / (r["throughput (intents/s)"] * DAYS * UTIL))); idle = G * r["idle power (W)"] * 24 / 1000; work = D * r["marginal J per intent"] / 3.6e6
    return {"N used": best, "GPUs": G, "idle kWh/day": idle, "work kWh/day": work, "GPU kWh/day": idle + work, "idle share": idle / (idle + work), "Wh per intent as served": (idle + work) * 1000 / D}
rows = []
for m in MN:
    for c in CF:
        for D_ in (1_000, 10_000, 100_000):
            p = project(m, c, D_)
            if p: rows.append({"model": m, "config": c, "intents per day": D_, **p})
e05 = pd.DataFrame(rows).set_index(["model", "config", "intents per day"]); display(e05.round(3))
rows60 = []
for m in MN:
    for c in CF:
        for D_ in (10_000, 100_000):
            p = project(m, c, D_, L=60.0)
            if p: rows60.append({"model": m, "config": c, "intents per day": D_, **p})
e05L = pd.DataFrame(rows60).set_index(["model", "config", "intents per day"]); print("Sensitivity: latency bound 60 s instead of 30 s (p95):"); display(e05L[["N used", "GPUs", "GPU kWh/day", "idle share", "Wh per intent as served"]].round(3))
sc = []
for PUE in (1.09, 1.56, 1.80):
    GRID = pd.read_csv(ROOT / "outputs/energy/ember_grid_factors_2025.csv").set_index("Area")["gCO2/kWh 2025"] / 1000
    for G_, name in ((GRID["France"], "France (Ember 2025)"), (GRID["EU"], "EU (Ember 2025)"), (GRID["United States of America"], "United States (Ember 2025)"), (GRID["World"], "World (Ember 2025)"), (GRID["Australia"], "Australia (Ember 2025)"), (GRID["India"], "India (Ember 2025)")):
        e = e05.xs(10_000, level="intents per day")["GPU kWh/day"] * PUE; sc.append({"PUE": PUE, "grid": name, "kg CO2e per kWh": G_, "min kg CO2e/day (10k intents)": float((e * G_).min()), "max kg CO2e/day (10k intents)": float((e * G_).max()),
                "g CO2e per intent, min": float((e * G_).min() / 10), "g CO2e per intent, max": float((e * G_).max() / 10)})
e05b = pd.DataFrame(sc).set_index(["PUE", "grid"]); print("Operational CO2e at 10,000 intents per day over the 24 configurations (min and max):"); display(e05b.round(3))
done("E-05", "Deployment projection and operational CO2e", "N", {"capacity and marginal energy": cap, "daily projection": e05, "projection with a 60 s latency bound": e05L, "CO2e scenarios at 10k intents per day": e05b})

