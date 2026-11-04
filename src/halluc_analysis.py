"""Hallucination analysis helpers (notebook 10): load the 716-record predictions of every condition, compute the flags, aggregate.
Run directories are chosen by name pattern; the latest matching directory wins."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_io import load_records, load_snapshot, template_id
from src.halluc import response_report
from src.score import canon_slo

ROOT = Path(__file__).resolve().parents[1]
MODELS = {  # label -> (family, valid-run size tag, invalid-run size tag)
    "Qwen3.5-4B": ("qwen35-4b",), "Qwen3.5-9B": ("qwen35-9b",), "Gemma4-E2B": ("gemma-4-e2b-it",), "Gemma4-E4B": ("gemma-4-e4b-it",)}
SETTINGS = ["zero-shot", "zero-shot + thinking", "few-shot (fs-f2)", "few-shot (fs-f2) + thinking"]
VALID_PREFIX = {"zero-shot": "zs_p_mc_", "zero-shot + thinking": "zs_think_p_mc_", "few-shot (fs-f2)": "fs-f2_", "few-shot (fs-f2) + thinking": "fs-f2_think_",
                "few-shot (fs-f1) ": "fs-f1_", "few-shot (fs-f1) + thinking": "fs-f1_think_"}
INVALID_PREFIX = {"zero-shot": "zs_p_mc_invalid_", "zero-shot + thinking": "zs_think_p_mc_invalid_", "few-shot (fs-f2)": "fs-f2_invalid_",
                  "few-shot (fs-f2) + thinking": "fs-f2_think_invalid_", "few-shot (fs-f1) ": "fs-f1_invalid_", "few-shot (fs-f1) + thinking": "fs-f1_think_invalid_"}


def latest(pattern):
    hits = sorted(glob.glob(str(ROOT / "outputs" / "runs" / pattern)))
    return hits[-1] if hits else None


def run_dirs(model, setting):
    tag = MODELS[model][0]
    return latest(f"{VALID_PREFIX[setting]}{tag}_2*"), latest(f"{INVALID_PREFIX[setting]}{tag}_2*")


def read_predictions(run):
    with open(Path(run) / "predictions.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def condition_table(models=MODELS, settings=SETTINGS):
    rows = []
    for m in models:
        for s in settings:
            v, i = run_dirs(m, s)
            rows.append({"model": m, "setting": s, "valid_run": Path(v).name if v else None, "invalid_run": Path(i).name if i else None})
    return pd.DataFrame(rows)


def compute_all(models=MODELS, settings=SETTINGS):
    """Flags for every response (716 per condition). Returns (responses, faults) DataFrames."""
    snap = load_snapshot()
    recs = {r["index"]: r for r in load_records()}
    resp, fault = [], []
    for m in models:
        for s in settings:
            v, i = run_dirs(m, s)
            if not v or not i:
                continue
            for run in (v, i):
                for p in read_predictions(run):
                    r = recs[p["index"]]
                    rep = response_report(p, r["label"]["slos"], r["meta"]["status"], snap)
                    resp.append({"model": m, "setting": s, "index": p["index"], "status": r["meta"]["status"], "level": r["meta"]["level"],
                                 "source": r["meta"]["source"], "template": template_id(r), "parse_ok": bool(p.get("parse_ok")),
                                 **rep["h"], "n_pred": rep["n_pred"], "n_gold": rep["n_gold"], "omissions": rep["omissions"],
                                 "clean": rep["clean_constraints"], "n_faults": len(rep["faults"])})
                    for f, sub in rep["faults"]:
                        fault.append({"model": m, "setting": s, "index": p["index"], "status": r["meta"]["status"], "level": r["meta"]["level"],
                                      "source": r["meta"]["source"], "flag": f, "subtype": sub, "base": sub.split("+")[0].split(":")[0]})
    return pd.DataFrame(resp), pd.DataFrame(fault)


# ---- definition variants (sub-type groups) ----
RELATION = {"unsupported_extremum_tag", "missing_extremum_tag", "wrong_extremum", "wrong_direction", "boundary_strictness"}
TOLERANT_DROP = {"boundary_strictness", "vacuous_placement", "unit_conversion", "unit_label", "other_scope_real_extremum", "placement_value_variant"}


def variant_flags(resp, fault, variant):
    """Recompute the response-level H1..H3 under a definition variant.
    ours: all sub-types;  paper_narrow: no relation errors (wording of paper section 5.5);  tolerant: ours without conventions/ambiguities."""
    drop = set()
    if variant == "paper_narrow":
        drop = RELATION
    elif variant == "tolerant":
        drop = TOLERANT_DROP
    f = fault[~fault.base.isin(drop)]
    keys = ["model", "setting", "index"]
    flags = f.groupby(keys + ["flag"]).size().unstack(fill_value=0).clip(upper=1)
    out = resp.set_index(keys)[[]].join(flags).fillna(0)
    for k in ("H1", "H2", "H3"):
        if k not in out:
            out[k] = 0
    return out[["H1", "H2", "H3"]].astype(int).reset_index()


def aggregate(flags_df):
    g = flags_df.groupby(["model", "setting"])[["H1", "H2", "H3"]].mean()
    g["H"] = g.sum(axis=1)
    return g


def cluster_bootstrap(resp, per_record_cols, groupby, B=2000, seed=20261020):
    """95% CI of the mean of per-record columns, resampling templates (clusters), for each group."""
    rng = np.random.default_rng(seed)
    out = {}
    for key, g in resp.groupby(groupby):
        tmpl = g.groupby("template")[per_record_cols].agg(["sum", "count"])
        s = tmpl.xs("sum", axis=1, level=1).values
        n = tmpl.xs("count", axis=1, level=1).values[:, 0]
        k = len(tmpl)
        est = s.sum(axis=0) / n.sum()
        boots = np.empty((B, len(per_record_cols)))
        for b in range(B):
            idx = rng.integers(0, k, k)
            boots[b] = s[idx].sum(axis=0) / n[idx].sum()
        out[key] = {c: (est[j], *np.percentile(boots[:, j], [2.5, 97.5])) for j, c in enumerate(per_record_cols)}
    return out


PAPER_TABLE5 = pd.DataFrame(
    [("Zero-shot", .3045, .1955, .1006), ("Few-shot", .1536, .2598, .0014), ("Zero-shot CoT", .3184, .2249, .0852),
     ("Few-shot CoT", .1341, .2612, .0056), ("Intent Engine", .0084, .0768, .0042)], columns=["method", "H1", "H2", "H3"]).set_index("method")
PAPER_TABLE5["H"] = PAPER_TABLE5.sum(axis=1)


def scope_conflict_records():
    """Valid records whose gold lookup value is the GLOBAL extremum although the intent restricts the placement layer (the 'scope errors' of the dataset).
    Returns {index: [metric, ...]}. For these records a model that follows the layer restriction is flagged `other_scope_real_extremum` although it is defensible."""
    from src.halluc import _extrema
    snap = load_snapshot()
    out = {}
    for r in load_records():
        if r["meta"]["status"] != "valid":
            continue
        gold = [canon_slo(s) for s in r["label"]["slos"]]
        layer = next((g[4] for g in gold if g[0] == "placement_location" and g[4] in ("edge", "cloud")), None)
        if not layer:
            continue
        for g in gold:
            if g[3] in ("highest", "lowest") and isinstance(g[4], float):
                ex = _extrema(snap, g[1])
                if layer in ex and abs(g[4] - ex["all"][g[3]]) < 1e-9 and abs(g[4] - ex[layer][g[3]]) > 1e-9:
                    out.setdefault(r["index"], []).append(g[1])
    return out
