"""Helpers for notebook 16 (analyses of the valid intents, numbered V-nn, for the manuscript).
Inputs are the stored predictions of the 40 runs (10 models x 4 baselines zs3, fs3, zs3 CoT, fs3 CoT); nothing is generated here."""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import per_metric
from src.score import canon_list, canon_slo, load_predictions, prf, record_counts
from src.zs3_report import find_run, records, scope_conflict_indices, summarize, taxonomy, valid_rows  # noqa: F401
from src.fs3_report import lookup_outcomes  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "validAnalysis"

# small model first inside each family; Llama is a normal family here (user decision 2026-10-25)
MODELS = {"Qwen3.5-4B": "qwen35-4b", "Qwen3.5-9B": "qwen35-9b", "Gemma4-E2B": "gemma-4-e2b-it", "Gemma4-E4B": "gemma-4-e4b-it",
          "Ministral-3-3B": "ministral-3-3b-instruct-2512", "Ministral-3-8B": "ministral-3-8b-instruct-2512",
          "Granite-4.1-3B": "granite-41-3b", "Granite-4.1-8B": "granite-41-8b",
          "Llama-3.2-3B": "llama-32-3b-instruct", "Llama-3.1-8B": "llama-31-8b-instruct"}
FAMILY = {m: m.split("-")[0].split("3.")[0] if False else m.rsplit("-", 1)[0].rstrip("0123456789.-") for m in MODELS}
FAMILY = {"Qwen3.5-4B": "Qwen", "Qwen3.5-9B": "Qwen", "Gemma4-E2B": "Gemma", "Gemma4-E4B": "Gemma", "Ministral-3-3B": "Ministral",
          "Ministral-3-8B": "Ministral", "Granite-4.1-3B": "Granite", "Granite-4.1-8B": "Granite", "Llama-3.2-3B": "Llama", "Llama-3.1-8B": "Llama"}
SMALL = {"Qwen": "Qwen3.5-4B", "Gemma": "Gemma4-E2B", "Ministral": "Ministral-3-3B", "Granite": "Granite-4.1-3B", "Llama": "Llama-3.2-3B"}
LARGE = {"Qwen": "Qwen3.5-9B", "Gemma": "Gemma4-E4B", "Ministral": "Ministral-3-8B", "Granite": "Granite-4.1-8B", "Llama": "Llama-3.1-8B"}
# paper order of the four baselines: ZS, FS, ZS-CoT, FS-CoT -> ours zs3, fs3, zs3 CoT, fs3 CoT
PROMPTS = ["zs3", "fs3", "zs3 CoT", "fs3 CoT"]
PAPER_NAME = {"zs3": "ZS", "fs3": "FS", "zs3 CoT": "ZS-CoT", "fs3 CoT": "FS-CoT"}
PATTERN = {"zs3": "zs3_{m}_2*", "fs3": "fs3_{m}_2*", "zs3 CoT": "zs3_cot_{m}_2*", "fs3 CoT": "fs3_cot_{m}_2*"}
PROMPT_COLOR = {"zs3": "#8e44ad", "fs3": "#c0392b", "zs3 CoT": "#c39bd3", "fs3 CoT": "#e6a23c"}

# paper Table 7 (exact), Real / Synthetic / Total
PAPER_T7 = {
    "GPT-4.1 mini": {"ZS": (.296, .345, .331), "FS": (.711, .768, .751), "ZS-CoT": (.555, .598, .585), "FS-CoT": (.716, .775, .758), "Intent Engine": (.993, .920, .941)},
    "Claude Sonnet 4.5": {"ZS": (.296, .348, .333), "FS": (.729, .777, .763), "ZS-CoT": (.729, .822, .794), "FS-CoT": (.737, .795, .778), "Intent Engine": (.878, .867, .870)},
    "DeepSeek V4-Flash": {"ZS": (.294, .344, .329), "FS": (.715, .776, .758), "ZS-CoT": (.731, .799, .779), "FS-CoT": (.711, .774, .756), "Intent Engine": (.869, .916, .903)}}
# paper Fig 5 chart readings (EM, Jaccard), about +-0.03
PAPER_F5 = {"GPT-4.1 mini": {"ZS": (.11, .20), "FS": (.50, .60), "ZS-CoT": (.30, .41), "FS-CoT": (.51, .61), "Intent Engine": (.85, .89)},
            "Claude Sonnet 4.5": {"ZS": (.11, .20), "FS": (.52, .62), "ZS-CoT": (.57, .66), "FS-CoT": (.54, .64), "Intent Engine": (.70, .77)},
            "DeepSeek V4-Flash": {"ZS": (.11, .20), "FS": (.51, .61), "ZS-CoT": (.54, .64), "FS-CoT": (.51, .61), "Intent Engine": (.77, .82)}}

ALLOWED_METRICS = {"cpu_capacity", "memory_capacity", "storage_capacity", "cpu_utilization", "memory_utilization", "storage_utilization",
                   "bandwidth", "port_utilization", None}
ALLOWED_TYPES = {"static_compute_resource", "dynamic_compute_resource", "node_network_resource", "placement_location"}
TYPE_OF = {"cpu_capacity": "static_compute_resource", "memory_capacity": "static_compute_resource", "storage_capacity": "static_compute_resource",
           "cpu_utilization": "dynamic_compute_resource", "memory_utilization": "dynamic_compute_resource", "storage_utilization": "dynamic_compute_resource",
           "bandwidth": "node_network_resource", "port_utilization": "node_network_resource"}
UNIT_OF = {"cpu_capacity": "millicores", "memory_capacity": "bytes", "storage_capacity": "bytes", "cpu_utilization": "percent",
           "memory_utilization": "percent", "storage_utilization": "percent", "port_utilization": "percent", "bandwidth": "Mbps"}


def load_all():
    """{(model, prompt): {'run': Path, 'rows': per-record rows, 'preds': predictions}}"""
    out = {}
    for m, tag in MODELS.items():
        for p in PROMPTS:
            d = find_run(PATTERN[p].format(m=tag))
            if d is None or not (d / "predictions.jsonl").exists():
                raise FileNotFoundError(f"missing run for {m} / {p}")
            out[(m, p)] = {"run": d, "rows": valid_rows(d), "preds": load_predictions(d / "predictions.jsonl")}
    return out


def f1_of(rows):
    return summarize(rows)["F1"] if rows else float("nan")


def rows_where(rows, key, value):
    return [r for r in rows if r[key] == value]


# ---------- V-08: explicit vs implicit ----------
def split_counts(rec, pred_slos, parse_ok, which):
    """tp/fp/fn on explicit (no highest/lowest; placement counted as explicit) or implicit (highest/lowest) constraints only"""
    gold = canon_list(rec["label"]["slos"])
    pred = canon_list(pred_slos) if parse_ok else []
    keep = (lambda t: t[3] not in ("highest", "lowest")) if which == "explicit" else (lambda t: t[3] in ("highest", "lowest"))
    return record_counts([t for t in pred if keep(t)], [t for t in gold if keep(t)])


def explicit_implicit(runs):
    recs = {r["index"]: r for r in records() if r["meta"]["status"] == "valid"}
    rows = []
    for (m, p), d in runs.items():
        for which in ("explicit", "implicit"):
            tp = fp = fn = 0
            for i, pr in d["preds"].items():
                if i not in recs:
                    continue
                c = split_counts(recs[i], (pr.get("parsed") or {}).get("slos") if pr.get("parse_ok") and isinstance(pr.get("parsed"), dict) else [], bool(pr.get("parse_ok") and isinstance(pr.get("parsed"), dict)), which)
                tp += c["tp"]; fp += c["fp"]; fn += c["fn"]
            rows.append({"model": m, "prompt": p, "scope": which, "tp": tp, "fp": fp, "fn": fn, "F1": prf(tp, fp, fn)["f1"], "gold": tp + fn})
    return pd.DataFrame(rows)


# ---------- V-07: per SLO kind ----------
def per_kind(runs):
    """F1 per SLO kind. Kinds are the 8 canonical metrics plus placement; every other (type, metric) pair (invented metric names, wrong types) is pooled into 'other (not in the schema)'."""
    recs = [r for r in records() if r["meta"]["status"] == "valid"]
    canon = {"cpu_capacity", "memory_capacity", "storage_capacity", "cpu_utilization", "memory_utilization", "storage_utilization", "port_utilization", "bandwidth"}
    rows = []
    for (m, p), d in runs.items():
        df = per_metric(recs, d["preds"]).reset_index()
        df["kind"] = [("placement" if t == "placement_location" and mm in ("-", None) else mm if (mm in canon and t == TYPE_OF.get(mm)) else "other (not in the schema)") for t, mm in zip(df["type"], df["metric"])]
        g = df.groupby("kind")[["tp", "fp", "fn"]].sum().reset_index()
        for _, r in g.iterrows():
            rows.append({"model": m, "prompt": p, "kind": r["kind"], "tp": int(r["tp"]), "fp": int(r["fp"]), "fn": int(r["fn"]), "gold": int(r["tp"] + r["fn"]),
                         "F1": prf(int(r["tp"]), int(r["fp"]), int(r["fn"]))["f1"]})
    return pd.DataFrame(rows)


# ---------- V-13: Llama diagnosis ----------
def _alias_metric(metric):
    if not isinstance(metric, str):
        return metric
    m = metric.split(".")[-1].strip().lower()
    m = re.sub(r"_(util|utilisation|utilization)(_percent|_pct)?$", "_utilization", m)
    m = re.sub(r"_(cap|capacity)(_millicores|_bytes)?$", "_capacity", m)
    m = {"cpu": "cpu_utilization", "memory_util": "memory_utilization"}.get(m, m)
    return m


def lenient_canon(slo):
    """Diagnostic only: repair surface form (metric prefix / alias, type from metric, cmp symbols, unit symbol, placement written as a metric)."""
    if not isinstance(slo, dict):
        return canon_slo(slo)
    s = json.loads(json.dumps(slo))
    t = s.get("target") if isinstance(s.get("target"), dict) else {}
    s["target"] = t
    metric = _alias_metric(s.get("metric"))
    val = t.get("value")
    if s.get("type") == "placement_location" or metric in ("layer", "placement", "placement_location") or (isinstance(val, str) and val.lower() in ("edge", "cloud", "both", "any") and metric not in TYPE_OF):
        return ("placement_location", None, "set", None, str(val).lower(), None)
    if metric in TYPE_OF:
        s["type"] = TYPE_OF[metric]
        s["metric"] = metric
        if s.get("unit") in ("%", "pct", "percentage"):
            s["unit"] = "percent"
    cmp_ = t.get("cmp")
    t["cmp"] = {">": ">=", "<": "<=", "≥": ">=", "≤": "<="}.get(cmp_, cmp_)
    return canon_slo(s)


def lenient_f1(rows_recs, preds):
    tp = fp = fn = 0
    for rec in rows_recs:
        pr = preds.get(rec["index"])
        ok = bool(pr and pr.get("parse_ok") and isinstance(pr.get("parsed"), dict))
        pred = [lenient_canon(s) for s in (pr["parsed"].get("slos") or [])] if ok else []
        c = record_counts(pred, canon_list(rec["label"]["slos"]))
        tp += c["tp"]; fp += c["fp"]; fn += c["fn"]
    return prf(tp, fp, fn)["f1"]


def schema_validity(preds):
    """share of predicted SLOs that break one rule of the allowed schema (types, metrics, cmp/op symbols, units)"""
    n = bad = 0
    kinds = Counter()
    for pr in preds.values():
        if not (pr.get("parse_ok") and isinstance(pr.get("parsed"), dict)):
            continue
        for s in pr["parsed"].get("slos") or []:
            n += 1
            if not isinstance(s, dict):
                bad += 1; kinds["not an object"] += 1; continue
            t = s.get("target") if isinstance(s.get("target"), dict) else {}
            problems = []
            if s.get("type") not in ALLOWED_TYPES: problems.append("type outside the allowed list")
            if s.get("metric") not in ALLOWED_METRICS: problems.append("metric name outside the allowed list")
            if t.get("cmp") not in ("<=", ">=", None): problems.append("cmp symbol outside <=, >=, null")
            if t.get("op") not in ("highest", "lowest", "set", None): problems.append("op outside highest, lowest, set, null")
            m = s.get("metric")
            if s.get("type") != "placement_location" and m in UNIT_OF and s.get("unit") != UNIT_OF[m]: problems.append("unit not the canonical unit")
            if s.get("type") == "placement_location" and (m is not None): problems.append("placement with a metric name")
            for pbl in problems: kinds[pbl] += 1
            bad += bool(problems)
    return {"SLOs": n, "SLOs with a schema problem": bad, "share": bad / n if n else float("nan"), **{k: v for k, v in kinds.items()}}


def field_agreement(rows_recs, preds):
    """after the lenient repair, pair predicted and gold SLOs of the same (type, metric) and count how often each field is equal"""
    acc = Counter(); pairs = 0
    for rec in rows_recs:
        pr = preds.get(rec["index"])
        if not (pr and pr.get("parse_ok") and isinstance(pr.get("parsed"), dict)):
            continue
        pred = [lenient_canon(s) for s in (pr["parsed"].get("slos") or [])]
        gold = canon_list(rec["label"]["slos"])
        left = list(pred)
        for g in gold:
            j = next((k for k, p in enumerate(left) if (p[0], p[1]) == (g[0], g[1])), None)
            if j is None:
                continue
            p = left.pop(j); pairs += 1
            for name, a, b in zip(("rel", "ext", "value", "unit"), p[2:], g[2:]):
                acc[name] += (a == b)
    return {"paired SLOs": pairs, **{f"{k} equal": acc[k] / pairs if pairs else float("nan") for k in ("rel", "ext", "value", "unit")}}


def ensure_dir(vid):
    d = OUT / vid
    d.mkdir(parents=True, exist_ok=True)
    return d


def emit(vid, title, kind, tables, figure=None, findings=None):
    """Write one numbered analysis: CSV + LaTeX per table, figure PNG + PDF, and V-nn.md with the manuscript text. `tables` = {name: DataFrame}."""
    d = ensure_dir(vid)
    for name, df in tables.items():
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        df.to_csv(d / f"{vid}_{slug}.csv")
        (d / f"{vid}_{slug}.tex").write_text(df.to_latex(float_format="%.3f", escape=True))
    figs = []
    if figure is not None:
        figure.savefig(d / f"{vid}.png", dpi=300, bbox_inches="tight"); figure.savefig(d / f"{vid}.pdf", bbox_inches="tight")
        figs = [f"{vid}.png", f"{vid}.pdf"]
    f = (findings or {}).get(vid, {})
    md = [f"# {vid}  {title}", "", f"Type: {kind}", ""]
    for k, label in (("purpose", "Purpose"), ("method", "Method (one paragraph for the manuscript)"), ("result", "Result"), ("finding", "Finding (manuscript text)"), ("limitation", "Limitation")):
        md += [f"**{label}.** {f.get(k, '(to be written)')}", ""]
    md += ["Files: " + ", ".join(sorted(p.name for p in d.iterdir() if p.name != f"{vid}.md")), ""]
    (d / f"{vid}.md").write_text("\n".join(md))
    return d


def write_index(entries, title="Valid-intent analyses (notebook 16)", folder="validAnalysis", serial="V-nn"):
    lines = [f"# {title}", "",
             f"Ask for an analysis by its serial number. Each folder holds the table (CSV and LaTeX), the figure (PNG and PDF) and `{serial}.md` with the manuscript text.", "",
             "| Serial | Type | Title | Folder |", "|---|---|---|---|"]
    for vid, t, kind in entries:
        lines.append(f"| {vid} | {kind} | {t} | `outputs/{folder}/{vid}/` |")
    (OUT / "index.md").write_text("\n".join(lines) + "\n")


# ---------- V-09: examples of the error types ----------
FIELDS = ("rel", "ext", "value", "unit")


def taxonomy_examples(rec, pred_slos):
    """list of (error type, gold tuple or None, predicted tuple or None) for the unmatched SLOs of one record (same pairing as src.analysis.error_taxonomy)"""
    gold = canon_list(rec["label"]["slos"]); pred = canon_list(pred_slos)
    p, g = Counter(pred), Counter(gold)
    common = p & g
    up = list((p - common).elements()); ug = list((g - common).elements())
    out = []
    for gt in ug:
        match = next((pt for pt in up if (pt[0], pt[1]) == (gt[0], gt[1])), None)
        if match is None:
            out.append(("missed", gt, None)); continue
        up.remove(match)
        diff = [f for f, a, b in zip(FIELDS, match[2:], gt[2:]) if a != b]
        out.append(("wrong " + "+".join(diff), gt, match))
    for pt in up:
        out.append(("spurious", None, pt))
    return out
