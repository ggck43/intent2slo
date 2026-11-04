"""Helpers for notebook 12 (zs3 and zs3 CoT on Qwen3.5 4B/9B and Gemma 4 E2B/E4B). Pure functions of predictions.jsonl + dataset.json.
Run directories are found by name pattern; the latest matching directory wins. Native thinking is off in all new runs."""
import glob
import json
import re
from collections import Counter
from pathlib import Path

from src.analysis import error_taxonomy
from src.data_io import load_records
from src.parse import find_top_level_objects, strip_fences
from src.reject import CATEGORIES, outcome
from src.score import canon_list, load_predictions, per_record, prf

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"Qwen3.5-4B": "qwen35-4b", "Qwen3.5-9B": "qwen35-9b", "Gemma4-E2B": "gemma-4-e2b-it", "Gemma4-E4B": "gemma-4-e4b-it"}
RECS = None

# (label -> glob with {m} = model tag); the first two are the new zs3 runs
VALID_ROWS = {
    "zs3": "zs3_{m}_2*", "zs3 CoT": "zs3_cot_{m}_2*",
    "zero-shot p_paper (old)": "zs_{m}_2*", "CoT p_paper (old)": "zs_cot_{m}_2*",
    "zero-shot p_mc": "zs_p_mc_{m}_2*", "thinking p_mc": "zs_think_p_mc_{m}_2*",
    "fs2": "fs-f2_{m}_2*", "fs2 + thinking": "fs-f2_think_{m}_2*"}
INVALID_ROWS = {
    "zs3": "zs3_invalid_{m}_2*", "zs3 CoT": "zs3_cot_invalid_{m}_2*",
    "zero-shot p_mc": "zs_p_mc_invalid_{m}_2*", "thinking p_mc": "zs_think_p_mc_invalid_{m}_2*",
    "fs2": "fs-f2_invalid_{m}_2*", "fs2 + thinking": "fs-f2_think_invalid_{m}_2*"}
NEW = ["zs3", "zs3 CoT"]

# paper, GPT-4.1 mini, chart readings (+-0.03), as in notebook 9: rejection rate per category (ambiguous, conflicting, malformed, unsupported)
PAPER_REJECT = {"zero-shot": (.98, .90, .69, .98), "few-shot": (.62, 0.0, .09, .91), "zero-shot CoT": (1.0, .88, .75, .98), "few-shot CoT": (.82, 0.0, .17, .87),
                "Intent Engine": (1.0, .94, 1.0, 1.0)}
N_CAT = {"ambiguous": 65, "conflicting": 49, "malformed": 35, "unsupported": 46}


def records():
    global RECS
    if RECS is None:
        RECS = load_records()
    return RECS


def find_run(pattern):
    hits = sorted(glob.glob(str(ROOT / "outputs" / "runs" / pattern)))
    return Path(hits[-1]) if hits else None


def registry(rows, subset):
    out = []
    for label, pat in rows.items():
        for mname, m in MODELS.items():
            d = find_run(pat.format(m=m))
            n = None
            if d is not None and (d / "predictions.jsonl").exists():
                n = sum(1 for _ in open(d / "predictions.jsonl"))
            out.append({"subset": subset, "prompt": label, "model": mname, "run dir": d.name if d else None, "records": n})
    return out


def scope_conflict_indices():
    """valid records whose gold lookup is the global extremum although a layer is requested (59 synthetic records; see notebook 10)"""
    from src.halluc_analysis import scope_conflict_records
    return set(scope_conflict_records())


# ---------- valid ----------
def valid_rows(run):
    preds = load_predictions(Path(run) / "predictions.jsonl")
    rows = per_record(records(), preds, only_predicted=True)
    meta = {r["index"]: r["meta"] for r in records()}
    for r in rows:
        r["level"] = meta[r["index"]]["level"]
        pr = preds[r["index"]]
        r["empty"] = bool(pr.get("parse_ok") and pr.get("parsed") and not pr["parsed"]["slos"])
        r["tokens"] = pr.get("completion_tokens")
    return rows


def summarize(rows, exclude=None):
    rows = [r for r in rows if not (exclude and r["index"] in exclude)]
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    s = prf(tp, fp, fn)
    n = len(rows)
    return {"n": n, "F1": s["f1"], "precision": s["precision"], "recall": s["recall"], "EM": sum(r["em"] for r in rows) / n,
            "Jaccard": sum(r["jaccard"] for r in rows) / n, "parse rate": sum(r["parse_ok"] for r in rows) / n,
            "truncated": sum(r["truncated"] for r in rows) / n, "false reject (empty slos)": sum(r["empty"] for r in rows) / n}


def taxonomy(run):
    preds = load_predictions(Path(run) / "predictions.jsonl")
    df = error_taxonomy(records(), preds)
    return df.groupby("error")["count"].sum().to_dict() if len(df) else {}


# ---------- run health ----------
def energy_wh(run):
    p = Path(run) / "energy.json"
    return json.loads(p.read_text())["gpu_energy_J"] / 3600 if p.exists() else None


def last_json_split(text):
    """(reasoning, answer, status): the answer starts at the last top-level JSON object with a list `slos`; <think> tags are ignored (decision 2026-10-25)"""
    last = None
    for sp in find_top_level_objects(strip_fences(text)):
        try:
            o = json.loads(sp)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and isinstance(o.get("slos"), list):
            last = sp
    if last is None:
        return text, "", "no_json"
    i = text.rfind(last)
    reasoning = re.sub(r"(```(json)?)\s*$", "", text[:i].rstrip()).rstrip()
    return reasoning, text[i:], "ok" if reasoning.strip() else "no_reasoning"


def max_repeated_line(text):
    c = Counter(l.strip() for l in text.splitlines() if len(l.strip()) > 25)
    return max(c.values()) if c else 0


def health(run, cap, base_run=None):
    preds = list(load_predictions(Path(run) / "predictions.jsonl").values())
    base = load_predictions(Path(base_run) / "predictions.jsonl") if base_run else {}
    n = len(preds)
    out = {"records": n, "parse rate": sum(p["parse_ok"] for p in preds) / n, "truncated": sum(bool(p["truncated"]) for p in preds) / n,
           "finish": dict(Counter(p["finish_reason"] for p in preds)), "median tokens": sorted(p["completion_tokens"] for p in preds)[n // 2],
           "max prompt tokens": max(p.get("prompt_tokens") or 0 for p in preds), "near cap (>=75%)": sum(p["completion_tokens"] >= 0.75 * cap for p in preds),
           "line repeated >=5x": sum(max_repeated_line(p["raw_text"]) >= 5 for p in preds), "JSON objects >=3": sum(p["n_json_objects"] >= 3 for p in preds),
           "contains <think>": sum("<think>" in p["raw_text"] for p in preds), "contains </think>": sum("</think>" in p["raw_text"] for p in preds),
           "energy Wh": energy_wh(run)}
    if base_run:
        sp = Counter(last_json_split(p["raw_text"])[2] for p in preds)
        out["reasoning before final JSON"] = sp.get("ok", 0)
        out["no reasoning (bare JSON)"] = sp.get("no_reasoning", 0)
        out["answer identical to base answer"] = sum(1 for p in preds if p["index"] in base and base[p["index"]]["raw_text"].strip() == p["raw_text"].strip())
    return out


# ---------- invalid ----------
def invalid_rows(run):
    from src.reject import rejection_rows
    rows = rejection_rows(run)
    preds = load_predictions(Path(run) / "predictions.jsonl")
    for r in rows:
        r["finish"] = preds[r["index"]].get("finish_reason")
    return rows


def rates(rows, strict=False):
    out = {}
    for c in CATEGORIES + ["overall"]:
        sub = [r for r in rows if c == "overall" or r["category"] == c]
        rej = sum((r["outcome"] == "rejected") or (not strict and r["outcome"] == "unparseable") for r in sub)
        out[c] = rej / len(sub) if sub else float("nan")
    return out


def counts(rows, strict=False):
    out = {}
    for c in CATEGORIES + ["overall"]:
        sub = [r for r in rows if c == "overall" or r["category"] == c]
        out[c] = sum((r["outcome"] == "rejected") or (not strict and r["outcome"] == "unparseable") for r in sub)
    return out
