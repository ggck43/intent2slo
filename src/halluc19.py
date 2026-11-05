"""Helpers for notebook 19 (hallucination analysis, rule v2). Plan: plans/20261026.hallucinationAnalysis.md.
Input: outputs/hallucinationAnalysis/<tag>/responses.csv (scripts/halluc19_responses.py). Rates use parseable responses only."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "hallucinationAnalysis"
TOK = {"Qwen3.5-4B": "qwen35-4b", "Qwen3.5-9B": "qwen35-9b", "Gemma4-E2B": "gemma-4-e2b-it", "Gemma4-E4B": "gemma-4-e4b-it",
       "Ministral-3-3B": "ministral-3-3b-instruct-2512", "Ministral-3-8B": "ministral-3-8b-instruct-2512",
       "Granite-4.1-3B": "granite-41-3b", "Granite-4.1-8B": "granite-41-8b", "Llama-3.2-3B": "llama-32-3b-instruct", "Llama-3.1-8B": "llama-31-8b-instruct"}
NAME = {v: k for k, v in TOK.items()}
MODELS = list(TOK)
FAMILY = {"Qwen3.5-4B": "Qwen", "Qwen3.5-9B": "Qwen", "Gemma4-E2B": "Gemma", "Gemma4-E4B": "Gemma", "Ministral-3-3B": "Ministral", "Ministral-3-8B": "Ministral",
          "Granite-4.1-3B": "Granite", "Granite-4.1-8B": "Granite", "Llama-3.2-3B": "Llama", "Llama-3.1-8B": "Llama"}
SMALL = {"Qwen": "Qwen3.5-4B", "Gemma": "Gemma4-E2B", "Ministral": "Ministral-3-3B", "Granite": "Granite-4.1-3B", "Llama": "Llama-3.2-3B"}
LARGE = {"Qwen": "Qwen3.5-9B", "Gemma": "Gemma4-E4B", "Ministral": "Ministral-3-8B", "Granite": "Granite-4.1-8B", "Llama": "Llama-3.1-8B"}
FAMILIES = ["Qwen", "Gemma", "Ministral", "Granite", "Llama"]
PROMPTS = ["zs3", "fs3", "zs3 CoT", "fs3 CoT"]
CATS = ["ambiguous", "conflicting", "malformed", "unsupported"]


def load(tag="v2j"):
    R = pd.read_csv(BASE / tag / "responses.csv")
    R["name"] = R.model.map(NAME)
    R["prompt"] = R.setup
    R["size"] = np.where(R.name.isin(SMALL.values()), "small", "large")
    return R


def parsed(R):
    return R[R.parsed]


def rate(R, by, flag="has_hall", mask=None):
    """share of parseable responses with `flag`, per group"""
    P = parsed(R) if mask is None else parsed(R)[mask]
    g = P.groupby(by)[flag]
    return pd.DataFrame({"rate": g.mean(), "n": g.size(), "k": g.sum()})


class Boot:
    """Template-cluster bootstrap: the 716 records fall into 525 templates; each replicate draws 525 templates with replacement.
    The same draws are used for every group, so differences between groups are paired."""

    def __init__(self, R, B=2000, seed=20260101):
        self.codes = {t: i for i, t in enumerate(sorted(R.template.unique()))}
        self.T = len(self.codes)
        rng = np.random.RandomState(seed)
        self.W = rng.multinomial(self.T, np.ones(self.T) / self.T, size=B).astype(float)
        self.B = B

    def vecs(self, R, by, flag="has_hall", mask=None):
        P = parsed(R) if mask is None else parsed(R)[mask]
        P = P.assign(_c=P.template.map(self.codes))
        out = {}
        for key, g in P.groupby(by):
            num = np.bincount(g._c, weights=g[flag].astype(float), minlength=self.T)
            den = np.bincount(g._c, minlength=self.T).astype(float)
            out[key] = (num, den)
        return out

    def _est(self, num, den):
        point = num.sum() / den.sum() if den.sum() else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            b = (self.W @ num) / (self.W @ den)
        return point, b

    def rates(self, R, by, flag="has_hall", mask=None):
        rows = {}
        for key, (num, den) in self.vecs(R, by, flag, mask).items():
            p, b = self._est(num, den)
            rows[key] = dict(rate=p, lo=np.nanpercentile(b, 2.5), hi=np.nanpercentile(b, 97.5), n=int(den.sum()))
        return pd.DataFrame(rows).T

    def diff(self, R, by, a, b, flag="has_hall", mask=None):
        """rate(b) - rate(a) with paired bootstrap interval; a and b are keys of the grouping"""
        v = self.vecs(R, by, flag, mask)
        pa, ba = self._est(*v[a])
        pb, bb = self._est(*v[b])
        d = bb - ba
        return dict(diff=pb - pa, lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5))

    def pooled_diff(self, R, name_pairs, flag="has_hall", mask=None):
        """mean over several (small, large) model pairs of the paired difference; `name_pairs` = list of (model_a, model_b)"""
        v = self.vecs(R, "name", flag, mask)
        ds, ps = [], []
        for a, b in name_pairs:
            pa, ba = self._est(*v[a])
            pb, bb = self._est(*v[b])
            ds.append(bb - ba)
            ps.append(pb - pa)
        d = np.mean(ds, axis=0)
        return dict(diff=float(np.mean(ps)), lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5))


# ---------- constraint-level categories (A-05) ----------
VALID_CAT = {
    "spurious_constraint": "extra constraint (metric or layer the text never asks for)",
    "unsupported_extremum_tag": "spurious highest/lowest tag on a plain threshold",
    "wrong_extremum": "wrong extremum or direction", "wrong_direction": "wrong extremum or direction", "missing_extremum_tag": "wrong extremum or direction",
    "wrong_explicit_value": "wrong number stated in the intent", "unit_conversion": "wrong number stated in the intent",
    "fabricated_value": "number that is in no source", "misgrounded_real_value": "snapshot number of the wrong node, layer or metric",
    "other_scope_real_extremum": "snapshot number of the wrong node, layer or metric", "wrong_metric_value": "snapshot number of the wrong node, layer or metric",
    "placement_value": "wrong placement", "unknown_metric": "metric name in no source",
}


def invalid_cat(why):
    w = str(why)
    if "names node" in w:
        return "node the text does not name"
    if "names layer" in w or "both layers" in w:
        return "layer the text does not name"
    if "extremum" in w and ("not the" in w or "grounded" in w) and "states no extremum" not in w and "other direction" not in w:
        return "number not supported by the text or snapshot"
    if "states no extremum" in w or "other direction" in w or "direction" in w or "bound on" in w or "states a" in w:
        return "extremum or direction the text does not state"
    if "not named in the text" in w or "does not name this kind" in w or "not in the text, the schema" in w or "family not readable" in w:
        return "metric the text does not name"
    if "not in the text" in w or "is not the" in w or "as a bound" in w or "unreadable metric" in w:
        return "number not supported by the text or snapshot"
    return "other"


def hall_constraints(tag="v2j", file=None):
    f = file or {"v2j": "policy_v2j_constraints.csv", "v2": "policy_v2_constraints.csv", "v1": "policy_v1_constraints.csv", "v2jb": "policy_v2jb_constraints.csv"}[tag]
    F = pd.read_csv(ROOT / "outputs/hallucination" / f).fillna("")
    F = F[(F.constraint != "") & (F.label == "Hallucination")].copy()
    F["name"] = F.model.map(NAME)
    F["validity"] = np.where(F.status == "valid", "valid", "invalid")
    F["sub0"] = F["sub"].str.split("+").str[0]
    F["cat"] = [VALID_CAT.get(s, "other") if v == "valid" else invalid_cat(w) for s, v, w in zip(F.sub0, F.validity, F.why)]
    return F


# ---------- readable constraints (A-07) ----------
ABBR = {"placement_location": "placement", "static_compute_resource": "static", "dynamic_compute_resource": "dynamic", "node_network_resource": "network"}


def short(s):
    t = s.get("target") or {}
    v = t.get("value")
    v = round(v, 4) if isinstance(v, float) else v
    head = f"{ABBR.get(s.get('type'), s.get('type'))}/{s.get('metric')}" if s.get("metric") else ABBR.get(s.get("type"), s.get("type"))
    return " ".join(str(x) for x in (head, t.get("cmp") or "", t.get("op") or "", v if v is not None else "null", s.get("unit") or "") if x != "")
