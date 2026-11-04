"""Helpers for notebook 13 (fs3 and fs3 CoT on Qwen3.5 4B/9B and Gemma 4 E2B/E4B). Builds on src/zs3_report.py (same run discovery and scoring)."""
import re
from pathlib import Path

from src.analysis import extremum_outcomes
from src.data_io import load_records
from src.score import load_predictions
from src.zs3_report import *  # noqa: F401,F403
from src.zs3_report import find_run, last_json_split, records

VALID_ROWS3 = {
    "fs3": "fs3_{m}_2*", "fs3 CoT": "fs3_cot_{m}_2*", "zs3": "zs3_{m}_2*", "zs3 CoT": "zs3_cot_{m}_2*",
    "zero-shot p_mc": "zs_p_mc_{m}_2*", "thinking p_mc": "zs_think_p_mc_{m}_2*",
    "fs1": "fs-f1_{m}_2*", "fs2": "fs-f2_{m}_2*", "fs2 + thinking": "fs-f2_think_{m}_2*"}
INVALID_ROWS3 = {
    "fs3": "fs3_invalid_{m}_2*", "fs3 CoT": "fs3_cot_invalid_{m}_2*", "zs3": "zs3_invalid_{m}_2*", "zs3 CoT": "zs3_cot_invalid_{m}_2*",
    "zero-shot p_mc": "zs_p_mc_invalid_{m}_2*", "thinking p_mc": "zs_think_p_mc_invalid_{m}_2*",
    "fs2": "fs-f2_invalid_{m}_2*", "fs2 + thinking": "fs-f2_think_invalid_{m}_2*"}
NEW3 = ["fs3", "fs3 CoT"]

NODE_RE = re.compile(r"rack-[abc]|pool-[xy]")
SERVICE_RE = re.compile(r"svc/example")
EXAMPLE_REF_RE = re.compile(r"[Ee]xample [123]")


def conflict_kind(expr):
    """Keyword rule (approximate, audited by printing every expression with its kind): which contradiction the conflicting intent contains.
    extremum pair = highest and lowest of one metric; placement pair = edge and cloud both required; bound pair = a lower bound above an upper bound.
    Two or more features in one intent = 'combined'."""
    e = expr.lower()
    f = []
    if "highest" in e and "lowest" in e:
        f.append("highest + lowest")
    if ("cloud" in e and "edge" in e) and ("only on" in e or "at the same time" in e or "simultaneously" in e or "both on" in e):
        f.append("edge + cloud both required")
    if "below" in e and "above" in e:
        f.append("upper bound below lower bound")
    return f[0] if len(f) == 1 else ("combined: " + " + ".join(f) if f else "other")


def conflict_rows():
    return {r["index"]: conflict_kind(r["input"]["intent"]["expression"]) for r in records() if r["meta"]["status"] == "conflicting"}


def leakage(run):
    """Example content in a run: records whose reasoning (before the last JSON) names an example node, cites 'Example n', or whose final JSON
    contains an example node/service name or the example number 25769803776."""
    preds = list(load_predictions(Path(run) / "predictions.jsonl").values())
    out = {"reasoning with example node": 0, "reasoning cites Example n": 0, "final JSON with example name/number": 0}
    for p in preds:
        reas, ans, _ = last_json_split(p["raw_text"])
        out["reasoning with example node"] += bool(NODE_RE.search(reas))
        out["reasoning cites Example n"] += bool(EXAMPLE_REF_RE.search(reas))
        out["final JSON with example name/number"] += bool(NODE_RE.search(ans) or SERVICE_RE.search(ans) or "25769803776" in ans)
    return out


def lookup_outcomes(run):
    preds = load_predictions(Path(run) / "predictions.jsonl")
    return dict(extremum_outcomes(records(), preds))
