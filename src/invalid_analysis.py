"""Helpers for notebook 17 (analyses of the 195 invalid intents, numbered I-nn, for the manuscript). Stored predictions only."""
import re
from collections import Counter

import numpy as np
import pandas as pd

from src.reject import CATEGORIES, outcome
from src.score import load_predictions
from src.valid_analysis import (FAMILY, LARGE, MODELS, PROMPTS, PROMPT_COLOR, PAPER_NAME, SMALL, OUT, ensure_dir, emit, write_index,  # noqa: F401
                                find_run, load_all as load_valid, records, summarize)
from src.zs3_report import PAPER_REJECT, N_CAT, counts, invalid_rows, rates, last_json_split  # noqa: F401
from src.fs3_report import conflict_rows  # noqa: F401

PATTERN_INV = {"zs3": "zs3_invalid_{m}_2*", "fs3": "fs3_invalid_{m}_2*", "zs3 CoT": "zs3_cot_invalid_{m}_2*", "fs3 CoT": "fs3_cot_invalid_{m}_2*"}
PAPER_KEY = {"zs3": "zero-shot", "fs3": "few-shot", "zs3 CoT": "zero-shot CoT", "fs3 CoT": "few-shot CoT"}
STRONG4 = ["Qwen3.5-9B", "Qwen3.5-4B", "Ministral-3-8B", "Gemma4-E4B"]  # fixed in the plan before the vote results were computed
WEAK = ["Llama-3.2-3B", "Llama-3.1-8B", "Granite-4.1-3B", "Gemma4-E2B"]


def load_invalid():
    out = {}
    for m, tag in MODELS.items():
        for p in PROMPTS:
            d = find_run(PATTERN_INV[p].format(m=tag))
            if d is None or not (d / "predictions.jsonl").exists():
                raise FileNotFoundError(f"missing invalid run for {m} / {p}")
            rows = invalid_rows(d)
            out[(m, p)] = {"run": d, "rows": rows, "preds": load_predictions(d / "predictions.jsonl")}
    return out


def rate_table(runs, strict=False):
    return pd.DataFrame({k: rates(v["rows"], strict=strict) for k, v in runs.items()}).T


def slo_text(s):
    """short readable form of one predicted SLO"""
    if not isinstance(s, dict):
        return str(s)[:40]
    t = s.get("target") if isinstance(s.get("target"), dict) else {}
    if s.get("type") == "placement_location":
        return f"placement={t.get('value')}"
    return f"{s.get('metric')} {t.get('cmp') or ''}{(' ' + t['op']) if t.get('op') in ('highest', 'lowest') else ''} {t.get('value')}".replace("  ", " ").strip()


def accepted_slos(pred):
    p = pred.get("parsed")
    return [s for s in (p.get("slos") or [])] if pred.get("parse_ok") and isinstance(p, dict) else []


def content_type(slos):
    """what an accepted answer consists of (rule based): placement only, lookups only, thresholds only, mixed"""
    ds = [s for s in slos if isinstance(s, dict)]
    plc = [s for s in ds if s.get("type") == "placement_location"]
    other = [s for s in ds if s.get("type") != "placement_location"]
    ext = [s for s in other if isinstance(s.get("target"), dict) and s["target"].get("op") in ("highest", "lowest")]
    thr = [s for s in other if s not in ext]
    if plc and not other:
        return "placement only"
    if other and not thr and ext:
        return "lookups only" + (" + placement" if plc else "")
    if other and not ext and thr:
        return "thresholds only" + (" + placement" if plc else "")
    if other and ext and thr:
        return "lookups and thresholds" + (" + placement" if plc else "")
    return "other"


def conflict_halves(kind, slos):
    """for conflicting intents: did the answer write both halves of the contradiction, one half, or neither (rule based)"""
    ds = [s for s in slos if isinstance(s, dict) and isinstance(s.get("target"), dict)]
    if kind.startswith("highest + lowest") or "highest + lowest" in kind:
        by = {}
        for s in ds:
            if s["target"].get("op") in ("highest", "lowest"):
                by.setdefault(s.get("metric"), set()).add(s["target"]["op"])
        if any(len(v) == 2 for v in by.values()):
            return "both halves"
        return "one half" if by else "neither"
    if kind.startswith("edge + cloud"):
        vals = {str(s["target"].get("value")).lower() for s in ds if s.get("type") == "placement_location"}
        if {"edge", "cloud"} <= vals:
            return "both halves"
        if vals & {"both", "any"}:
            return "merged into both / any"
        return "one half" if vals & {"edge", "cloud"} else "neither"
    if kind.startswith("upper bound"):
        by = {}
        for s in ds:
            if s.get("type") != "placement_location" and s["target"].get("cmp") in ("<=", ">=") and s["target"].get("op") not in ("highest", "lowest"):
                by.setdefault(s.get("metric"), set()).add(s["target"]["cmp"])
        if any(len(v) == 2 for v in by.values()):
            return "both halves"
        return "one half" if by else "neither"
    return "mixed kind"


REFUSAL_WORDS = re.compile(r"\b(reject|rejected|reject it|cannot|can't|unsupported|not supported|ambiguous|contradict\w*|inconsistent|not clear|unclear|garbled|incomplete|no measurable|empty)\b", re.I)


def reasoning_mentions_refusal(text):
    reas, _, status = last_json_split(text)
    return bool(REFUSAL_WORDS.search(reas)) if status == "ok" else None
