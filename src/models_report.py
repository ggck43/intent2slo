"""Helpers for notebook 14 (zs3 / zs3 CoT on Ministral 3, Granite 4.1 and Llama; Qwen3.5 and Gemma 4 as reference rows).
Builds on src/zs3_report.py and src/fs3_report.py (same run discovery, scoring, rejection and conflict-kind rule)."""
import re
from pathlib import Path

from src.fs3_report import *  # noqa: F401,F403
from src.fs3_report import conflict_rows, find_run, last_json_split  # noqa: F401
from src.score import load_predictions

NEW_MODELS = {"Ministral-3-8B": "ministral-3-8b-instruct-2512", "Ministral-3-3B": "ministral-3-3b-instruct-2512",
              "Granite-4.1-8B": "granite-41-8b", "Granite-4.1-3B": "granite-41-3b",
              "Llama-3.1-8B": "llama-31-8b-instruct", "Llama-3.2-3B": "llama-32-3b-instruct"}
REF_MODELS = {"Qwen3.5-4B": "qwen35-4b", "Qwen3.5-9B": "qwen35-9b", "Gemma4-E2B": "gemma-4-e2b-it", "Gemma4-E4B": "gemma-4-e4b-it"}
ALL_MODELS = {**NEW_MODELS, **REF_MODELS}
FAMILY = {"Ministral-3-8B": "Ministral", "Ministral-3-3B": "Ministral", "Granite-4.1-8B": "Granite", "Granite-4.1-3B": "Granite",
          "Llama-3.1-8B": "Llama (side study)", "Llama-3.2-3B": "Llama (side study)", "Qwen3.5-4B": "Qwen", "Qwen3.5-9B": "Qwen",
          "Gemma4-E2B": "Gemma", "Gemma4-E4B": "Gemma"}
ROWS_VALID = {"zs3": "zs3_{m}_2*", "zs3 CoT": "zs3_cot_{m}_2*"}
ROWS_INVALID = {"zs3": "zs3_invalid_{m}_2*", "zs3 CoT": "zs3_cot_invalid_{m}_2*"}
THINK_OPEN = re.compile(r"<think>")


def tag_stats(run):
    """CoT format use: records with <think> and </think>, with a fenced answer, with reasoning before the final JSON, with the final JSON inside the tags."""
    preds = list(load_predictions(Path(run) / "predictions.jsonl").values())
    n = len(preds)
    out = {"records": n, "has <think>": 0, "has </think>": 0, "fenced answer": 0, "reasoning before JSON": 0, "bare JSON": 0}
    for p in preds:
        t = p["raw_text"]
        out["has <think>"] += "<think>" in t; out["has </think>"] += "</think>" in t
        out["fenced answer"] += "```" in t
        reas, ans, st = last_json_split(t)
        out["reasoning before JSON"] += st == "ok"; out["bare JSON"] += st == "no_reasoning"
    return out


def run_date(run):
    return Path(run).name.rsplit("_", 1)[-1][:8]
