"""Helpers for notebook 18 (energy of the zs3 / fs3 baselines, 6 models). Reads outputs/energy/<run_id>/<model-tag>/ written by scripts/energy_probe2.py."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"Qwen3.5-4B": "qwen35-4b", "Qwen3.5-9B": "qwen35-9b", "Gemma4-E2B": "gemma-4-e2b-it", "Gemma4-E4B": "gemma-4-e4b-it",
          "Ministral-3-3B": "ministral-3-3b-instruct-2512", "Ministral-3-8B": "ministral-3-8b-instruct-2512"}
FAMILY = {"Qwen3.5-4B": "Qwen", "Qwen3.5-9B": "Qwen", "Gemma4-E2B": "Gemma", "Gemma4-E4B": "Gemma", "Ministral-3-3B": "Ministral", "Ministral-3-8B": "Ministral"}
CFGS = ["zs3", "fs3", "zs3_cot", "fs3_cot"]
CFG_NAME = {"zs3": "zs3", "fs3": "fs3", "zs3_cot": "zs3 CoT", "fs3_cot": "fs3 CoT"}
KINDS = ("n1_valid", "n1_rep1", "n1_rep2", "n1_invalid", "c16", "c64")


def load_run(run_dir, models=MODELS):
    """{'req': DataFrame of all requests, 'phases': DataFrame, 'trace': {model: DataFrame}, 'env': {model: dict}}"""
    run_dir = Path(run_dir)
    reqs, phs, trace, env = [], [], {}, {}
    for name, tag in models.items():
        d = run_dir / tag
        if not (d / "requests.jsonl").exists():
            continue
        r = pd.DataFrame([json.loads(l) for l in open(d / "requests.jsonl") if l.strip()])
        r["model"] = name
        r["kind"] = [p[len(c) + 1:] for p, c in zip(r["phase"], r["config"])]
        reqs.append(r)
        p = pd.DataFrame([json.loads(l) for l in open(d / "phases.jsonl") if l.strip()]); p["model_name"] = name; phs.append(p)
        trace[name] = pd.read_csv(d / "power_trace.csv")
        env[name] = json.loads((d / "env.json").read_text())
    return {"req": pd.concat(reqs, ignore_index=True), "phases": pd.concat(phs, ignore_index=True), "trace": trace, "env": env}


def integrate(trace, t0, t1):
    """trapezoid integral of power (W) over [t0, t1] in joules, from the 0.25 s power trace"""
    t = trace["t_epoch"].to_numpy(); p = trace["power_W"].to_numpy()
    m = (t >= t0) & (t <= t1)
    if m.sum() < 2:
        return float("nan")
    return float(np.trapezoid(p[m], t[m]))


def steady_state(trace, reqs, window):
    """Energy per request of a closed-loop window. `reqs`: requests of the window (t_start, t_end), `window`: the phases row (t_window_start, t_window_end, window_energy_J, concurrency).
    Steady state = from the completion of the N-th request (first wave finished) to the start of the last request (the loop starts to drain). Power is integrated from the trace
    and scaled so that the integral over the whole window equals the NVML counter reading. Returns whole-window and steady-state energy per request."""
    N = int(window["concurrency"]); K = len(reqs)
    whole = window["window_energy_J"] / K
    if K < N:
        return {"whole_window_J": whole, "steady_J": float("nan"), "scale": float("nan"), "n_steady": 0, "steady_s": 0.0}
    t_end = np.sort(reqs["t_end"].to_numpy())
    ta, tb = float(t_end[N - 1]), float(reqs["t_start"].max())
    full = integrate(trace, window["t_window_start"], window["t_window_end"])
    scale = window["window_energy_J"] / full if full and not math.isnan(full) else float("nan")
    n_done = int(((t_end > ta) & (t_end <= tb)).sum())
    if tb <= ta or n_done < max(4, N // 4):
        return {"whole_window_J": whole, "steady_J": float("nan"), "scale": scale, "n_steady": n_done, "steady_s": max(0.0, tb - ta)}
    e = integrate(trace, ta, tb) * scale
    return {"whole_window_J": whole, "steady_J": e / n_done, "scale": scale, "n_steady": n_done, "steady_s": tb - ta}


def fit_cost(df):
    """E = a + b * uncached_prompt_tokens + c * completion_tokens; returns coefficients and R^2 (in-sample)"""
    X = np.column_stack([np.ones(len(df)), df["uncached_prompt_tokens"], df["completion_tokens"]]).astype(float)
    y = df["energy_J"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return coef, 1 - ss_res / ss_tot if ss_tot else float("nan")


def cv_cost(df, folds=5, seed=0):
    """cross-validated prediction of per-request energy: returns R^2, mean absolute percentage error, and the error of the summed energy"""
    idx = np.arange(len(df)); rng = np.random.RandomState(seed); rng.shuffle(idx)
    parts = np.array_split(idx, folds); pred = np.zeros(len(df))
    X = np.column_stack([np.ones(len(df)), df["uncached_prompt_tokens"], df["completion_tokens"]]).astype(float); y = df["energy_J"].to_numpy(float)
    for k in range(folds):
        te = parts[k]; tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        coef, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None); pred[te] = X[te] @ coef
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"cv_r2": 1 - ss_res / ss_tot, "mape": float(np.mean(np.abs(pred - y) / y)), "sum_error": float(pred.sum() / y.sum() - 1)}


def pareto_front(points):
    """indices of the non-dominated points; points = list of (energy, accuracy): lower energy and higher accuracy are better"""
    keep = []
    for i, (e, a) in enumerate(points):
        if not any((e2 <= e and a2 >= a) and (e2 < e or a2 > a) for j, (e2, a2) in enumerate(points) if j != i):
            keep.append(i)
    return keep


def fit_cost2(df):
    """Two-parameter model E = a + c * completion_tokens. The uncached prompt length varies by only a few percent between requests (5.2k to 5.6k tokens),
    so the prefill term cannot be separated from the fixed term; a is the output-independent cost of a request (prefill of a typical prompt plus overheads)."""
    X = np.column_stack([np.ones(len(df)), df["completion_tokens"]]).astype(float); y = df["energy_J"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return coef, 1 - ss_res / ss_tot if ss_tot else float("nan")


def cv_cost2(df, folds=5, seed=0):
    idx = np.arange(len(df)); rng = np.random.RandomState(seed); rng.shuffle(idx); parts = np.array_split(idx, folds); pred = np.zeros(len(df))
    X = np.column_stack([np.ones(len(df)), df["completion_tokens"]]).astype(float); y = df["energy_J"].to_numpy(float)
    for k in range(folds):
        te = parts[k]; tr = np.concatenate([parts[j] for j in range(folds) if j != k]); coef, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None); pred[te] = X[te] @ coef
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"cv_r2": 1 - ss_res / ss_tot, "mape": float(np.mean(np.abs(pred - y) / y)), "sum_error": float(pred.sum() / y.sum() - 1)}
