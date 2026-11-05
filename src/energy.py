"""Helpers for the deployment energy probe (sampling, prompt layouts, cost model). Pure functions, no GPU needed."""
import random
import re

import numpy as np


def stratified_sample(records, per_level=4, seed=20261018):
    """per_level valid intents from each difficulty level (1-5), seeded random, returned sorted by (level, index)."""
    rng = random.Random(seed)
    out = []
    for level in sorted({r["meta"]["level"] for r in records if r["meta"]["status"] == "valid"}):
        pool = sorted((r for r in records if r["meta"]["status"] == "valid" and r["meta"]["level"] == level), key=lambda r: r["index"])
        out += rng.sample(pool, per_level)
    return out


_SERVICE = re.compile(r'^(?P<ind>\s*)"service": "\{service\}",\s*$', re.M)
_EXPR = re.compile(r'^(?P<ind>\s*)"expression": "\{expression\}",\s*$', re.M)
_SNAP = re.compile(r'^(?P<ind>\s*)"snapshot": \{snapshot\}\s*$', re.M)


def cache_friendly(template):
    """Move the snapshot to the front of the final Input block so that it precedes the per-request intent
    (same content, different order): "snapshot", "service", "expression". Everything before the block is unchanged."""
    for rx in (_SERVICE, _EXPR, _SNAP):
        if len(rx.findall(template)) != 1:
            raise ValueError("template does not contain exactly one service/expression/snapshot line each")
    s = _SERVICE.search(template); e = _EXPR.search(template); n = _SNAP.search(template)
    assert s.start() < e.start() < n.start()
    ind = n.group("ind")
    block = f'{ind}"snapshot": {{snapshot}},\n{s.group("ind")}"service": "{{service}}",\n{e.group("ind")}"expression": "{{expression}}"'
    return template[:s.start()] + block + template[n.end():]


def fit_cost(uncached_tokens, gen_tokens, energy_j):
    """Least squares E = a + b * uncached_prompt_tokens + c * generated_tokens. Returns coefficients and R^2."""
    X = np.column_stack([np.ones(len(energy_j)), uncached_tokens, gen_tokens]).astype(float)
    y = np.asarray(energy_j, dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"a_J": float(coef[0]), "b_J_per_input_token": float(coef[1]), "c_J_per_output_token": float(coef[2]), "r2": 1 - ss_res / ss_tot if ss_tot else float("nan")}


# ---------------------------------------------------------------- notebook 18 (energy of the zs3 / fs3 baselines): samples
def n1_sample(records, per_level=12, seed=20261025):
    """per_level valid intents per level for the one-request-at-a-time windows (sorted by level, index)"""
    return stratified_sample(records, per_level, seed)


def loop_sample(records, k, exclude, seed=20261026):
    """k distinct valid intents for a closed-loop window, per-level quotas proportional to the number of valid records per level, excluding `exclude` (indices)."""
    rng = random.Random(seed)
    pools = {}
    for r in records:
        if r["meta"]["status"] == "valid" and r["index"] not in exclude:
            pools.setdefault(r["meta"]["level"], []).append(r)
    total = sum(len(v) for v in pools.values())
    quota = {lv: max(1, round(k * len(v) / total)) for lv, v in pools.items()}
    while sum(quota.values()) > k:
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < k:
        quota[max(quota, key=lambda lv: len(pools[lv]) - quota[lv])] += 1
    out = []
    for lv in sorted(pools):
        out += rng.sample(sorted(pools[lv], key=lambda r: r["index"]), quota[lv])
    rng.shuffle(out)
    return out


def invalid_sample(records, per_category=10, seed=20261027):
    """per_category invalid intents from each of the four categories (ambiguous, conflicting, malformed, unsupported)"""
    rng = random.Random(seed)
    out = []
    for cat in ("ambiguous", "conflicting", "malformed", "unsupported"):
        pool = sorted((r for r in records if r["meta"]["status"] == cat), key=lambda r: r["index"])
        out += rng.sample(pool, per_category)
    return out
