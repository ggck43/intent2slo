"""Hallucination flags H1/H2/H3 for predicted SLO constraints (draft v0, definitions agreed 2026-10-20).

Each emitted constraint is checked against three sources of truth; every fault is counted in exactly one category:
  H3  schema      the constraint is not well formed in the supported constraint space (gold is not used)
  H1  intent      the constraint (what resource, relation, qualifier) has no counterpart in the gold label;
                  any constraint returned for an invalid intent
  H2  value       a gold counterpart exists, but the value (or unit) is wrong, missing or not supported by the snapshot
Omissions (gold constraints that are not predicted) are NOT hallucinations; they are counted separately.
"""
import re

from src.data_io import metric_table
from src.score import canon_slo

TYPES: dict[str, set[str | None]] = {
    "static_compute_resource": {"cpu_capacity", "memory_capacity", "storage_capacity"},
    "dynamic_compute_resource": {"cpu_utilization", "memory_utilization", "storage_utilization"},
    "node_network_resource": {"port_utilization", "bandwidth"},
    "placement_location": {None},
}
BYTES = {"bytes", "B", "KB", "MB", "GB", "KiB", "MiB", "GiB"}
UNITS = {"cpu_capacity": {"cores", "millicores"}, "memory_capacity": BYTES, "storage_capacity": BYTES,
         "cpu_utilization": {"percent", "%"}, "memory_utilization": {"percent", "%"}, "storage_utilization": {"percent", "%"},
         "port_utilization": {"percent", "%"}, "bandwidth": {"Bps", "Kbps", "Mbps", "Gbps"}}
CMP_OK = {None, "<=", ">=", "<", ">", "=", "==", "between"}  # "between" is in the prompt's own cmp list
OP_OK = {None, "set", "highest", "lowest", "<=", ">=", "="}
PLACEMENT = {"edge", "cloud", "both", "any"}
CONVERSIONS = [1000, 1 / 1000, 1e9 / 2 ** 30, 2 ** 30 / 1e9, 1e6 / 2 ** 20, 2 ** 20 / 1e6, 1024, 1 / 1024, 1000 / 1024, 1024 / 1000]


def schema_faults(s):
    """H3 faults of one raw predicted SLO (list of subtype strings)."""
    if not isinstance(s, dict):
        return ["not_an_object"]
    t = s.get("target") if isinstance(s.get("target"), dict) else {}
    typ, metric, unit = s.get("type"), s.get("metric"), s.get("unit")
    cmp_, op, val = t.get("cmp"), t.get("op"), t.get("value")
    if typ not in TYPES:
        return ["unknown_type"]
    out = []
    if typ == "placement_location":
        if val is None:
            out.append("placement_without_value")
        elif str(val).lower() not in PLACEMENT:
            words = set(re.findall(r"[a-z]+", str(val).lower())) - {"node", "nodes", "layer", "and", "only", "the", "or"}
            plain = re.fullmatch(r"[a-z ,]+", str(val).lower()) is not None
            out.append("placement_value_variant" if plain and words and words <= PLACEMENT else "placement_value_outside_domain")
        return out
    if metric not in TYPES[typ]:
        if metric is None:
            return ["metric_missing"]
        return ["metric_outside_type" if metric in set().union(*TYPES.values()) else "unknown_metric"]
    if unit not in UNITS[metric]:
        out.append("unit_outside_domain")
    if cmp_ not in CMP_OK:
        out.append("operator_outside_enum")
    if op not in OP_OK:
        out.append("operator_outside_enum")
    if val is not None and (isinstance(val, bool) or not isinstance(val, (int, float))):
        try:
            float(val)
        except (TypeError, ValueError):
            out.append("value_not_numeric")
    return out


def self_inconsistent(s):
    """extremum tag contradicts the comparator (e.g. highest with <=); a relation error (H1), not a schema fault"""
    t = s.get("target") if isinstance(s, dict) and isinstance(s.get("target"), dict) else {}
    cmp_, op = t.get("cmp"), t.get("op")
    return (op == "highest" and cmp_ in ("<=", "<")) or (op == "lowest" and cmp_ in (">=", ">"))


def _extrema(snap, metric):
    """{scope: {'highest': v, 'lowest': v}} over all nodes, edge nodes, cloud nodes (nulls skipped)."""
    rows = metric_table(snap)
    res = {}
    for scope in ("all", "edge", "cloud"):
        vals = [m[metric] for _, layer, m in rows if (scope == "all" or layer == scope) and m.get(metric) is not None]
        if vals:
            res[scope] = {"highest": max(vals), "lowest": min(vals)}
    return res


_SNAP_NUMS: dict = {}


def _snapshot_numbers(snap):
    key = id(snap)
    if key not in _SNAP_NUMS:
        nums = []

        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                nums.append(float(o))

        walk(snap)
        _SNAP_NUMS[key] = nums
    return _SNAP_NUMS[key]


def _close(a, b):
    try:
        return a is not None and b is not None and abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):  # a non-numeric value such as "lowest" never matches
        return False


def classify(s, gold, status, snap):
    """Flags for one raw predicted SLO against the canonical gold list. Returns [(flag, subtype), ...]."""
    flags = [("H3", f) for f in schema_faults(s)]
    if not isinstance(s, dict) or any(f in ("not_an_object", "unknown_type") for _, f in flags):
        return flags
    k = canon_slo(s)
    if any(f in ("operator_outside_enum", "metric_missing", "metric_outside_type", "unknown_metric", "placement_without_value", "placement_value_outside_domain", "placement_value_variant") for _, f in flags) and status == "valid":
        return flags  # semantics cannot be read: the fault is structural, nothing is compared with gold
    if status != "valid":
        return flags + [("H1", f"accepted_invalid:{status}")]
    same = [g for g in gold if g[:4] == k[:4]]
    if not same:
        keyed = [g for g in gold if g[:2] == k[:2]]
        if not keyed:
            sub = "vacuous_placement" if k[0] == "placement_location" and k[4] == "any" else "spurious_constraint"
            return flags + [("H1", sub)]
        base = lambda rel: {"<": "<=", ">": ">="}.get(rel, rel)  # strict and non-strict comparators have the same direction
        same_dir = [x for x in keyed if base(x[2]) == base(k[2])]
        same_ext = [x for x in keyed if x[3] == k[3]]
        if same_dir:  # right direction: qualifier or strictness differs
            g = same_dir[0]
            if k[3] and not g[3]:
                sub = "unsupported_extremum_tag"
            elif g[3] and not k[3]:
                # qualifier dropped: harmless if the number is still the gold number, a wrong value otherwise
                sub = "missing_extremum_tag" if _close(k[4], g[4]) else "missing_extremum_tag+wrong_value"
            elif k[3] != g[3]:
                sub = "wrong_extremum"
            else:
                sub = "boundary_strictness"
        elif same_ext:  # right qualifier, opposite direction
            sub = "wrong_direction"
        else:
            sub = "wrong_extremum"
        sub += "+self_inconsistent" if self_inconsistent(s) else ""
        if not sub.startswith("boundary_strictness"):
            return flags + [("H1", sub)]
        # strictness (< for <=) is a convention note: the constraint still matches gold, so its value is checked too
        flags = flags + [("H1", sub)]
        k = (k[0], k[1], same_dir[0][2], k[3], k[4], k[5])
        same = [g for g in gold if g[:4] == k[:4]]
    if any(f == "unit_outside_domain" for _, f in flags):  # the unit fault is already counted in H3
        k = k[:5] + (same[0][5],)
    if k in same:
        return flags
    if any(g[5] == k[5] and isinstance(g[4], float) and isinstance(k[4], float) and abs(g[4] - k[4]) <= 1e-9 * max(1.0, abs(g[4])) for g in same):
        return flags  # same number up to float noise (17th digit)
    g = min(same, key=lambda x: (x[4] is None or k[4] is None or x[4] != k[4], 0))
    if k[0] == "placement_location":
        return flags + [("H2", "placement_value")]
    gv, pv = g[4], k[4]
    if g[3] in ("highest", "lowest"):  # lookup
        if pv is None:
            return flags + [("H2", "abstention_null_lookup")]
        if not isinstance(pv, float):
            return flags
        ex = _extrema(snap, k[1])
        for scope, e in ex.items():
            if _close(pv, e[g[3]]):
                return flags + [("H2", "other_scope_real_extremum")]
        from src.data_io import SNAPSHOT_FIELDS  # noqa: F401
        vals = [m[k[1]] for _, _, m in metric_table(snap) if m.get(k[1]) is not None]
        if any(_close(pv, v) for v in vals):
            return flags + [("H2", "misgrounded_real_value")]
        if any(_close(pv, v) for v in _snapshot_numbers(snap)):
            return flags + [("H2", "wrong_metric_value")]  # a real snapshot number, but of another metric
        return flags + [("H2", "fabricated_value")]
    if any(f == "value_not_numeric" for _, f in flags):  # the value fault is already counted in H3
        return flags
    if pv is None:
        return flags + [("H2", "missing_value")]
    if isinstance(pv, float) and gv and any(abs(pv / gv - c) < 1e-3 * c for c in CONVERSIONS):
        return flags + [("H2", "unit_conversion")]
    if pv == gv and k[5] != g[5]:
        return flags + [("H2", "unit_label")]
    return flags + [("H2", "wrong_explicit_value")]


def response_report(pred, gold_raw, status, snap):
    """pred: the prediction row (parse_ok, parsed). Returns flags per response, constraint-level faults and omissions.

    h: {'H1','H2','H3'} -> 0/1;  faults: list of (flag, subtype) per constraint fault (identical duplicate constraints collapsed);
    n_pred: number of distinct emitted constraints; omissions: gold constraints with no equal predicted constraint (not a hallucination)."""
    gold = [canon_slo(g) for g in gold_raw]
    if not pred.get("parse_ok"):
        return {"h": {"H1": 0, "H2": 0, "H3": 1}, "faults": [("H3", "unparseable")], "n_pred": 0, "omissions": len(gold), "n_gold": len(gold), "clean_constraints": 0}
    faults, seen, clean, pred_canon = [], set(), 0, []
    for s in pred["parsed"]["slos"]:
        c = canon_slo(s)
        if c in seen:
            continue
        seen.add(c)
        pred_canon.append(c)
        f = classify(s, gold, status, snap)
        faults += f
        clean += not f
    h = {k: int(any(x[0] == k for x in faults)) for k in ("H1", "H2", "H3")}
    from collections import Counter
    om = sum((Counter(gold) - Counter(pred_canon)).values())
    return {"h": h, "faults": faults, "n_pred": len(pred_canon), "omissions": om, "n_gold": len(gold), "clean_constraints": clean}
