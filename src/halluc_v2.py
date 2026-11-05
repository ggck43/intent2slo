"""Policy v2 (2026-10-26): meaning-first reading of format-broken constraints in VALID intents.

v1 stopped at the first structural fault (wrong `type`, odd metric name, odd operator, unit outside the list) and labelled the
constraint Mistake/Format without looking at its content. v2 repairs the surface (deterministic, listed below) and then judges the
repaired constraint with the unchanged v1 flagger. The surface fault itself stays a Mistake/Format. The strongest label wins.
Nothing in v1 (src/halluc.py, src/halluc_verdict.py) is changed.
"""
import copy
import re

from src.halluc import TYPES, UNITS

STRUCT = {"unknown_metric", "unknown_type", "metric_outside_type", "metric_missing", "placement_value_outside_domain",
          "operator_outside_enum", "unit_outside_domain", "placement_value_variant", "placement_without_value"}
TYPE_OF = {m: t for t, ms in TYPES.items() for m in ms if m}
DEFAULT_UNIT = {"cpu_capacity": "millicores", "memory_capacity": "bytes", "storage_capacity": "bytes", "cpu_utilization": "percent",
                "memory_utilization": "percent", "storage_utilization": "percent", "port_utilization": "percent"}
CMP_WORDS = {"lowest": "<=", "highest": ">=", "below": "<=", "under": "<=", "above": ">=", "over": ">=", "at most": "<=", "at least": ">="}


def canon_metric(m, unit=None):
    """Map a variant metric name to a schema metric, or None if it cannot be read with certainty."""
    if not isinstance(m, str):
        return None
    low = m.lower().strip().split(".")[-1]
    if low in TYPE_OF:
        return low
    fam = next((f for f in ("cpu", "memory", "storage", "port") if low.startswith(f)), None)
    if low.startswith("bandwidth"):
        return "bandwidth"
    if fam is None:
        return None
    if re.search(r"util|usage|percent|pct", low):
        return f"{fam}_utilization"
    if re.search(r"capacity|total|bytes|millicores|cores", low):
        return f"{fam}_capacity" if fam != "port" else None
    if low == fam:  # bare family name: the unit decides
        if unit in ("percent", "%"):
            return f"{fam}_utilization"
        if unit in ("bytes", "B", "GB", "MB", "millicores", "cores"):
            return f"{fam}_capacity" if fam != "port" else None
    return None


def repair(slo):
    """Returns (repaired slo, list of repairs) or (None, []) when the constraint cannot be read."""
    if not isinstance(slo, dict):
        return None, []
    s = copy.deepcopy(slo)
    t = s.get("target") if isinstance(s.get("target"), dict) else {}
    s["target"] = t
    done = []
    if s.get("type") == "placement_location" and s.get("metric") in ("placement", None) and t.get("value") is not None:
        words = set(re.findall(r"[a-z]+", str(t["value"]).lower())) - {"node", "nodes", "layer", "side", "and", "only", "the", "or", "on"}
        if words and words <= {"edge", "cloud", "both", "any"} and len(words) == 1:
            t["value"] = words.pop()
            return s, ["placement value variant"]
        return None, []
    m = canon_metric(s.get("metric"), s.get("unit"))
    if m is None:
        # the metric slot may hold a type name while the real metric sits in the type slot ("storage_capacity" as type)
        m = canon_metric(s.get("type"), s.get("unit")) if s.get("type") not in TYPES else None
        if m is None:
            return None, []
        done.append("metric taken from the type slot")
    if s.get("metric") != m:
        done.append("metric name")
        s["metric"] = m
    if s.get("type") != TYPE_OF[m]:
        done.append("type from metric")
        s["type"] = TYPE_OF[m]
    cmp_, op = t.get("cmp"), t.get("op")
    if cmp_ in ("lowest", "highest"):
        op, cmp_ = (cmp_ if op in (None, cmp_) else op), CMP_WORDS[cmp_]
        done.append("cmp holds highest/lowest")
    elif isinstance(cmp_, str) and cmp_.lower() in CMP_WORDS:
        cmp_ = CMP_WORDS[cmp_.lower()]
        done.append("cmp word")
    if op in ("null", "below", "above", "set") or (isinstance(op, str) and op.lower() in CMP_WORDS and op not in ("highest", "lowest")):
        op = None
        done.append("op word")
    if cmp_ is None and op in ("highest", "lowest"):
        cmp_ = CMP_WORDS[op]
        done.append("cmp from op")
    t["cmp"], t["op"] = cmp_, op
    if m in DEFAULT_UNIT and s.get("unit") not in UNITS[m]:
        s["unit"] = DEFAULT_UNIT[m]
        done.append("unit")
    return s, done
