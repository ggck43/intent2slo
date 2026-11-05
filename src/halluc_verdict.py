"""Policy v1 (2026-10-26): label every flagged constraint as Hallucination, Mistake, Label error, or no fault.
Input is a flag row from src.halluc.classify (sub-type) plus the record. The rules follow outputs/hallucination/codebook.md.
This module only decides labels; it does not change the flagger or the scorer."""
import re

from src.data_io import load_snapshot
from src.score import canon_slo

SI_BINARY = (1e9 / 2 ** 30, 2 ** 30 / 1e9)  # decimal GB instead of 2^30 bytes
EXT_WORDS = re.compile(r"\b(highest|lowest|most|least|maximum|minimum|max|min|largest|smallest|best|greatest|fewest)\b", re.I)
FAMILY = {"cpu": r"cpu|processor|core", "memory": r"memory|ram", "storage": r"storage|disk", "port": r"port", "bandwidth": r"bandwidth|network|throughput", "network": r"bandwidth|network|throughput"}
KINDW = {"utilization": r"utili[sz]ation|usage|load|busy|used", "capacity": r"capacity|cores|\bgb\b|\btb\b|size"}

METRIC_STEMS = ("cpu", "memory", "storage", "port", "bandwidth", "network")
FIELD_WORDS = {"highest", "lowest", "edge", "cloud", "both", "any", "set", "threshold", "max", "min"}
REQUIREMENT = {"spurious_constraint", "unsupported_extremum_tag", "wrong_direction", "wrong_extremum", "unknown_metric", "unknown_type"}
# proposed intrinsic / extrinsic tag (Ji et al.): intrinsic = contradicts something the sources state; extrinsic = adds what they do not contain
INTRINSIC = {"wrong_direction", "wrong_extremum", "wrong_explicit_value", "misgrounded_real_value", "wrong_metric_value", "other_scope_real_extremum", "unit_conversion", "missing_extremum_tag+wrong_value"}
EXTRINSIC = {"spurious_constraint", "unsupported_extremum_tag", "unknown_metric", "unknown_type", "fabricated_value"}


def _keys(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            acc.add(str(k))
            _keys(v, acc)
    elif isinstance(o, list):
        for v in o:
            _keys(v, acc)
    return acc


_KNOWN: dict = {}


def known_identifiers():
    """Names that exist in the sources: every key of the snapshot, the schema metrics and types, and the dotted type.metric forms."""
    if not _KNOWN:
        from src.halluc import UNITS
        ids = _keys(load_snapshot(), set()) | set(UNITS) | {"static_compute_resource", "dynamic_compute_resource", "node_network_resource", "placement_location"}
        types = {"static_compute_resource": ("cpu_capacity", "memory_capacity", "storage_capacity"), "dynamic_compute_resource": ("cpu_utilization", "memory_utilization", "storage_utilization"),
                 "node_network_resource": ("port_utilization", "bandwidth")}
        ids |= {f"{t}.{m}" for t, ms in types.items() for m in ms}
        _KNOWN["ids"] = ids
    return _KNOWN["ids"]


def _close(a, b):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= 1e-6 * max(1.0, abs(b))


def _gold_like(rec, k):
    return [canon_slo(g) for g in rec["label"]["slos"] if canon_slo(g)[:2] == k[:2]]


def text_support(rec, slo):
    """Invalid intents: is the returned constraint stated in the intent text? Heuristic; the audit measures how good it is."""
    from src.data_io import metric_table
    expr = rec["input"]["intent"]["expression"]
    t = (slo or {}).get("target") or {}
    pv = t.get("value")
    if (slo or {}).get("type") == "placement_location":
        if pv in (None, ""):
            return False, "empty placement object (format)"
        if pv == "any":
            return True, "placement 'any' carries no requirement"
        ok = str(pv).lower() in expr.lower()
        return ok, "placement is in the text" if ok else "placement not in the text"
    m = (slo or {}).get("metric") or ""
    if m not in known_identifiers() and (slo or {}).get("unit") in known_identifiers():
        m = slo["unit"]  # fields shifted: the metric name sits in the unit slot
    if not m or m in ("static_compute_resource", "dynamic_compute_resource", "node_network_resource", "placement_location"):
        return False, "no metric can be read from the constraint"
    fam, kind = m.split("_")[0], m.split("_")[-1]
    if fam in FAMILY and not re.search(FAMILY[fam], expr, re.I):
        return False, f"metric family '{fam}' not in the text"
    if kind in KINDW and not re.search(KINDW[kind], expr, re.I):
        return False, f"'{kind}' not in the text"
    snap_vals = [mm.get(m) for _, _, mm in metric_table(load_snapshot()) if mm.get(m) is not None]
    ext = t.get("op") in ("highest", "lowest") or t.get("cmp") in ("highest", "lowest")
    if ext:
        if not EXT_WORDS.search(expr):
            return False, "extremum not requested in the text"
        if isinstance(pv, (int, float)) and snap_vals and not any(_close(pv, v) for v in snap_vals):
            return False, "extremum value is not a snapshot value of this metric"
        return True, "extremum request on this metric is in the text and the value comes from the snapshot"
    num = isinstance(pv, (int, float)) and re.search(rf"\b{re.escape(str(int(pv)) if float(pv).is_integer() else str(pv))}\b", expr) is not None
    if not num and isinstance(pv, (int, float)) and snap_vals and any(_close(pv, v) for v in snap_vals) and EXT_WORDS.search(expr):
        return True, "snapshot value for a metric the text asks about"
    return num, "number is in the text" if num else "number not in the text"


def judge(row, rec, scope_conflict):
    """row: dict with sub, canon, status, index, slo. Returns dict(label, kind, why, tag). Labels: Hallucination, Mistake, Label error."""
    sub, base = row["sub"], row["sub"].split("+")[0]
    full = sub.replace("+self_inconsistent", "")
    k, slo = row["canon"], row["slo"]
    if sub.startswith("accepted_invalid"):
        ok, why = text_support(rec, slo)
        if ok:
            return dict(label="Mistake", kind="Accepted invalid (text-supported)", why=why, tag="")
        return dict(label="Hallucination", kind="Value" if why == "number not in the text" else "Requirement", why=why, tag="Extrinsic")
    if base in ("unknown_metric", "unknown_type"):
        name = (slo or {}).get("metric" if base == "unknown_metric" else "type")
        low = name.lower() if isinstance(name, str) else ""
        if name is None or (isinstance(name, str) and (name in known_identifiers() or low in FIELD_WORDS or any(st in low for st in METRIC_STEMS))):
            return dict(label="Mistake", kind="Format", why=f"'{name}' (missing field, schema name or snapshot name, a vocabulary word in the wrong field, or a variant of a real metric; the idea is understood", tag="")
        return dict(label="Hallucination", kind="Requirement", why=f"'{name}' has no link to the schema, the snapshot or the intent", tag="Extrinsic")
    if base in ("spurious_constraint", "unsupported_extremum_tag", "wrong_extremum"):
        return dict(label="Hallucination", kind="Requirement", why=base, tag="Extrinsic" if full in EXTRINSIC else "Intrinsic")
    if base == "wrong_direction":
        gs = _gold_like(rec, k)
        t = (slo or {}).get("target") or {}
        if gs and t.get("op") == gs[0][3] and _close(t.get("value"), gs[0][4]):
            return dict(label="Mistake", kind="Format", why="op and value are right; only the comparator is reversed", tag="")
        return dict(label="Hallucination", kind="Requirement", why=base, tag="Intrinsic")
    if base == "missing_extremum_tag":
        if "wrong_value" in sub:
            return dict(label="Hallucination", kind="Value", why="qualifier dropped and the number is wrong", tag="Intrinsic")
        return dict(label="Mistake", kind="Convention", why="qualifier dropped, number equals gold", tag="")
    if base in ("boundary_strictness", "vacuous_placement", "unit_label"):
        return dict(label="Mistake", kind="Convention", why=base, tag="")
    if base == "other_scope_real_extremum":
        if row["index"] in scope_conflict:
            return dict(label="Label error", kind="", why="gold uses the global extremum although the intent names a layer", tag="")
        return dict(label="Hallucination", kind="Value", why=base, tag="Intrinsic")
    if base == "unit_conversion":
        gs = [canon_slo(g) for g in rec["label"]["slos"] if canon_slo(g)[:4] == k[:4]]
        ratio = k[4] / gs[0][4] if gs and gs[0][4] else None
        if ratio and any(abs(ratio / c - 1) < 1e-3 for c in SI_BINARY):
            return dict(label="Mistake", kind="Convention", why="decimal GB (10^9) instead of 2^30", tag="")
        return dict(label="Hallucination", kind="Value", why="arithmetic slip in the unit conversion", tag="Intrinsic")
    if base in ("fabricated_value", "misgrounded_real_value", "wrong_metric_value", "wrong_explicit_value", "placement_value"):
        return dict(label="Hallucination", kind="Value", why=base, tag="Extrinsic" if full in EXTRINSIC else "Intrinsic")
    if base in ("abstention_null_lookup", "missing_value"):
        return dict(label="Mistake", kind="Abstention", why="no value asserted", tag="")
    return dict(label="Mistake", kind="Format", why=base, tag="")
