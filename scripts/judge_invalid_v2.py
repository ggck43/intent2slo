"""Judge the constraints returned for INVALID intents (rule_v2.md section 5). Written 2026-10-26.

How it works. I read every distinct invalid intent text once (35 texts cover all 195 records) and wrote down what that text supports:
named layers, named metrics (family, kind, direction, numbers), named concepts outside the schema (carbon, privacy, ...), and kind hints.
This file holds that reading as SPEC. Each returned constraint is then read for what it asserts (placement, metric family and kind,
extremum or threshold direction, number) and compared with the SPEC of its own intent. Gold is not used. The v1 keyword heuristic
(`heur_label`) is not used and is only compared afterwards.
Writes outputs/hallucination/judge_invalid_v2.csv (one row per distinct (record, constraint)) with a one-line reason. Refuses to overwrite.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = sys.argv[:1] + sys.argv[1:]
import pandas as pd  # noqa: E402

from src.data_io import load_records, load_snapshot, metric_table  # noqa: E402
from src.halluc import _extrema  # noqa: E402
from src.halluc_v2 import canon_metric  # noqa: E402

OUT = ROOT / "outputs/hallucination"
snap = load_snapshot()
RECS = {r["index"]: r for r in load_records()}
ROWS = metric_table(snap)
NODES = {n: layer for n, layer, _ in ROWS}
CAP = {"cpu_capacity", "memory_capacity", "storage_capacity"}
KIND = {m: ("cap" if m in CAP else "util") for m in ("cpu_capacity", "memory_capacity", "storage_capacity", "cpu_utilization", "memory_utilization", "storage_utilization", "port_utilization")}
BOTH = {"cap", "util"}


def atom(fam, kinds=BOTH, ext=None, thr=None):
    """ext: {kind: {tags}}; thr: [(direction, number or None)]; with neither, the metric is only mentioned."""
    return dict(fam=fam, kinds=set(kinds), ext={k: set(v) for k, v in (ext or {}).items()}, thr=thr or [])


# ---- the reading of the 35 distinct intent texts ----
def VAGUE(hints=()) -> dict:
    return dict(layers=set(), atoms=[], concepts={}, kind_hints=set(hints))
UNS = lambda concepts: dict(layers=set(), atoms=[], concepts=concepts, kind_hints=set())
SPEC = [
    # malformed
    (r"below memory 50 and port maybe", dict(layers=set(), atoms=[atom("memory", thr=[("<=", 50)]), atom("port", {"util"})], concepts={}, kind_hints=set())),
    (r"cloud node best storage broken", dict(layers={"cloud"}, atoms=[atom("storage", ext={"cap": {"highest"}, "util": {"lowest"}})], concepts={}, kind_hints=set())),
    (r"put where cpu under and storage", dict(layers=set(), atoms=[atom("cpu", thr=[("<=", None)]), atom("storage")], concepts={}, kind_hints=set())),
    (r"edge cpu words broken", dict(layers={"edge"}, atoms=[atom("cpu")], concepts={}, kind_hints=set())),
    (r"edge strongest cpu lowest memory", dict(layers={"edge"}, atoms=[atom("cpu", {"cap"}, ext={"cap": {"highest"}}), atom("memory", ext={"cap": {"lowest"}, "util": {"lowest"}})], concepts={}, kind_hints=set())),
    (r"edge cloud highest memory words broken", dict(layers={"edge", "cloud"}, atoms=[atom("memory", ext={"cap": {"highest"}, "util": {"highest"}})], concepts={}, kind_hints=set())),
    (r"latency good maybe cpu if storage", dict(layers=set(), atoms=[atom("cpu"), atom("storage")], concepts={"latency": "any"}, kind_hints=set())),
    # conflicting
    (r"highest CPU capacity and the lowest CPU capacity", dict(layers=set(), atoms=[atom("cpu", {"cap"}, ext={"cap": {"highest", "lowest"}})], concepts={}, kind_hints=set())),
    (r"highest memory capacity and the lowest memory capacity", dict(layers=set(), atoms=[atom("memory", {"cap"}, ext={"cap": {"highest", "lowest"}})], concepts={}, kind_hints=set())),
    (r"CPU utilization below 40% and above 80%", dict(layers=set(), atoms=[atom("cpu", {"util"}, thr=[("<=", 40), (">=", 80)])], concepts={}, kind_hints=set())),
    (r"CPU utilization below 25% while also requiring it to stay above 80%", dict(layers={"edge", "cloud"}, atoms=[atom("cpu", {"util"}, thr=[("<=", 25), (">=", 80)])], concepts={}, kind_hints=set())),
    (r"storage utilization below 30% and above 70%", dict(layers=set(), atoms=[atom("storage", {"util"}, thr=[("<=", 30), (">=", 70)])], concepts={}, kind_hints=set())),
    (r"lowest storage utilization and the highest storage utilization", dict(layers={"edge", "cloud"}, atoms=[atom("storage", {"util"}, ext={"util": {"highest", "lowest"}})], concepts={}, kind_hints=set())),
    (r"only on the cloud and only on the edge|edge and in the cloud at the same time", dict(layers={"edge", "cloud"}, atoms=[], concepts={}, kind_hints=set())),
    # unsupported (concepts the schema cannot hold)
    (r"carbon footprint and a GDPR", UNS({"carbon": "lo", "footprint": "lo", "gdpr": "any", "compl": "any", "location": "any", "residency": "any"})),
    (r"latency, uptime, policy compliance", UNS({"latency": "any", "uptime": "hi", "polic": "any", "compl": "any", "carbon": "lo"})),
    (r"lowest energy use and the best privacy", UNS({"energy": "lo", "privacy": "any", "guarantee": "any"})),
    (r"best error rate, strongest compliance", UNS({"error": "lo", "rate": "lo", "compl": "any", "posture": "any", "cost": "lo", "monthly": "lo", "operating": "lo"})),
    (r"greenest node with the strongest privacy", UNS({"green": "any", "privacy": "any", "guarantee": "any"})),
    (r"lowest electricity tariff", UNS({"tariff": "lo", "electricity": "lo", "residency": "any", "data": "any", "guarantee": "any"})),
    (r"energy cost is lowest, the privacy", UNS({"energy": "lo", "cost": "lo", "privacy": "any", "posture": "any", "compl": "any", "story": "any"})),
    (r"sustainability score is highest", UNS({"sustainab": "hi", "score": "hi", "audit": "any", "posture": "any", "jurisdiction": "any"})),
    (r"carbon impact is minimal", UNS({"carbon": "lo", "impact": "lo", "regulat": "any", "posture": "any", "trust": "hi"})),
    (r"environmental impact is lowest", UNS({"environ": "lo", "impact": "lo", "governance": "any", "profile": "any"})),
    (r"carbon efficiency, regulatory compliance", UNS({"carbon": "hi", "efficiency": "hi", "regulat": "any", "compl": "any", "business": "lo", "risk": "lo"})),
    # ambiguous (no metric, no number, no layer in any of them; kind hints only)
    (r"best overall compromise across capacity, utilization", VAGUE(BOTH)),
    (r"enough capacity", VAGUE({"cap"})),
    (r"room to grow", VAGUE({"cap"})),
    (r"enough room", VAGUE({"cap"})),
    (r"enough headroom|good headroom", VAGUE(BOTH)),
    (r"too busy|pushing resources too hard|pushing anything too far", VAGUE({"util"})),
    (r"not too constrained", VAGUE(BOTH)),
    (r"healthy, balanced|looks right, stays comfortable", VAGUE()),
]


def spec_for(expr, svc):
    t = expr.replace(svc, "<s>")
    for pat, sp in SPEC:
        if re.search(pat, t):
            return sp
    raise KeyError(t)


# ---- extremum values ----
def ext_values(fam, kind, tag, layers):
    """snapshot values the text can ground for this metric and tag: layers named in the text, else all scopes"""
    out = []
    for m in (k for k in KIND if k.startswith(fam + "_") and (kind is None or KIND[k] == kind)):
        ex = _extrema(snap, m)
        for sc in (layers or {"all", "edge", "cloud"}) | ({"all"} if not layers else set()):
            if sc in ex and tag in ex[sc]:
                out.append(ex[sc][tag])
    return out


def nodes_with(fam, kind, tag, layers):
    res = set()
    for m in (k for k in KIND if k.startswith(fam + "_") and (kind is None or KIND[k] == kind)):
        for n, layer, mm in ROWS:
            v = mm.get(m)
            if v is None:
                continue
            for sc, e in _extrema(snap, m).items():
                if tag in e and abs(v - e[tag]) <= 1e-9 * max(1, abs(v)) and (sc == "all" or sc == layer):
                    res.add(n)
    return res


def close(a, b):
    return abs(a - b) <= 1e-4 * max(1.0, abs(b))


SCALES = {"bytes": [1], "B": [1], "KB": [1e3, 1024], "MB": [1e6, 2 ** 20], "GB": [1e9, 2 ** 30], "KiB": [1024], "MiB": [2 ** 20], "GiB": [2 ** 30],
          "cores": [1000], "millicores": [1], "percent": [1], "%": [1]}


def candidates(v, unit):
    """value read in schema units (bytes, millicores, percent) under the unit label, plus the raw number"""
    return {v} | {v * s for s in SCALES.get(unit, [1])}


def num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*%?\s*", v)
        return float(m.group(1)) if m else None
    return None


def judge(slo, sp):
    """returns (label, kind, reason)"""
    H, M = [], []  # (kind, reason)
    if not isinstance(slo, dict):
        return "Mistake", "Format", "not an object"
    t = slo.get("target") if isinstance(slo.get("target"), dict) else {}
    typ, metric, unit = slo.get("type"), slo.get("metric"), slo.get("unit")
    cmp_, op, val = t.get("cmp"), t.get("op"), t.get("value")
    layers = sp["layers"]
    tags = {x for x in (cmp_, op) if x in ("highest", "lowest")}
    ext = next(iter(tags), None)
    d = next((x for x in (cmp_, op) if x in ("<=", "<", ">=", ">")), None)
    d = {"<": "<=", ">": ">="}.get(d, d)
    low = str(metric).lower() if metric is not None else ""
    is_place_slot = metric is None or low in ("placement", "placement_location", "layer", "location", "placement_layer")
    mm = canon_metric(metric, unit)
    if mm is None and typ in ("static_compute_resource", "dynamic_compute_resource", "node_network_resource") and metric is None:
        mm = None
    fam = mm.split("_")[0] if mm else None
    kind = KIND.get(mm)
    # bare family or kind names
    if mm is None and low in ("memory", "storage", "cpu", "port"):
        fam = low
    # ---------- placement ----------
    if typ == "placement_location" and is_place_slot and ext is None and d is None:
        words = val if isinstance(val, list) else [val]
        if val is None or (isinstance(val, str) and not val.strip()):
            return "Mistake", "Format", "placement without a value (no claim)"
        for w in words:
            ws = str(w).lower()
            toks = set(re.findall(r"[a-z][a-z0-9\-]*", ws))
            lay = toks & {"edge", "cloud"}
            node = next((n for n in NODES if n.lower() == ws.strip()), None)
            if node:
                allowed = set()
                for a in sp["atoms"]:
                    for k, tg in a["ext"].items():
                        for x in tg:
                            allowed |= nodes_with(a["fam"], k, x, layers)
                if node in allowed:
                    M.append("node holds the requested extremum")
                else:
                    H.append(("Requirement", f"names node {node}, which the text does not name and which holds no requested extremum"))
            elif ws.strip() == "any" or (not lay and "both" not in toks):
                M.append("placement any/echo (no claim)")
            elif ws.strip() == "both" or ("both" in toks) or len(lay) == 2:
                if layers >= {"edge", "cloud"}:
                    M.append("both layers named in the text")
                else:
                    H.append(("Requirement", "places on both layers; the text names no such request"))
            else:
                l = next(iter(lay))
                ok = l in layers or (not layers and _implied_layer(sp, l))
                if ok:
                    M.append(f"layer {l} stated or implied")
                else:
                    H.append(("Requirement", f"names layer {l}; the text does not"))
        return _out(H, M)
    # ---------- metric constraints ----------
    placement_prefix = ""
    if typ == "placement_location" and not is_place_slot:
        M.append("metric under a placement type (surface)")
    if typ == "node_network_resource" and mm and mm != "port_utilization" and mm != "bandwidth":
        M.append("metric under a network type (surface)")
    if mm and unit not in SCALES and unit not in (None, "Mbps", "Gbps", "Kbps", "Bps"):
        M.append(f"unit {unit!r} outside the list (surface)")
    n = num(val)
    # concept outside the schema
    if mm is None and fam is None:
        if is_place_slot or low in ("utilization", "capacity"):
            pass
        name = low.replace(" ", "_")
        if low in ("utilization", "capacity"):
            kk = "util" if low == "utilization" else "cap"
            if kk in sp["kind_hints"]:
                M.append(f"{kk} kind named in the text; no family chosen")
                if n is not None:
                    H.append(("Value", f"number {n:g} not in the text"))
                return _out(H, M)
            H.append(("Requirement", f"names a {kk} metric; the text does not"))
            return _out(H, M)
        if metric is None or is_place_slot or re.fullmatch(r"(static_compute_resource|dynamic_compute_resource|node_network_resource)", low):
            # no metric slot filled: unreadable; judge the number
            if n is not None:
                pool = set()
                for a in sp["atoms"]:
                    pool |= {x[1] for x in a["thr"] if x[1] is not None}
                    for k, tg in a["ext"].items():
                        for x in tg:
                            pool |= set(ext_values(a["fam"], k, x, layers))
                if not any(close(c, p) for c in candidates(n, unit) for p in pool):
                    H.append(("Value", f"number {n:g} on an unreadable metric; not in the text or grounded"))
            M.append("metric not readable (surface)")
            return _out(H, M)
        stems = [s for s in sp["concepts"] if s in name]
        if stems:
            pol = sp["concepts"][stems[0]]
            if ext and pol != "any" and ext != {"lo": "lowest", "hi": "highest"}[pol]:
                H.append(("Requirement", f"{ext} on '{metric}', but the text wants the opposite"))
            else:
                M.append(f"'{metric}' echoes a concept of the text (outside the schema)")
            if n is not None:
                H.append(("Value", f"number {n:g} for '{metric}' is not in the text"))
            return _out(H, M)
        H.append(("Requirement", f"metric '{metric}' is not in the text, the schema or a stated concept"))
        return _out(H, M)
    # schema metric with a family
    if fam in ("bandwidth",) or fam is None:
        fam_atoms = []
    else:
        fam_atoms = [a for a in sp["atoms"] if a["fam"] == fam]
    if not fam_atoms:
        if fam is None:
            H.append(("Requirement", "metric family not readable and not stated"))
        else:
            H.append(("Requirement", f"{mm or fam} is not named in the text"))
        return _out(H, M)
    if kind is not None:
        fam_atoms = [a for a in fam_atoms if kind in a["kinds"]]
        if not fam_atoms:
            H.append(("Requirement", f"{mm}: the text does not name this kind ({kind}) of {fam}"))
            return _out(H, M)
    a = fam_atoms[0]
    kinds_try = [kind] if kind else sorted(a["kinds"])
    if ext:
        okt = any(ext in a["ext"].get(k, set()) for k in kinds_try)
        if not okt:
            if a["ext"]:
                H.append(("Requirement", f"{ext} on {mm or fam}; the text asks for the other direction or none"))
            else:
                H.append(("Requirement", f"{ext} on {mm or fam}; the text states no extremum"))
            return _out(H, M)
        if n is not None:
            vals = [v for k in kinds_try for v in ext_values(fam, k, ext, layers)]
            if not any(close(c, v) for c in candidates(n, unit) for v in vals):
                H.append(("Value", f"{n:g} is not the {ext} {mm or fam} of the requested scope"))
            else:
                M.append("grounded extremum value")
        elif val is None:
            M.append("value null (abstention)")
        if d and ((ext == "highest" and d == "<=") or (ext == "lowest" and d == ">=")):
            M.append("comparator reversed against the tag (surface)")
        return _out(H, M)
    # threshold or plain
    if d is not None:
        wanted = [x for x in a["thr"] if x[0] == d]
        if not a["thr"] and not a["ext"]:
            H.append(("Requirement", f"states a {d} bound on {mm or fam}; the text gives no bound"))
            return _out(H, M)
        if not a["thr"] and a["ext"]:
            # extremum request written as a bare threshold
            if n is not None:
                vals = [v for k in kinds_try for tg in a["ext"].get(k, ()) for v in ext_values(fam, k, tg, layers)]
                if any(close(c, v) for c in candidates(n, unit) for v in vals):
                    M.append("extremum value written as a threshold (qualifier dropped)")
                else:
                    H.append(("Value", f"{n:g} as a bound on {mm or fam}; it is not a number of the text or an extremum"))
            else:
                M.append("bound without number (abstention)")
            return _out(H, M)
        if not wanted:
            H.append(("Requirement", f"{d} bound on {mm or fam} reverses the direction of the text"))
            return _out(H, M)
        if n is not None:
            if wanted[0][1] is None or not any(close(c, wanted[0][1]) for c in candidates(n, unit)):
                H.append(("Value", f"number {n:g} is not in the text"))
            else:
                M.append("bound stated in the text")
        else:
            M.append("bound without number (abstention)")
        return _out(H, M)
    # no comparator, no tag
    if n is not None:
        pool = {x[1] for x in a["thr"] if x[1] is not None}
        for k in kinds_try:
            for tg in a["ext"].get(k, ()):
                pool |= set(ext_values(fam, k, tg, layers))
        if any(close(c, p) for c in candidates(n, unit) for p in pool):
            M.append("number from the text or grounded")
        else:
            H.append(("Value", f"number {n:g} is not in the text"))
        return _out(H, M)
    M.append(f"{mm or fam} mentioned in the text; no number, no direction")
    return _out(H, M)


def _implied_layer(sp, l):
    """a layer holding a node that has the requested extremum is implied by 'the node with the highest/lowest ...'"""
    for a in sp["atoms"]:
        for k, tg in a["ext"].items():
            for x in tg:
                if any(NODES[n] == l for n in nodes_with(a["fam"], k, x, set())):
                    return True
    return False


def _out(H, M):
    if H:
        kinds = [k for k, _ in H]
        return "Hallucination", ("Requirement" if "Requirement" in kinds else "Value"), "; ".join(r for _, r in H)
    surf = [m for m in M if "surface" in m or "not readable" in m]
    kind = "Format" if surf else ("Abstention" if any("abstention" in m or "null" in m for m in M) else
                                  "Convention" if any("qualifier dropped" in m for m in M) else "Accepted invalid (text-supported)")
    return "Mistake", kind, "; ".join(M)


def main():
    path = OUT / "judge_invalid_v2.csv"
    if path.exists():
        sys.exit("judge_invalid_v2.csv exists; pick a new file name (existing results are never overwritten)")
    F = pd.read_csv(OUT / "policy_v2_constraints.csv").fillna("")
    D = F[(F.status != "valid") & (F.constraint != "")].drop_duplicates(["index", "constraint"])
    rows = []
    for _, r in D.iterrows():
        rec = RECS[r["index"]]["input"]["intent"]
        sp = spec_for(rec["expression"], rec["service"])
        lab, kind, why = judge(json.loads(r.constraint), sp)
        rows.append(dict(index=r["index"], status=r.status, constraint=r.constraint, label=lab, kind=kind, reason=why, heur_label=r.heur_label))
    R = pd.DataFrame(rows)
    R.to_csv(path, index=False)
    print(len(R)); print(R.groupby(["label", "kind"]).size()); print(pd.crosstab(R.heur_label, R.label))


if __name__ == "__main__":
    main()
