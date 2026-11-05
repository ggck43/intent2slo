"""Policy v2 labels on the 40 conditions: v1 rules plus meaning-first reading of format-broken constraints in valid intents.
Writes outputs/hallucination/policy_v2_constraints.csv (v1 files are not touched). Invalid-intent constraints keep the v1 heuristic
label as a PROPOSAL (column heur_label); the judged labels are merged later from the judge file.
Run: python3 scripts/halluc_workload_v2.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd  # noqa: E402

from halluc_workload import FAMILIES, SETUPS, SC, find, read, recs, snap  # noqa: E402
from src.halluc import classify  # noqa: E402
from src.halluc_v2 import STRUCT, repair  # noqa: E402
from src.halluc_verdict import judge  # noqa: E402
from src.score import canon_slo  # noqa: E402

RANK = {"Hallucination": 0, "Label error": 1, "Mistake": 2}


def judged(slo, gold, status, idx, rec):
    c = canon_slo(slo)
    return [(f, sub, judge(dict(sub=sub, canon=c, status=status, index=idx, slo=slo), rec, SC)) for f, sub in classify(slo, gold, status, snap)]


def label_constraint(sl, gold, status, idx, rec):
    """Returns (res, repairs) where res = [(flag, sub, verdict)]; v2 reads structural faults of valid intents by meaning."""
    res = judged(sl, gold, status, idx, rec)
    if status != "valid" or not any(sub in STRUCT for _, sub, _ in res):
        return res, []
    rep, done = repair(sl)
    if rep is None:
        return res, []
    sem = judged(rep, gold, "valid", idx, rec)
    if any(sub in STRUCT for _, sub, _ in sem):  # still unreadable after the repair
        return res, []
    surface = [x for x in res if x[1] in STRUCT]
    return surface + sem, done


def main():
    rows = []
    for fam, toks in FAMILIES.items():
        for tok in toks:
            for setup, pre in SETUPS.items():
                for prefix in (pre, pre + "_invalid"):
                    d = find(prefix, tok)
                    for p in read(d):
                        r = recs[p["index"]]
                        status = r["meta"]["status"]
                        gold = [canon_slo(g) for g in r["label"]["slos"]]
                        base = dict(family=fam, model=tok, setup=setup, index=p["index"], status=status)
                        if not p.get("parse_ok"):
                            rows.append({**base, "flag": "H3", "sub": "unparseable", "all_subs": "unparseable", "label": "Mistake", "kind": "Format", "why": "unparseable", "tag": "", "constraint": "", "repairs": "", "heur_label": ""})
                            continue
                        seen = set()
                        for sl in p["parsed"]["slos"]:
                            c = canon_slo(sl)
                            if c in seen:
                                continue
                            seen.add(c)
                            res, done = label_constraint(sl, gold, status, p["index"], r)
                            res = [x for x in res if not (status != "valid" and not x[1].startswith("accepted"))]
                            if not res:
                                continue
                            f, sub, j = sorted(res, key=lambda x: RANK[x[2]["label"]])[0]
                            rows.append({**base, "flag": f, "sub": sub, "all_subs": "|".join(x[1] for x in res), **j, "constraint": json.dumps(sl, sort_keys=True),
                                         "repairs": ";".join(done), "heur_label": j["label"] if status != "valid" else ""})
    F = pd.DataFrame(rows)
    F.to_csv(ROOT / "outputs/hallucination/policy_v2_constraints.csv", index=False)
    print(len(F))


if __name__ == "__main__":
    main()
