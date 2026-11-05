"""Step 1 of plans/20261026.hallucinationAnalysis.md: one row per response (40 conditions x 716 = 28,640) for notebook 19.
Labels come from a constraints file in outputs/hallucination/ (default policy_v2j_constraints.csv: one row per FAULTY distinct constraint).
Writes outputs/hallucinationAnalysis/<tag>/responses.csv; refuses to overwrite a tag.
Usage: python3 scripts/halluc19_responses.py --tag v2j [--constraints policy_v2j_constraints.csv]"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd  # noqa: E402

from halluc_workload import FAMILIES, SETUPS, find, read, recs  # noqa: E402
from src.data_io import template_id  # noqa: E402
from src.halluc_analysis import scope_conflict_records  # noqa: E402
from src.score import canon_slo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--constraints", default="policy_v2j_constraints.csv")
    a = ap.parse_args()
    out = ROOT / "outputs/hallucinationAnalysis" / a.tag
    if out.exists():
        sys.exit(f"{out} exists; pick a new --tag")
    out.mkdir(parents=True)
    F = pd.read_csv(ROOT / "outputs/hallucination" / a.constraints).fillna("")
    SC = scope_conflict_records()
    g = {k: v for k, v in F.groupby(["model", "setup", "index"])}
    rows = []
    for fam, toks in FAMILIES.items():
        for tok in toks:
            for setup, pre in SETUPS.items():
                for prefix in (pre, pre + "_invalid"):
                    d = find(prefix, tok)
                    for p in read(d):
                        r = recs[p["index"]]
                        status = r["meta"]["status"]
                        gold = Counter(canon_slo(x) for x in r["label"]["slos"])
                        ok = bool(p.get("parse_ok"))
                        pred = Counter(canon_slo(s) for s in p["parsed"]["slos"]) if ok else Counter()
                        f = g.get((tok, setup, p["index"]))
                        f = f[f.constraint != ""] if f is not None else None
                        nh = int((f.label == "Hallucination").sum()) if f is not None else 0
                        nl = int((f.label == "Label error").sum()) if f is not None else 0
                        nm = int((f.label == "Mistake").sum()) if f is not None else 0
                        om = sum((gold - pred).values()) if status == "valid" else 0
                        rows.append(dict(family=fam, model=tok, setup=setup, run=d.name, index=p["index"], status=status, validity="valid" if status == "valid" else "invalid",
                                         parsed=ok, n_pred=len(pred), rejected=ok and len(pred) == 0, n_hall=nh, n_label=nl, n_mistake=nm, omissions=om,
                                         has_hall=nh > 0, has_label=nl > 0,
                                         has_error=(not ok) or nh + nl + nm > 0 or om > 0 or (status != "valid" and len(pred) > 0),
                                         in59=p["index"] in SC, level=r["meta"]["level"], source=r["meta"]["source"], n_gold=len(r["label"]["slos"]), template=template_id(r)))
    R = pd.DataFrame(rows)
    R.to_csv(out / "responses.csv", index=False)
    print(len(R), R.groupby("validity").size().to_dict(), "unparsed", int((~R.parsed).sum()))


if __name__ == "__main__":
    main()
