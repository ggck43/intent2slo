"""Merge the labels of the invalid intents into the constraint file (the step between judge_invalid_v2.py and halluc19_responses.py).

Input : outputs/hallucination/policy_v2_constraints.csv   (scripts/halluc_workload_v2.py: rule labels for valid intents; the invalid rows still
                                                           carry the placeholder labels of the first rule set)
        outputs/hallucination/judge_invalid_v2.csv        (scripts/judge_invalid_v2.py: one label per distinct (intent, constraint) of the invalid intents)
Output: outputs/hallucination/policy_v2j_constraints.csv  (the label file used for every rate of RQ3)
Rows of valid intents are copied unchanged. For every row of an invalid intent that has a constraint, label, kind and why are taken from the judge file.
Rows for unparseable responses (no constraint) keep their label. Refuses to overwrite an existing output file.
Run: python3 scripts/merge_judged_labels.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "hallucination"


def main():
    target = OUT / "policy_v2j_constraints.csv"
    if target.exists():
        sys.exit(f"{target} exists; existing results are never overwritten")
    F = pd.read_csv(OUT / "policy_v2_constraints.csv", dtype=str).fillna("")
    J = pd.read_csv(OUT / "judge_invalid_v2.csv", dtype=str).fillna("").set_index(["index", "constraint"])
    inv = (F.status != "valid") & (F.constraint != "")
    keys = list(zip(F.loc[inv, "index"], F.loc[inv, "constraint"]))
    F.loc[inv, "label"] = [J.loc[k, "label"] for k in keys]
    F.loc[inv, "kind"] = [J.loc[k, "kind"] for k in keys]
    F.loc[inv, "why"] = [J.loc[k, "reason"] for k in keys]
    F.to_csv(target, index=False)
    print(len(F), "rows;", int(inv.sum()), "invalid-intent rows relabelled")


if __name__ == "__main__":
    main()
