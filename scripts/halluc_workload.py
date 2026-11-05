"""Policy v1 on the final 40 conditions: 5 families x 2 sizes x {zs3, zs3 CoT, fs3, fs3 CoT}, valid + invalid runs.
Classifies every emitted constraint (src.halluc.classify), labels it (src.halluc_verdict.judge), and writes
outputs/hallucination/policy_v1_constraints.csv (one row per fault), policy_v1_workload.md (counts for the annotation plan) and
policy_v1_rates_preliminary.csv (rule-based, NOT audited). Run: python3 scripts/halluc_workload.py"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from src.data_io import load_records, load_snapshot  # noqa: E402
from src.halluc import classify  # noqa: E402
from src.halluc_analysis import scope_conflict_records  # noqa: E402
from src.halluc_verdict import judge  # noqa: E402
from src.score import canon_slo  # noqa: E402

FAMILIES = {"Qwen": ["qwen35-4b", "qwen35-9b"], "Gemma": ["gemma-4-e2b-it", "gemma-4-e4b-it"], "Ministral": ["ministral-3-8b-instruct-2512", "ministral-3-3b-instruct-2512"],
            "Granite": ["granite-41-8b", "granite-41-3b"], "Llama": ["llama-31-8b-instruct", "llama-32-3b-instruct"]}
SETUPS = {"zs3": "zs3", "zs3 CoT": "zs3_cot", "fs3": "fs3", "fs3 CoT": "fs3_cot"}
snap, recs, SC = load_snapshot(), {r["index"]: r for r in load_records()}, scope_conflict_records()


def find(prefix, tok):
    c = sorted(d for d in (ROOT / "outputs/runs").iterdir() if re.fullmatch(rf"{re.escape(prefix)}_{re.escape(tok)}_\d{{8}}T\d{{6}}", d.name))
    return c[-1] if c else None


def read(d):
    return [json.loads(x) for x in open(d / "predictions.jsonl") if x.strip()]


def main():
    rows, resp, missing = [], [], []
    for fam, toks in FAMILIES.items():
        for tok in toks:
            for setup, pre in SETUPS.items():
                for kind, prefix in (("valid", pre), ("invalid", pre + "_invalid")):
                    d = find(prefix, tok)
                    if d is None:
                        missing.append((prefix, tok))
                        continue
                    for p in read(d):
                        r = recs[p["index"]]
                        status = r["meta"]["status"]
                        gold = [canon_slo(g) for g in r["label"]["slos"]]
                        base = dict(family=fam, model=tok, setup=setup, index=p["index"], status=status)
                        n_h = n_m = n_l = 0
                        if not p.get("parse_ok"):
                            rows.append({**base, "flag": "H3", "sub": "unparseable", "all_subs": "unparseable", "label": "Mistake", "kind": "Format", "why": "unparseable", "tag": "", "constraint": ""})
                            n_m += 1
                        else:
                            seen = set()
                            for sl in p["parsed"]["slos"]:
                                c = canon_slo(sl)
                                if c in seen:
                                    continue
                                seen.add(c)
                                res = []
                                for f, sub in classify(sl, gold, status, snap):
                                    if status != "valid" and not sub.startswith("accepted"):
                                        continue  # schema faults of a returned constraint are judged once, by the accepted-invalid row
                                    res.append((f, sub, judge(dict(sub=sub, canon=c, status=status, index=p["index"], slo=sl), r, SC)))
                                if not res:
                                    continue
                                # one row per constraint: Hallucination beats Label error beats Mistake
                                win = sorted(res, key=lambda x: {"Hallucination": 0, "Label error": 1, "Mistake": 2}[x[2]["label"]])[0]
                                f, sub, j = win
                                rows.append({**base, "flag": f, "sub": sub, "all_subs": "|".join(x[1] for x in res), **j, "constraint": json.dumps(sl, sort_keys=True)})
                                n_h += j["label"] == "Hallucination"
                                n_m += j["label"] == "Mistake"
                                n_l += j["label"] == "Label error"
                        n_pred = len(p["parsed"]["slos"]) if p.get("parse_ok") else 0
                        resp.append({**base, "parsed": bool(p.get("parse_ok")), "n_pred": n_pred, "has_hall": n_h > 0, "has_mistake": n_m > 0, "has_label_error": n_l > 0,
                                     "rejection_failure": status != "valid" and n_pred > 0, "false_reject": status == "valid" and p.get("parse_ok") and n_pred == 0})
    F, R = pd.DataFrame(rows), pd.DataFrame(resp)
    F.to_csv(ROOT / "outputs/hallucination/policy_v1_constraints.csv", index=False)
    # unparseable responses are excluded from the rate denominators [D 2026-10-26]; their share is reported next to the rates
    g = R.groupby(["family", "model", "setup"])
    P = R[R.parsed]
    gp = P.groupby(["family", "model", "setup"])
    rates = pd.DataFrame({
        "valid_resp": g.apply(lambda x: (x.status == "valid").sum()), "invalid_resp": g.apply(lambda x: (x.status != "valid").sum()),
        "unparseable_rate": g.apply(lambda x: 1 - x.parsed.mean()),
        "hall_all": gp.apply(lambda x: x.has_hall.mean()), "hall_valid": gp.apply(lambda x: x[x.status == "valid"].has_hall.mean()),
        "hall_invalid": gp.apply(lambda x: x[x.status != "valid"].has_hall.mean()), "rejection_failure_invalid": g.apply(lambda x: x[x.status != "valid"].rejection_failure.mean()),
    }).round(3).reset_index()
    rates.to_csv(ROOT / "outputs/hallucination/policy_v1_rates_preliminary.csv", index=False)

    F["ck"] = F["constraint"]
    uniq = lambda d: d.drop_duplicates(["index", "ck"]).shape[0]
    L = ["# Policy v1 workload (40 conditions: 5 families x 2 sizes x 4 setups)\n",
         f"Generated by `scripts/halluc_workload.py`. {len(R)} responses, {len(F)} labelled constraint faults, {uniq(F)} distinct (record, constraint, sub-type) items. "
         f"Missing run directories: {len(missing)}. Labels are rule-based and NOT audited.\n",
         "## Faults by label and status\n", "| label | valid records | invalid records | distinct items |\n|---|---|---|---|"]
    for lab in ["Hallucination", "Mistake", "Label error"]:
        d = F[F.label == lab]
        L.append(f"| {lab} | {(d.status == 'valid').sum()} | {(d.status != 'valid').sum()} | {uniq(d)} |")
    L += ["\n## Distinct items per sub-type (the annotation strata)\n", "| sub-type | label | kind | faults | distinct items |\n|---|---|---|---|---|"]
    F["base"] = F["sub"].str.split("+").str[0]
    F.loc[F["sub"].str.startswith("accepted"), "base"] = "accepted_invalid:" + F["sub"].str.split(":").str[-1]
    for (b, lab, kd), d in sorted(F.groupby(["base", "label", "kind"]), key=lambda x: -uniq(x[1])):
        L.append(f"| `{b}` | {lab} | {kd} | {len(d)} | {uniq(d)} |")
    L += ["\n## Distinct items per family (for the transfer check)\n", "| family | distinct items | hallucination | mistake | label error |\n|---|---|---|---|---|"]
    for fam, d in F.groupby("family"):
        L.append(f"| {fam} | {uniq(d)} | {uniq(d[d.label == 'Hallucination'])} | {uniq(d[d.label == 'Mistake'])} | {uniq(d[d.label == 'Label error'])} |")
    (ROOT / "outputs/hallucination/policy_v1_workload.md").write_text("\n".join(L) + "\n")
    print(len(R), len(F), "missing:", missing)


if __name__ == "__main__":
    main()
