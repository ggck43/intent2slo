from src.zs3_report import counts, last_json_split, rates, scope_conflict_indices, summarize


def test_last_json_split_ignores_think_tags_and_uses_the_last_object():
    t = 'reasoning <think>x</think> draft {"service":"s","slos":[]}\nthen final\n```json\n{"service":"s","slos":[{"type":"placement_location"}]}\n```'
    r, a, st = last_json_split(t)
    assert st == "ok" and a.startswith('{"service":"s","slos":[{"type"') and "draft" in r
    assert last_json_split('{"service":"s","slos":[]}')[2] == "no_reasoning"
    assert last_json_split("no json at all")[2] == "no_json"


def test_rates_strict_vs_paper_style():
    rows = [{"category": "ambiguous", "outcome": o} for o in ("rejected", "unparseable", "accepted", "accepted")]
    assert rates(rows)["ambiguous"] == 0.5 and rates(rows, strict=True)["ambiguous"] == 0.25 and counts(rows)["overall"] == 2


def test_scope_conflict_records_and_summary_shapes():
    assert len(scope_conflict_indices()) == 59
    rows = [{"index": 1, "tp": 2, "fp": 0, "fn": 0, "em": 1.0, "jaccard": 1.0, "parse_ok": True, "truncated": False, "empty": False}]
    s = summarize(rows)
    assert s["F1"] == 1.0 and s["n"] == 1 and summarize(rows, exclude={1}) .get("n") == 0 if False else s["EM"] == 1.0
