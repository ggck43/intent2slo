"""Policy v1 label rules (src/halluc_verdict.py) on small hand-made cases."""
from src.halluc_verdict import judge


def rec(expr, gold):
    return {"input": {"intent": {"expression": expr}}, "label": {"slos": gold}}


def slo(metric, cmp_, op, value, unit, typ="static_compute_resource"):
    return {"type": typ, "metric": metric, "target": {"cmp": cmp_, "op": op, "value": value}, "unit": unit}


GOLD_LOW = slo("memory_capacity", "<=", "lowest", 100.0, "bytes")


def row(sub, s, idx=1, status="valid"):
    from src.score import canon_slo
    return dict(sub=sub, canon=canon_slo(s), status=status, index=idx, slo=s)


def test_reversed_comparator_with_right_op_and_value_is_a_mistake():
    s = slo("memory_capacity", ">=", "lowest", 100.0, "bytes")
    assert judge(row("wrong_direction+self_inconsistent", s), rec("lowest memory", [GOLD_LOW]), set())["label"] == "Mistake"


def test_wrong_direction_with_wrong_op_is_a_hallucination():
    s = slo("memory_capacity", ">=", "highest", 100.0, "bytes")
    assert judge(row("wrong_direction", s), rec("lowest memory", [GOLD_LOW]), set())["label"] == "Hallucination"


def test_scope_conflict_record_is_a_label_error_and_null_is_abstention():
    s = slo("memory_capacity", "<=", "lowest", 200.0, "bytes")
    assert judge(row("other_scope_real_extremum", s, idx=7), rec("edge lowest memory", [GOLD_LOW]), {7})["label"] == "Label error"
    n = slo("memory_capacity", "<=", "lowest", None, "bytes")
    assert judge(row("abstention_null_lookup", n), rec("lowest memory", [GOLD_LOW]), set())["kind"] == "Abstention"


def test_decimal_gb_is_a_mistake_and_a_mixed_slip_is_a_hallucination():
    g = slo("memory_capacity", ">=", None, 2 ** 30 * 8.0, "bytes")
    dec = slo("memory_capacity", ">=", None, 8e9, "bytes")
    slip = slo("memory_capacity", ">=", None, 8000 * 2 ** 20 * 1.0, "bytes")
    assert judge(row("unit_conversion", dec), rec("8 GB", [g]), set())["label"] == "Mistake"
    assert judge(row("unit_conversion", slip), rec("8 GB", [g]), set())["label"] == "Hallucination"


def test_accepted_invalid_is_judged_by_text_support():
    p = {"type": "placement_location", "metric": None, "target": {"cmp": None, "op": "set", "value": "cloud"}, "unit": None}
    assert judge(row("accepted_invalid:conflicting", p, status="conflicting"), rec("only on the cloud and only on the edge", []), set())["label"] == "Mistake"
    q = {"type": "placement_location", "metric": None, "target": {"cmp": None, "op": "set", "value": "edge"}, "unit": None}
    assert judge(row("accepted_invalid:ambiguous", q, status="ambiguous"), rec("somewhere reasonable", []), set())["label"] == "Hallucination"
