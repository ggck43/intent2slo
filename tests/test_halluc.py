import json

from src.data_io import load_records, load_snapshot
from src.halluc import classify
from src.score import canon_slo

SNAP = load_snapshot()
T = lambda typ, metric, cmp_, op, val, unit: {"type": typ, "metric": metric, "target": {"cmp": cmp_, "op": op, "value": val}, "unit": unit}
S, D, P = "static_compute_resource", "dynamic_compute_resource", "placement_location"


def flags(pred, gold_raw, status="valid"):
    gold = [canon_slo(g) for g in gold_raw]
    return [f for f in classify(pred, gold, status, SNAP)]


def test_gold_label_has_no_flags_for_every_valid_record():
    for r in load_records():
        if r["meta"]["status"] == "valid":
            for s in r["label"]["slos"]:
                assert flags(s, r["label"]["slos"]) == [], r["index"]


def test_each_category_on_a_controlled_mutation():
    gold = [T(S, "memory_capacity", None, ">=", 2147483648, "bytes"), T(P, None, None, "set", "edge", None)]
    assert flags(T(S, "memory_capacity", None, ">=", 2147483648, "bytes"), gold) == []
    assert flags(T(S, "memory_capacity", None, ">=", 1073741824, "bytes"), gold) == [("H2", "wrong_explicit_value")]
    assert flags(T(S, "memory_capacity", None, ">=", 2000000000, "bytes"), gold)[0][0] == "H2"
    assert flags(T(D, "storage_utilization", "<=", None, 50, "percent"), gold) == [("H1", "spurious_constraint")]
    assert flags(T(S, "memory_capacity", None, "<=", 2147483648, "bytes"), gold) == [("H1", "wrong_direction")]
    assert flags(T(D, "latency", "<=", None, 5, "ms"), gold) == [("H3", "unknown_metric")]
    assert flags(T(P, None, None, "set", "any", None), [gold[0]]) == [("H1", "vacuous_placement")]


def test_invalid_intent_any_slo_is_h1_and_empty_is_clean():
    assert flags(T(S, "cpu_capacity", None, ">=", 1000, "millicores"), [], "conflicting") == [("H1", "accepted_invalid:conflicting")]


def test_omission_is_not_a_hallucination_and_value_fault_is_counted_once():
    gold = [T(S, "memory_capacity", None, ">=", 2147483648, "bytes")]
    assert flags(T(S, "memory_capacity", None, ">=", "a few GB", "bytes"), gold) == [("H3", "value_not_numeric")]


def test_unit_fault_is_not_counted_twice_and_strictness_does_not_hide_a_null_value():
    gold = [T(S, "memory_capacity", None, ">=", 2147483648, "bytes")]
    assert flags(T(S, "memory_capacity", None, ">=", 2147483648, "furlongs"), gold) == [("H3", "unit_outside_domain")]
    gold2 = [T(D, "memory_utilization", "<=", "lowest", 34.27, "percent")]
    got = flags(T(D, "memory_utilization", "<", "lowest", None, "percent"), gold2)
    assert ("H1", "boundary_strictness") in got and ("H2", "abstention_null_lookup") in got
