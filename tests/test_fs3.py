"""fs3 prompt: zs3 system text unchanged, three invented examples with recomputed outputs, (the leak test is released with the archive)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import fewshot3, data_io as D  # noqa: E402
from src.parse import parse_response  # noqa: E402
from src.prompts import load_fs_prompt, render_user  # noqa: E402


def test_system_text_is_zs3_unchanged():
    zs3_system, _ = load_fs_prompt("zs3_prompt.txt")
    fs3_system, _ = load_fs_prompt("fs3_prompt.txt")
    assert fs3_system == zs3_system


def test_committed_file_matches_generator():
    assert (ROOT / "prompts" / "fs3_prompt.txt").read_text(encoding="utf-8") == fewshot3.file_text()


def test_three_examples_in_order_valid_invalid_valid():
    exs = fewshot3.examples()
    assert [e["key"] for e in exs] == ["V-explicit", "I-conflict", "V-mixed"]
    assert [bool(e["slos"]) for e in exs] == [True, False, True]
    _, user = load_fs_prompt("fs3_prompt.txt")
    assert user.count("\nOutput:\n") == 3 and re.findall(r"^Example \d$", user, re.M) == ["Example 1", "Example 2", "Example 3"]
    assert user.rstrip().endswith("Answer:")


def test_example_values_recomputed():
    v1, i1, v2 = fewshot3.examples()
    t = {s["metric"]: s["target"] for s in v1["slos"]}
    assert t["cpu_capacity"]["value"] == 6000 and t["memory_capacity"]["value"] == 24 * 2 ** 30 == 25769803776
    assert all(s["target"]["op"] is None for s in v1["slos"]) and not any(s["type"] == "placement_location" for s in v1["slos"])
    assert i1["slos"] == []
    by = {s["metric"]: s["target"] for s in v2["slos"] if s["metric"]}
    assert v2["slos"][0]["target"] == {"cmp": None, "op": "set", "value": "edge"}
    assert by["cpu_utilization"] == {"cmp": "<=", "op": None, "value": 33}
    assert by["memory_utilization"] == {"cmp": "<=", "op": "lowest", "value": 28.2}
    assert by["storage_utilization"] == {"cmp": ">=", "op": "highest", "value": 35.0}
    # the edge extremum differs from the global one, and the all-null edge node is skipped
    nodes = fewshot3.SNAPSHOT["nodes"]
    assert min(n["utilization_now"]["memory_util_percent"] for n in nodes if n["utilization_now"]["memory_util_percent"] is not None) == 19.6
    assert max(n["utilization_now"]["storage_util_percent"] for n in nodes if n["utilization_now"]["storage_util_percent"] is not None) == 48.0
    assert any(n["layer"] == "edge" and n["utilization_now"]["cpu_util_percent"] is None for n in nodes)


def test_rendered_outputs_are_valid_json_and_empty_answer_is_compact():
    for ex in fewshot3.examples():
        obj = json.loads(fewshot3.output_text(ex))
        assert obj["service"] == ex["service"] and obj["slos"] == ex["slos"]
        assert parse_response(fewshot3.output_text(ex))["parse_ok"]
    assert '"slos": []' in fewshot3.render(fewshot3.examples()[1])


def test_final_input_placeholders_render():
    _, user = load_fs_prompt("fs3_prompt.txt")
    rec = D.load_records()[0]
    snap = D.snapshot_compact(D.load_snapshot())
    intent = rec["input"]["intent"]
    text = render_user(user, intent["service"], intent["expression"], snap)
    assert "{service}" not in text and "{expression}" not in text and "{snapshot}" not in text
    assert intent["expression"] in text




def test_fs3_cot_is_fs3_plus_the_zs3_cot_trigger():
    zs3_sys, zs3_user = load_fs_prompt("zs3_prompt.txt")
    _, zs3_cot_user = load_fs_prompt("zs3_cot_prompt.txt")
    sys_c, user_c = load_fs_prompt("fs3_cot_prompt.txt")
    sys_f, user_f = load_fs_prompt("fs3_prompt.txt")
    trigger = zs3_cot_user[len(zs3_user):]
    assert "Let's think step-by-step." in trigger and "<think>" in trigger
    assert sys_c == sys_f == zs3_sys
    assert user_c == user_f + trigger
    assert (ROOT / "prompts" / "fs3_cot_prompt.txt").read_text(encoding="utf-8") == fewshot3.cot_file_text()
