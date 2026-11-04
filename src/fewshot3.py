"""fs3 (v2, with two notes) = zs3 system prompt (byte-identical) + three invented examples (valid explicit, invalid, valid mixed).
fs3_cot = fs3 + the CoT trigger line of zs3_cot_prompt.txt (examples unchanged, i.e. they show bare JSON answers).

Design: plans/20261025.fs3_promptDesign.md. The helpers of src/fewshot.py are imported, not changed. All example
values (snapshot, services, wording) are invented; scripts/fs3_leak_test.py checks them against the dataset.
"""
import json
from pathlib import Path

from src.fewshot import node, explicit, implicit, placement, render_example, render_output, user_template, _j
from src.prompts import load_fs_prompt

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
ZS3_NAME = "zs3_prompt.txt"
FS3_NAME = "fs3_prompt.txt"
ZS3_COT_NAME = "zs3_cot_prompt.txt"
FS3_COT_NAME = "fs3_cot_prompt.txt"
SYSTEM_MARK, USER_MARK = "=== SYSTEM ===", "=== USER ==="

GB = 1024 ** 3

# one invented snapshot (real field names, reduced), repeated in every example: 3 edge nodes (one with null
# utilization, like `edge-node` in the real snapshot) and 2 cloud nodes
SNAPSHOT = {"nodes": [
    node("rack-a", "edge", 8000.0, 32 * GB, 22.4, 61.0, 35.0, 200 * GB, 0.310),
    node("rack-b", "edge", 6000.0, 16 * GB, 7.5, 28.2, 12.0, 100 * GB, 0.095),
    node("rack-c", "edge", 1000.0, 4 * GB),
    node("pool-x", "cloud", 16000.0, 64 * GB, 41.3, 55.8, 48.0, 500 * GB, 0.220),
    node("pool-y", "cloud", 12000.0, 32 * GB, 3.9, 19.6, 9.0, 300 * GB, 0.041),
]}


def examples():
    """Ordered list: valid (explicit only), invalid (conflict with readable parts), valid (mixed, edge)."""
    return [
        dict(key="V-explicit", service="svc/example-api", snapshot=SNAPSHOT,
             expression="svc/example-api requires 6 CPU cores and 24 GB of memory, with storage utilization capped at 72%.",
             slos=[explicit("cpu_capacity", ">=", 6 * 1000),
                   explicit("memory_capacity", ">=", 24 * GB),
                   explicit("storage_utilization", "<=", 72)]),
        dict(key="I-conflict", service="svc/example-batch", snapshot=SNAPSHOT,
             expression="svc/example-batch should run on the edge with at least 4 CPU cores, and the node needs at least 32 GB of memory but no more than 8 GB of memory.",
             slos=[]),
        dict(key="V-mixed", service="svc/example-feed", snapshot=SNAPSHOT,
             expression="Place svc/example-feed on edge hardware. Its CPU utilization should stay under 33%; among those nodes, take the one with the lowest memory utilization and the highest storage utilization.",
             slos=[placement("edge"),
                   explicit("cpu_utilization", "<=", 33),
                   implicit(SNAPSHOT, "memory_utilization", "lowest", layer="edge"),
                   implicit(SNAPSHOT, "storage_utilization", "highest", layer="edge")]),
    ]


def output_text(ex):
    if not ex["slos"]:
        return "{\n  \"service\": %s,\n  \"slos\": []\n}" % _j(ex["service"])
    return render_output(ex)


def render(ex):
    text = render_example(ex)
    return text if ex["slos"] else text.replace(render_output(ex), output_text(ex))


def system_text():
    return load_fs_prompt(ZS3_NAME)[0]


NOTE_A = ("The next three examples only show the pattern of how constraints are extracted. Their service names, node names and numbers "
          "are invented and are not part of your task. Never reuse them. Learn the pattern, not the names.")
NOTE_B = "End of examples. The real task follows. Use only the snapshot given in the real Input below."
_REAL = "\n\nInput:\n{\n  \"service\": \"{service}\""


def user_text():
    """Examples with two notes (added 2026-10-25 after the fs3b pilot): NOTE_A above the examples, NOTE_B before the real Input.
    The first version without notes is kept in prompts/archive/fs3_v1_*.txt."""
    u = user_template([render(ex) for ex in examples()])
    assert u.count(_REAL) == 1
    return NOTE_A + "\n\n" + u.replace(_REAL, "\n\n" + NOTE_B + _REAL)


def file_text():
    return f"{SYSTEM_MARK}\n{system_text()}\n{USER_MARK}\n{user_text()}\n"


def cot_suffix():
    """The CoT trigger of zs3_cot_prompt.txt (text appended to the zs3 user part), reused unchanged."""
    zs3_user = load_fs_prompt(ZS3_NAME)[1]
    zs3_cot_user = load_fs_prompt(ZS3_COT_NAME)[1]
    assert zs3_cot_user.startswith(zs3_user)
    return zs3_cot_user[len(zs3_user):]


def cot_file_text():
    return f"{SYSTEM_MARK}\n{system_text()}\n{USER_MARK}\n{user_text()}{cot_suffix()}\n"


def write(directory=PROMPT_DIR):
    paths = []
    for name, text in ((FS3_NAME, file_text()), (FS3_COT_NAME, cot_file_text())):
        path = Path(directory) / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    print(write())
