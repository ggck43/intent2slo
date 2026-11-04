"""Tests for the zs3 prompt files (plans/20261025.zs3_promptDesign.md)."""
import difflib
import re

from src.prompts import PROMPT_DIR, load_fs_prompt, load_system, load_user_template

TRIGGER = ("Let's think step-by-step. Write your reasoning inside <think> and </think>, then give the final answer after </think> "
           "as one JSON object matching the schema above.")




def test_system_never_mentions_think_and_has_no_dataset_wording():
    s, _ = load_fs_prompt("zs3_prompt.txt")
    assert "<think>" not in s and "think" not in s.lower()
    for w in ("carbon", "privacy", "compliance", "latency", "sensible", "enough", " good", "gdpr"):
        assert w not in s.lower(), w




def test_expected_blocks_and_values():
    s, _ = load_fs_prompt("zs3_prompt.txt")
    for k in ("Allowed cmp/op combinations:", "Placement:", "Rejection:", "cpu_capacity=millicores;", "1 GB = 1 GiB = 1024^3 = 1073741824 bytes"):
        assert k in s
    assert "make the final answer {\"service\": \"<service>\", \"slos\": []}" in s
    assert (PROMPT_DIR / "zs3_cot_prompt.txt").exists()
