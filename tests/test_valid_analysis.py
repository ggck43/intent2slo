"""Notebook 16 helpers: the 40 runs load, known totals are reproduced, the lenient repair does what it says."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.valid_analysis import MODELS, PROMPTS, lenient_canon, load_all, summarize  # noqa: E402
from src.score import canon_slo  # noqa: E402



@pytest.fixture(scope="module")
def R():
    """The 40 stored runs. They are released with the camera-ready archive, so these tests are skipped without them."""
    try:
        return load_all()
    except FileNotFoundError as e:
        pytest.skip(f"needs the run outputs (outputs/runs): {e}")


def test_forty_runs_with_521_records(R):
    assert len(R) == len(MODELS) * len(PROMPTS) == 40
    assert all(len(d["rows"]) == 521 for d in R.values())


def test_known_totals(R):
    assert round(summarize(R[("Qwen3.5-9B", "zs3")]["rows"])["F1"], 3) == 0.921
    assert round(summarize(R[("Ministral-3-8B", "fs3 CoT")]["rows"])["F1"], 3) == 0.920
    assert round(summarize(R[("Gemma4-E4B", "fs3 CoT")]["rows"])["F1"], 3) == 0.943


def test_lenient_repair_only_fixes_surface_form():
    ok = {"metric": "memory_utilization", "target": {"cmp": "<=", "op": None, "value": 50}, "type": "dynamic_compute_resource", "unit": "percent"}
    assert lenient_canon(ok) == canon_slo(ok)
    messy = {"metric": "dynamic_compute_resource.memory_util_percent", "target": {"cmp": "<", "op": None, "value": 50}, "type": "x", "unit": "%"}
    assert lenient_canon(messy) == canon_slo(ok)
    plc = {"metric": "layer", "target": {"cmp": "set", "op": None, "value": "edge"}, "type": "placement_location", "unit": None}
    assert lenient_canon(plc)[0] == "placement_location" and lenient_canon(plc)[4] == "edge"
    wrong = {"metric": "memory_utilization", "target": {"cmp": "<=", "op": None, "value": 60}, "type": "dynamic_compute_resource", "unit": "percent"}
    assert lenient_canon(wrong) != canon_slo(ok)
