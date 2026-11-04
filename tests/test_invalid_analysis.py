import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.invalid_analysis import counts, load_invalid, conflict_halves, content_type  # noqa: E402



@pytest.fixture(scope="module")
def R():
    """The 40 stored invalid-intent runs. They are released with the camera-ready archive, so these tests are skipped without them."""
    try:
        return load_invalid()
    except FileNotFoundError as e:
        pytest.skip(f"needs the run outputs (outputs/runs): {e}")


def test_forty_runs_195_records(R):
    assert len(R) == 40 and all(len(d["rows"]) == 195 for d in R.values())


def test_known_counts(R):
    assert counts(R[("Ministral-3-8B", "fs3 CoT")]["rows"])["overall"] == 186
    assert counts(R[("Qwen3.5-9B", "zs3 CoT")]["rows"])["overall"] == 180
    assert counts(R[("Llama-3.2-3B", "zs3")]["rows"])["overall"] == 0


def test_rules():
    a = {"type": "static_compute_resource", "metric": "cpu_capacity", "target": {"cmp": ">=", "op": "highest", "value": 1}}
    b = {"type": "static_compute_resource", "metric": "cpu_capacity", "target": {"cmp": "<=", "op": "lowest", "value": 1}}
    assert conflict_halves("highest + lowest", [a, b]) == "both halves" and conflict_halves("highest + lowest", [a]) == "one half"
    assert content_type([a]) == "lookups only"
