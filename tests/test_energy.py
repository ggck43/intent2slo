import numpy as np

from src.data_io import load_records
from src.energy import cache_friendly, fit_cost, stratified_sample
from src.prompts import load_fs_prompt, load_user_template, render_user


def test_sample_is_four_per_level_valid_and_reproducible():
    recs = load_records()
    s = stratified_sample(recs)
    assert len(s) == 20 and all(r["meta"]["status"] == "valid" for r in s)
    assert [sum(r["meta"]["level"] == lv for r in s) for lv in range(1, 6)] == [4] * 5
    assert [r["index"] for r in s] == [r["index"] for r in stratified_sample(recs)]




def test_fit_cost_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    u, g = rng.integers(100, 6000, 200), rng.integers(10, 4000, 200)
    f = fit_cost(u, g, 5 + 0.01 * u + 1.5 * g)
    assert abs(f["c_J_per_output_token"] - 1.5) < 1e-6 and abs(f["b_J_per_input_token"] - 0.01) < 1e-6 and f["r2"] > 0.999999


def test_nb18_samples():
    from src.data_io import load_records
    from src.energy import invalid_sample, loop_sample, n1_sample
    recs = load_records()
    a = n1_sample(recs, 12)
    assert len(a) == 60 and all(r["meta"]["status"] == "valid" for r in a)
    used = {r["index"] for r in a}
    b = loop_sample(recs, 256, used)
    assert len(b) == 256 and len({r["index"] for r in b}) == 256 and not ({r["index"] for r in b} & used)
    c = loop_sample(recs, 96, used)
    assert len(c) == 96
    inv = invalid_sample(recs, 10)
    assert len(inv) == 40 and all(r["meta"]["status"] != "valid" for r in inv)
