import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.energy_analysis import cv_cost, fit_cost, integrate, pareto_front, steady_state  # noqa: E402


def test_integrate_constant_power():
    t = np.arange(0, 10, .25); tr = pd.DataFrame({"t_epoch": t, "power_W": 200.0})
    assert abs(integrate(tr, 2, 6) - 800) < 1e-6


def test_cost_fit_recovers_coefficients():
    rng = np.random.RandomState(1); n = 200
    df = pd.DataFrame({"uncached_prompt_tokens": rng.randint(500, 6000, n), "completion_tokens": rng.randint(20, 1500, n)})
    df["energy_J"] = 20 + 0.02 * df.uncached_prompt_tokens + 0.3 * df.completion_tokens
    coef, r2 = fit_cost(df)
    assert np.allclose(coef, [20, 0.02, 0.3], atol=1e-6) and r2 > 0.999
    assert cv_cost(df)["cv_r2"] > 0.999


def test_steady_state_constant_load():
    # 16 requests of 1 s each on a constant 300 W GPU, N=2 -> 150 J per request (two share the GPU)
    t = np.arange(-1, 12, .25); tr = pd.DataFrame({"t_epoch": t, "power_W": 300.0})
    req = pd.DataFrame({"t_start": np.repeat(np.arange(8) * 1.0, 2), "t_end": np.repeat(np.arange(8) * 1.0 + 1.0, 2)})
    win = {"concurrency": 2, "t_window_start": 0.0, "t_window_end": 8.0, "window_energy_J": 2400.0}
    r = steady_state(tr, req, win)
    assert abs(r["whole_window_J"] - 150) < 1e-6 and abs(r["steady_J"] - 150) < 10


def test_pareto():
    assert pareto_front([(1, .5), (2, .9), (3, .8), (0.5, .4)]) == [0, 1, 3]


def test_cost2():
    from src.energy_analysis import cv_cost2, fit_cost2
    rng = np.random.RandomState(2); n = 120
    df = pd.DataFrame({"completion_tokens": rng.randint(20, 1500, n)}); df["energy_J"] = 40 + 0.5 * df.completion_tokens
    coef, r2 = fit_cost2(df)
    assert np.allclose(coef, [40, 0.5], atol=1e-6) and cv_cost2(df)["cv_r2"] > 0.999
