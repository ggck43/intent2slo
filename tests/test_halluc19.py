import numpy as np, pandas as pd
import src.halluc19 as H


def _df():
    rows = []
    for t in range(6):
        for m in ("qwen35-4b", "qwen35-9b"):
            for k in range(2):
                rows.append(dict(name=H.NAME[m], template=f"t{t}", parsed=True, has_hall=(t + k + (m == "qwen35-9b")) % 3 == 0, validity="valid"))
    return pd.DataFrame(rows)


def test_boot_point_equals_plain_rate():
    R = _df(); B = H.Boot(R, B=200)
    r = B.rates(R, "name"); plain = H.rate(R, "name").rate
    assert np.allclose(r.rate.astype(float), plain.reindex(r.index))
    assert (r.lo <= r.rate).all() and (r.rate <= r.hi).all()


def test_diff_is_antisymmetric():
    R = _df(); B = H.Boot(R, B=200); a, b = "Qwen3.5-4B", "Qwen3.5-9B"
    d1, d2 = B.diff(R, "name", a, b), B.diff(R, "name", b, a)
    assert abs(d1["diff"] + d2["diff"]) < 1e-12 and abs(d1["lo"] + d2["hi"]) < 1e-12


def test_invalid_category_mapping():
    assert H.invalid_cat("names layer edge; the text does not") == "layer the text does not name"
    assert H.invalid_cat("cpu_utilization is not named in the text") == "metric the text does not name"
    assert H.invalid_cat("5 is not the lowest cpu_utilization of the requested scope") == "number not supported by the text or snapshot"
