"""The tests read the dataset and the node snapshot from data/raw/. Download the two files of the Zenodo record
(doi:10.5281/zenodo.20810799) into that folder first; see data/README.md."""
from pathlib import Path

import pytest

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = ("dataset.json", "snapshot_2026-04-25T23-12-00.393306Z.json")


def pytest_configure(config):
    missing = [n for n in FILES if not (RAW / n).exists()]
    if missing:
        raise pytest.UsageError(
            f"missing {', '.join(missing)} in data/raw/. Download the two files of the dataset "
            "(Zenodo, doi:10.5281/zenodo.20810799) into data/raw/ before running the tests; see data/README.md.")
