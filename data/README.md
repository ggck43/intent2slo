# Data

## Source dataset (not copied here)
The intents, the node snapshot and the ground-truth SLOs come from the dataset of Islam and Calheiros:

> Islam, K., Calheiros, R. N. *Natural-Language Intent Translation for Intent-Driven Orchestration in the Compute Continuum Dataset*, version 2, Zenodo. doi:10.5281/zenodo.20810799 (licences listed on the record: CC BY 4.0 and Apache 2.0).

Download the two files of the record, `dataset.json` and `snapshot_2026-04-25T23-12-00.393306Z.json`, into `data/raw/` (this folder is ignored by Git). The code reads them from there. The dataset has 716 intents (521 valid, 195 invalid: 65 ambiguous, 49 conflicting, 35 malformed, 46 unsupported) and one snapshot of a 6-node edge and cloud cluster.

The dataset is described in: Islam, K., Calheiros, R. N. *Intent Engine: Natural-language Intent Translation for Intent-Driven Orchestration in the Compute Continuum*. Journal of Systems Architecture 179 (2026) 103938. doi:10.1016/j.sysarc.2026.103938.

## Robustness benchmark sample (`robust/`)
A sample of 10 intents with their three rephrasings each (30 rows). The full benchmark (2,148 variants) is available on request. See [`robust/README.md`](robust/README.md).
