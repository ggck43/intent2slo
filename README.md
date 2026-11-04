# Intent-to-SLO: lightweight language models for intent-based networking

> Draft outline. This README is built step by step; sections are filled as each research question is added.

Code and results for the ICC 2027 paper (link to be added). Ten open models of at most 10B parameters translate a network operator's intent into SLOs.

## Contents (planned)
1. Overview and reproducibility tiers (A rebuild tables and figures from released CSVs; B re-score from predictions, at camera-ready; C regenerate with the models, documented only)
2. Naming: the code uses `zs3`, `zs3_cot`, `fs3`, `fs3_cot`; the paper writes `zs`, `zs-cot`, `fs`, `fs-cot`
3. Data: the source dataset is linked, not copied
4. RQ1 accuracy, RQ2 intent validation, RQ3 hallucination, RQ4 energy and CO2e, RQ5 robustness
5. Installation, how to run, tests
6. Citation, licences, contact

## Licences
Code: Apache-2.0 (`LICENSE`). Data and tables created here: CC BY 4.0 (`LICENSE-DATA.txt`). See `NOTICE`.
