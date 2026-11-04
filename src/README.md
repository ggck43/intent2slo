# `src/`: what each file does

All files are plain Python modules (run them from the repository root, e.g. `python -m src.score <run_dir>`). None of them calls a model, except through `scripts/run_zs_cot.py`.

Some comments and docstrings mention "notebook N" or "plans/...". These point to the authors' working notes, which are not part of this repository.

## Data, prompts, parsing, scoring (the core)
| File | Purpose |
|---|---|
| `data_io.py` | Read-only loaders for the dataset and the node snapshot in `data/raw/` (download both from Zenodo, see the top README). Also the template id used for the bootstrap. |
| `prompts.py` | Builds the chat messages and reads the prompt files in `prompts/`. |
| `parse.py` | Extracts the final answer from a model response: the last top-level JSON object with a list `slos` (handles code fences and braces inside strings). |
| `score.py` | The strict scorer (S0): exact match of type, metric, comparator, qualifier, value and unit; micro-F1, precision, recall, exact match, Jaccard, and the template-cluster bootstrap interval. Command line: `python -m src.score <run_dir>` (it rewrites `scores.json` in that run directory). |
| `reject.py` | Intent validation (RQ2): a response is rejected (empty `slos`), unparseable or accepted; rejection rates per invalid category. |
| `analysis.py` | Error-analysis helpers for comparing runs on valid intents (per-metric F1, error types, highest/lowest lookups). |

## Prompts for the four setups
| File | Purpose |
|---|---|
| `fewshot.py` | Helpers shared by the few-shot builders (example formatting, snapshot rendering). |
| `fewshot3.py` | Generates the `fs3` and `fs3_cot` prompt files from the `zs3` system text plus three invented examples. `python -m src.fewshot3` regenerates `prompts/fs3_prompt.txt` and `prompts/fs3_cot_prompt.txt`. |

## Reports over many runs (used by the analysis of RQ1 to RQ5)
| File | Purpose |
|---|---|
| `zs3_report.py` | Finds run directories by name, scores them, and summarises `zs3` and `zs3_cot` results (rejection, error taxonomy, scope-conflict records). |
| `fs3_report.py` | The same for `fs3` and `fs3_cot`, plus the conflict-kind breakdown. |
| `models_report.py` | Table helpers across all ten models. |
| `halluc.py` | Rules that flag emitted constraints against the intent, snapshot and schema (used by the report helpers). |
| `halluc_analysis.py` | Loads the predictions of every condition and aggregates the hallucination flags. |

## Analysis of one research question
| File | Purpose |
|---|---|
| `valid_analysis.py` | RQ1: builds the numbered result tables (V-01 to V-16) from the model predictions: F1 per model and setup, exact match, explicit versus implicit SLOs, error types. Needs the predictions of the camera-ready archive; the released CSVs in `rq1_performance/results/` are its output. |
| `invalid_analysis.py` | RQ2: builds the numbered result tables (I-01 to I-12) from the predictions on the 195 invalid intents: rejection rate per category, counts, effect of examples and reasoning, conflicts by kind, what the false accepts contain. Needs the predictions of the camera-ready archive; the released CSVs in `rq2_validation/results/` are its output. |

The modules for the later research questions (the hallucination labelling, energy, robustness) are added together with the matching RQ folder.
