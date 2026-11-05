# Robustness benchmark: sample of 10 intents with 3 rephrasings each

`robust_benchmark_sample.json` and `robust_benchmark_sample.csv` hold **30 rows: 10 intents of the dataset, each with its three rephrasings C1, C2 and C3**. They are a sample of the full robustness benchmark of the paper (RQ5), which has **2,148 variants (716 intents x 3)**.

> **The full benchmark is available on request. Please write to sabbir.ahmed@queensu.ca.**

## What is in the file
| Field | Meaning |
|---|---|
| `parent_index` | Index of the original intent in the source dataset. |
| `source` | `real` or `synthetic` (as in the source dataset). |
| `validity` | `valid` or `invalid`. |
| `invalid_category` | For invalid intents: `ambiguous`, `conflicting`, `malformed` or `unsupported`; otherwise empty. |
| `level` | For valid intents: complexity level 1 to 5 (number and mix of SLOs); empty for invalid intents. |
| `variant_type` | `C1`, `C2` or `C3` (below). |
| `variant_text` | The rephrased intent. |
| `original_expression` | The original intent text of the source dataset. |
| `ground_truth_slos` | The ground-truth SLOs of the original intent (an empty list for an invalid intent). Every variant keeps them unchanged. |
| `generated_by` | The model that wrote the variant text (`claude-sonnet-5-5`). |

The CSV file holds `ground_truth_slos` as JSON text in one column.

## The three variants
- **C1, wrapped request.** Keeps the request close to the original, or lightly reworded, and wraps it with task-related opening and closing sentences.
- **C2, restructured request.** Rewords the request, reorders its constraints and adds one sentence on the service's role.
- **C3, noisy list.** Turns the request into a list with typically 3 to 5 typos and common shortcuts (for example "pls", "mem"), followed by a short closing.

The rewriting rules preserve the meaning: numbers, units, service names and layer words are never changed by a typo. Added sentences contain no numbers, metrics or hints about validity, and every invalid intent keeps its defect (for example, a malformed fragment stays broken). Each variant therefore keeps the ground-truth SLO set of its original intent.

## How the texts were made
The variants were written one intent at a time with LLM assistance (Claude Sonnet 5.5), following the definitions above. Automatic tests check that every variant keeps the numbers, units, service, layer and superlative words of its original, and that the added sentences are unique and free of digits.

## How the sample was chosen
The 10 intents were drawn at random with a fixed seed: five valid intents, one per complexity level (level 3 is the example shown in Table I of the paper), and five invalid intents (one per category, plus one more malformed intent with a different wording). Five of the ten are real intents and five are synthetic, as labelled in `source`.

## Licence and credit
The ground truth and the original intent texts come from the dataset of Islam and Calheiros (Zenodo, doi:10.5281/zenodo.20810799; CC BY 4.0 and Apache 2.0). The rephrased texts are released under CC BY 4.0 (`LICENSE-DATA.txt`); please credit both the dataset and this repository.

- Islam, K., Calheiros, R. N. *Natural-Language Intent Translation for Intent-Driven Orchestration in the Compute Continuum Dataset*, version 2, Zenodo. doi:10.5281/zenodo.20810799.
- Islam, K., Calheiros, R. N. *Intent Engine: Natural-language Intent Translation for Intent-Driven Orchestration in the Compute Continuum*. Journal of Systems Architecture 179 (2026) 103938. doi:10.1016/j.sysarc.2026.103938.
- This paper: to be added.
