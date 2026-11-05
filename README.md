# intent2slo: how far can lightweight LLMs go in intent-to-SLO translation?

Code, result tables and documentation for a five-dimensional evaluation of **ten lightweight open language models** (at most 10B parameters, five families) on translating a network operator's natural-language placement intent into enforceable **service-level objectives (SLOs)**, grounded in a live snapshot of the cluster.

> **Paper:** Sabbir Ahmed, Hazem Abbas, Salimur Choudhury (School of Computing, Queen's University, Canada). *How Far Can Lightweight LLMs Go in Intent-to-SLO Translation?* Manuscript. A link will be added when the preprint is posted.
>
> **Status of reproducibility.** Every table and figure of the paper can be rebuilt from the result tables in this repository today (tier A below). The results are **fully reproducible once the camera-ready archive is released**: the model outputs, the raw energy measurements and the per-constraint labels are archived separately (see [What is not here yet](#what-is-not-here-yet)).

## The five research questions

| | Question | Folder | In the paper |
|---|---|---|---|
| RQ1 | How accurately do the models translate valid intents? | [`rq1_performance/`](rq1_performance/) | Table II, Fig. 2 |
| RQ2 | Can they refuse invalid intents? | [`rq2_validation/`](rq2_validation/) | Fig. 3 |
| RQ3 | How often do they invent unsupported SLOs? | [`rq3_hallucination/`](rq3_hallucination/) | Fig. 4, Table III, rulebook |
| RQ4 | What do they cost in energy and carbon? | [`rq4_energy/`](rq4_energy/) | Fig. 5 |
| RQ5 | Do they survive operator phrasing? | [`rq5_robustness/`](rq5_robustness/) | Fig. 6, Fig. 7 |

Each folder has its own README with the tables, the figures, the key findings as stated in the paper, and how to reproduce them. The headline results (from the abstract): the best configurations translate intents with a strict F1 of 0.943 and reject up to 191 of 195 invalid intents; with few-shot reasoning the strongest models hallucinate in about one response in a hundred; a translated intent costs 0.044 to 0.148 Wh of GPU energy without reasoning; and realistic rephrasing lowers the rejection of invalid intents in 15 of 16 configurations.

## Names used in the code and in the paper

The code and the result files use `zs3`, `fs3`, `zs3_cot`, `fs3_cot`; the paper and the READMEs of the research questions use the names in the right column.

| Code and result files | Paper | Meaning |
|---|---|---|
| `zs3` | `zs` | zero-shot prompt |
| `fs3` | `fs` | few-shot prompt (three invented examples) |
| `zs3_cot` (`zs3 CoT`) | `zs-cot` | zero-shot prompt with step-by-step reasoning |
| `fs3_cot` (`fs3 CoT`) | `fs-cot` | few-shot prompt with step-by-step reasoning |

## Three levels of reproducibility

| Tier | What you can do | What you need | Status |
|---|---|---|---|
| **A** | Rebuild the tables and figures of the paper from the released result tables; check each number of the paper text. | `pip install -r requirements-analysis.txt` (laptop) | **works now** |
| **B** | Re-score the model outputs, re-label hallucinations, rebuild the result tables from the raw measurements. | The camera-ready archive (model outputs, raw energy measurements, label files) | after the archive is released |
| **C** | Regenerate the model outputs. | GPUs and `requirements-generation.txt` (vLLM 0.31.0); see [`docs/implementation.md`](docs/implementation.md) | documented; not bit-for-bit reproducible |

Model outputs depend on the GPU, the driver and the batch size, so regenerated outputs can differ slightly from the released ones. The Llama 3.2 chat template also inserts the date of the run into the prompt.

## Quick start (tier A)

```bash
git clone https://github.com/ggck43/intent2slo.git
cd intent2slo
pip install -r requirements-analysis.txt

# rebuild the tables and figures of the paper
python rq1_performance/make_table_and_figure.py     # Table II, Fig. 2
python rq2_validation/make_figure.py                # Fig. 3
python rq3_hallucination/make_figure.py             # Fig. 4
python rq4_energy/make_figure.py                    # Fig. 5
python rq5_robustness/make_figures.py               # Fig. 6, Fig. 7

# check the numbers of the paper text against the data: open one notebook per research question
jupyter notebook rq1_performance/rq1.ipynb          # likewise rq2.ipynb ... rq5.ipynb
```

The rebuilt tables are identical to the paper's. The PNG figures can differ in a few pixels, because font rendering depends on the FreeType build of the plotting library.

Tests (they read the two dataset files from `data/raw/`, see [Data](#data); the checks that need the model outputs are skipped until the archive is present):

```bash
python -m pytest -q tests        # 49 passed, 4 skipped
```

## Data

- **Source dataset (not copied here).** The 716 placement intents (521 valid, 195 invalid), the node snapshot and the ground-truth SLOs come from the dataset of Islam and Calheiros, Zenodo [doi:10.5281/zenodo.20810799](https://doi.org/10.5281/zenodo.20810799). Download its two files into `data/raw/`; see [`data/README.md`](data/README.md). The code reads them from there.
- **Robustness benchmark.** [`data/robust/`](data/robust/) holds a **sample of 10 intents with 3 rephrasings each (30 rows)**. **The full benchmark of 2,148 variants is available on request: sabbir.ahmed@queensu.ca.**
- **Hallucination rulebook.** [`rq3_hallucination/RULEBOOK.md`](rq3_hallucination/RULEBOOK.md) defines what counts as a hallucinated SLO and how the labels are assigned.

## Repository layout

```
intent2slo/
├── README.md  CITATION.cff  LICENSE  LICENSE-DATA.txt  NOTICE
├── requirements-analysis.txt  requirements-generation.txt
├── rq1_performance/  rq2_validation/  rq3_hallucination/  rq4_energy/  rq5_robustness/
│     each: README.md, results/ (result tables), figures/, a notebook, a figure script
├── src/        scoring, parsing, prompts, analysis and labelling code   (src/README.md)
├── prompts/    the four prompts: zs3, zs3_cot, fs3, fs3_cot
├── scripts/    generation runner and launchers, hallucination labelling, energy measurement   (scripts/README.md)
├── tests/      unit tests
├── data/       dataset instructions, robustness sample
└── docs/       models.md (the ten models), implementation.md (settings and measurement details)
```

## What is not here yet

These are released with the camera-ready archive (a Zenodo record with a DOI that will be added here):

- the model outputs of all runs (the input of tier B for RQ1, RQ2, RQ3 and RQ5);
- the raw energy measurements of RQ4;
- the per-constraint hallucination label files;
- the human-evaluation material (samples, protocol and responses);
- the leak check of the few-shot prompt.

## How to cite

Please cite the paper and the source dataset (see [`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button uses it):

```bibtex
@misc{ahmed2026intent2slo,
  author = {Ahmed, Sabbir and Abbas, Hazem and Choudhury, Salimur},
  title  = {How Far Can Lightweight {LLMs} Go in Intent-to-{SLO} Translation?},
  year   = {2026},
  note   = {Manuscript. Link to be added.},
  url    = {https://github.com/ggck43/intent2slo}
}
```

- Dataset: Islam, K., Calheiros, R. N. *Natural-Language Intent Translation for Intent-Driven Orchestration in the Compute Continuum Dataset*, version 2, Zenodo. doi:10.5281/zenodo.20810799.
- Intent Engine: Islam, K., Calheiros, R. N. *Intent Engine: Natural-language Intent Translation for Intent-Driven Orchestration in the Compute Continuum*. Journal of Systems Architecture 179 (2026) 103938. doi:10.1016/j.sysarc.2026.103938.

## Licences

- **Code:** Apache License 2.0 ([`LICENSE`](LICENSE)).
- **Data and tables created in this project** (result tables, the robustness sample, documentation): CC BY 4.0 ([`LICENSE-DATA.txt`](LICENSE-DATA.txt)). Please credit the source dataset as well; its Zenodo record lists CC BY 4.0 and Apache 2.0.
- The models are used under their own licences; see [`docs/models.md`](docs/models.md).
- See [`NOTICE`](NOTICE) for the copyright statement.
