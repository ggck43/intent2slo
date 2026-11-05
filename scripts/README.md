# `scripts/`: what each file does

Run all scripts from the repository root. Scripts that generate model outputs need GPUs and the packages of `requirements-generation.txt`; the others only need the packages of `requirements-analysis.txt` and, for the label scripts, the model outputs of the camera-ready archive.

## Generation (RQ1 to RQ3 inputs)
| File | Purpose |
|---|---|
| `run_zs_cot.py` | The generation runner: reads the dataset and the snapshot, builds the prompts of one setup (`zs3`, `zs3_cot`, `fs3`, `fs3_cot`), runs one model with vLLM, and writes the run directory (predictions, token counts, run README). Only the four prompts in `prompts/` are supported; its command-line help also lists options for earlier prompt files that are not part of this repository. |
| `run_zs3_full.sh`, `run_fs3_full.sh` | Launch one model on the valid and the invalid intents with the `zs3` / `zs3_cot` or `fs3` / `fs3_cot` prompts (Qwen3.5 and Gemma 4). |
| `run_models_zs3_full.sh`, `run_models_fs3_full.sh` | The same for Ministral 3, Granite 4.1 and Llama 3. |

## Hallucination labels (RQ3)
| File | Purpose |
|---|---|
| `halluc_workload.py` | Helpers shared by the label scripts (finding the run directories, loading predictions). |
| `halluc_workload_v2.py` | Labels every returned constraint of the valid intents with the rules and writes `outputs/hallucination/policy_v2_constraints.csv`. |
| `judge_invalid_v2.py` | The per-intent specifications of the 35 invalid intent texts and the code that applies them to every returned constraint; writes `judge_invalid_v2.csv`. |
| `merge_judged_labels.py` | Puts the invalid-intent labels into the constraint file, `policy_v2j_constraints.csv`. |
| `halluc19_responses.py` | One row per response (28,640) with the number of hallucinated constraints; the input of `src/halluc19.py`. |

The rules are explained in `rq3_hallucination/RULEBOOK.md`.
