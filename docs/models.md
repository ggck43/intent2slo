# Evaluated models

Ten instruction-tuned language models from five families, two sizes per family. All have open weights and at most 10B parameters.

| Family | Model (Hugging Face id) | Parameters (stored) | Weights | Released |
|---|---|---:|---|---|
| Qwen3.5 | `Qwen/Qwen3.5-4B` | 4.66B | BF16 | Mar. 2026 |
| | `Qwen/Qwen3.5-9B` | 9.65B | BF16 | Mar. 2026 |
| Gemma 4 | `google/gemma-4-E2B-it` | 5.12B | BF16 | Mar. 2026 |
| | `google/gemma-4-E4B-it` | 8.00B | BF16 | Mar. 2026 |
| Ministral 3 | `mistralai/Ministral-3-3B-Instruct-2512` | 3.85B | FP8 | Dec. 2025 |
| | `mistralai/Ministral-3-8B-Instruct-2512` | 8.92B | FP8 | Dec. 2025 |
| Granite 4.1 | `ibm-granite/granite-4.1-3b` | 3.40B | BF16 | Apr. 2026 |
| | `ibm-granite/granite-4.1-8b` | 8.79B | BF16 | Apr. 2026 |
| Llama 3 | `meta-llama/Llama-3.2-3B-Instruct` | 3.21B | BF16 | Sep. 2024 |
| | `meta-llama/Llama-3.1-8B-Instruct` | 8.03B | BF16 | Jul. 2024 |

- **Parameters** are the stored totals from the Hugging Face model metadata (queried 2026-10-28), including vision encoders of multimodal models that were not used. The two Gemma 4 models are named by their *effective* size (E2B and E4B); their stored sizes are 5.12B and 8.00B.
- **Weights.** The Ministral 3 models are published with FP8 weights and were run as stored, on the same GPU as the others. Energy comparisons across families are therefore not like for like (RQ4).
- **Release months** come from the vendors' release logs and announcements.
- The Llama models are gated on Hugging Face (access has to be requested).

## Why these models
Three requirements guided the choice.
1. **Open weights.** A telemetry snapshot exposes the topology, load and capacity of the operator's infrastructure. An open model can be served on premises, so this data never leaves the operator's domain.
2. **At most 10B parameters**, so that the model fits on a single GPU at the edge or in an operator data center. This keeps the cost of the intent-processing function small next to the services it manages.
3. **Five vendors and recent releases**, so that a finding that holds across families is not a property of one training recipe. Two sizes per family show whether a larger model buys reliability.

All models are evaluated with their native reasoning ("thinking") mode disabled, because three of the five families do not offer one; reasoning is requested in the prompt instead (the `zs-cot` and `fs-cot` setups).

## Notes
- Granite 4.2 was published (2026-10-20) after the models were chosen. Its chat template switches thinking on by default, so Granite 4.1 was kept.
- The 500 synthetic intents of the dataset (of 716) were written by Llama-3.3-70B. The Llama models of this study are the only models from that vendor, so results for the Llama family should be read with this in mind. This was not tested.
- RQ4 (energy) covers the three families with the highest accuracy in RQ1: Qwen3.5, Gemma 4 and Ministral 3.

## Model sources
- Qwen3.5: Qwen Team, *Qwen3.5 Small Models*, https://huggingface.co/Qwen/Qwen3.5-9B (accessed 2026-10-28).
- Gemma 4: Gemma Team, Google DeepMind, *Gemma 4 Model Card*, https://huggingface.co/google/gemma-4-E4B-it (accessed 2026-10-28).
- Ministral 3: Liu, A. H. et al., *Ministral 3*, arXiv:2601.08584 (2026).
- Granite 4.1: IBM Granite Team, *Granite 4.1 Language Models*, https://huggingface.co/ibm-granite/granite-4.1-8b (accessed 2026-10-28).
- Llama 3: Grattafiori, A. et al., *The Llama 3 Herd of Models*, arXiv:2407.21783 (2024).
